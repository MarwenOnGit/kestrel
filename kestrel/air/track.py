"""Emitter belief state (Doc 06 §6).

An Extended Kalman Filter over emitter position, updated by bearing
observations. The covariance P *is* the error ellipse, which is why tracking
and information-gain planning are the same mathematics viewed from two ends.
"""
from __future__ import annotations

import math

import numpy as np

from kestrel.common.messages import Detection, ErrorEllipse

# Beyond this, the fix is not useful at operational scale: the emitter could be
# anywhere in the area of operations. Report "unresolved" instead of a number.
MAX_USEFUL_SIGMA_M = 5000.0

# Bearings spanning less than this carry almost no range information. Below it,
# a fix is reported as unresolved rather than published with a covariance that
# understates the truth.
MIN_AZIMUTH_SPREAD_DEG = 20.0

# A solve landing further than this from the observers is not a fix, it is a
# diverged optimisation. Emitters we care about are within radio horizon.
MAX_SOLUTION_RANGE_M = 60_000.0

# chi-square critical values, 2 degrees of freedom
CHI2 = {0.95: 5.991, 0.90: 4.605, 0.50: 1.386}


class EmitterTrack:
    def __init__(self, track_id: str, freq_hz: float, signal_class: str,
                 initial_xy=(0.0, 0.0), initial_sigma_m: float = 5000.0):
        self.track_id = track_id
        self.freq_hz = freq_hz
        self.signal_class = signal_class
        self.xy = np.asarray(initial_xy, dtype=float)
        self.P = np.eye(2) * initial_sigma_m ** 2
        self.n_observations = 0
        self.detections: list[str] = []
        self.last_t_ns: int = 0
        self.truth_xy = None          # simulator only, for metrics
        self._bootstrapped = False
        self.resolved = False
        self.azimuth_spread_deg = 0.0

        # Observation geometry history. The planner needs this: information
        # gain is meaningful only against everything already measured, not
        # against the single most recent position.
        self.obs_positions: list[tuple[float, float]] = []
        self.obs_sigmas_deg: list[float] = []
        self.obs_bearings_deg: list[float] = []
        self.resolve_every = 5
        self._dirty = True

        # Process noise. A bearing-only EKF is a classic over-confident filter:
        # linearization error accumulates while the covariance keeps shrinking,
        # so it ends up certain and wrong. A small Q per update keeps the
        # reported ellipse honest, which the containment test enforces.
        self.q_per_update_m2 = 400.0

        # Residual array calibration accuracy, from the ground calibration
        # procedure in Doc 10 §3 ("< 3 deg RMS"). This is a property of the
        # HARDWARE, not of the data, and it sets the floor on how well any
        # amount of observation can locate an emitter. Update it from the
        # aircraft's calibration record; degrade it as calibration ages.
        self.bearing_bias_sigma_deg = 3.0

    # ------------------------------------------------------------------
    def update_bearing(self, obs_xy, bearing_deg: float, sigma_deg: float) -> None:
        obs = np.asarray(obs_xy, dtype=float)
        sigma = math.radians(max(sigma_deg, 0.1))

        if not self._bootstrapped:
            # Cold start. One bearing constrains DIRECTION well and RANGE not
            # at all, so an isotropic covariance would be a lie in both
            # directions at once. Seed an ellipse elongated along the bearing:
            # tight across it (r * sigma), enormous along it.
            self._bootstrap(obs, bearing_deg, sigma)

        dx, dy = self.xy[0] - obs[0], self.xy[1] - obs[1]
        r2 = dx * dx + dy * dy
        if r2 < 1.0:
            return
        h = math.atan2(dy, dx)
        H = np.array([[-dy / r2, dx / r2]])
        R = np.array([[sigma ** 2]])

        # Inflate before updating: accounts for linearization error and for a
        # target that may not be perfectly stationary.
        self.P = self.P + np.eye(2) * self.q_per_update_m2

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        innov = math.atan2(math.sin(math.radians(bearing_deg) - h),
                           math.cos(math.radians(bearing_deg) - h))
        self.xy = self.xy + (K @ np.array([innov])).ravel()

        # Joseph form: numerically stable and stays positive-definite even when
        # the gain is large, unlike the (I-KH)P shortcut.
        A = np.eye(2) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)          # keep it symmetric

        self.obs_positions.append((float(obs[0]), float(obs[1])))
        self.obs_sigmas_deg.append(float(sigma_deg))
        self.obs_bearings_deg.append(float(bearing_deg))
        self.n_observations += 1

        # Periodically re-derive the estimate from scratch. The EKF keeps the
        # belief fresh between solves; the batch solve stops it drifting.
        self._dirty = True
        if self.n_observations % self.resolve_every == 0:
            self.solve_batch()

    # ------------------------------------------------------------------
    def _bootstrap(self, obs: np.ndarray, bearing_deg: float,
                   sigma_rad: float, nominal_range_m: float = 2000.0,
                   range_sigma_m: float = 2500.0) -> None:
        b = math.radians(bearing_deg)
        u = np.array([math.cos(b), math.sin(b)])          # along the bearing
        v = np.array([-u[1], u[0]])                       # across it
        self.xy = obs + u * nominal_range_m

        along = range_sigma_m ** 2
        across = max((nominal_range_m * sigma_rad) ** 2, 100.0)
        R_rot = np.column_stack([u, v])
        self.P = R_rot @ np.diag([along, across]) @ R_rot.T
        self._bootstrapped = True

    # ------------------------------------------------------------------
    def update_from_detection(self, det: Detection) -> None:
        if det.bearing_deg is None:
            return
        self.update_bearing(det.aircraft_pos_xy, det.bearing_deg,
                            det.bearing_sigma_deg or 5.0)
        self.detections.append(det.detection_id)
        self.last_t_ns = max(self.last_t_ns, det.t_gpsdo_ns)
        if det.truth_xy is not None:
            self.truth_xy = det.truth_xy

    # ------------------------------------------------------------------
    def solve_batch(self, max_obs: int = 200) -> bool:
        """Re-estimate position from ALL stored bearings at once.

        A sequential EKF linearizes around its current estimate, so once it
        drifts it keeps confirming its own error - it ends up confident and
        wrong, especially when the observation baseline is short. A batch
        solve re-linearizes from scratch every time and cannot lock in that
        way; the covariance from J^T W J is the Cramer-Rao bound, which is
        what makes the reported ellipse honest.

        Robust loss handles multipath: one badly wrong bearing must not drag
        the fix hundreds of metres (Doc 07 §5).
        """
        from scipy.optimize import least_squares

        n = len(self.obs_positions)
        if n < 3:
            return False

        idx = (np.arange(n) if n <= max_obs
               else np.unique(np.linspace(0, n - 1, max_obs).astype(int)))
        P_obs = np.asarray(self.obs_positions, dtype=float)[idx]
        sig = np.radians(np.asarray(self.obs_sigmas_deg, dtype=float)[idx])
        brg = np.radians(np.asarray(self.obs_bearings_deg, dtype=float)[idx])

        # Need real angular spread, or range is unobservable and the solve is
        # meaningless however many observations we have.
        if float(np.ptp(np.unwrap(brg))) < math.radians(2.0):
            return False

        # Weight by the TOTAL per-observation uncertainty. Using the random
        # part alone would weight high-SNR bearings as if calibration were
        # perfect, which it never is.
        sig_eff = np.sqrt(sig ** 2 + math.radians(self.bearing_bias_sigma_deg) ** 2)

        def residuals(xy):
            d = xy - P_obs
            pred = np.arctan2(d[:, 1], d[:, 0])
            return np.arctan2(np.sin(pred - brg), np.cos(pred - brg)) / sig_eff

        try:
            sol = least_squares(residuals, self.xy.copy(), loss="soft_l1",
                                f_scale=3.0, max_nfev=100)
        except Exception:
            return False
        if not sol.success:
            return False

        J = sol.jac
        JtJ = J.T @ J
        try:
            cov = np.linalg.inv(JtJ)
        except np.linalg.LinAlgError:
            return False
        if not np.all(np.isfinite(cov)):
            return False

        # Scale by the observed residual spread: if bearings disagree more than
        # our sigma model claims, the ellipse must grow to match reality.
        #
        # Computed from the ACTUAL residuals, not from sol.cost: the robust
        # loss deliberately down-weights large residuals, which is right for
        # estimating position but wrong for estimating uncertainty - it would
        # discount exactly the misfit that reveals systematic error. Trim the
        # worst 5% so a multipath outlier cannot inflate the ellipse on its own,
        # then let everything else speak.
        res = np.asarray(residuals(sol.x), dtype=float)
        dof = max(len(idx) - 2, 1)
        if len(res) > 8:
            keep = res[np.abs(res) <= np.percentile(np.abs(res), 95)]
        else:
            keep = res
        chi2_red = float(np.sum(keep ** 2) / max(len(keep) - 2, 1))
        cov = cov * max(chi2_red, 1.0)

        # SYSTEMATIC uncertainty. The covariance above describes random scatter
        # only, and shrinks like 1/n. But the dominant real error - residual
        # array calibration and airframe reflection - is a bias COMMON to every
        # bearing from this aircraft. It fits self-consistently, so it leaves
        # almost no residual, and averaging cannot remove it: taking a thousand
        # bearings through a 3-degree calibration error yields a precise fix in
        # the wrong place.
        #
        # A shared bias b displaces the solution by A*b, where
        #   A = (J^T J)^-1 J^T (dr/db),   dr/db = -1/sigma  per observation.
        # So its contribution is sigma_bias^2 * A A^T. This term does not
        # shrink with n, which is exactly the point - and it is what makes the
        # reported ellipse contain the truth.
        # Two mechanisms, both needed:
        #
        # (a) A bias COMMON to every bearing displaces the solution by A*b,
        #     where A = (J^T J)^-1 J^T (dr/db) and dr/db = -1/sigma. Its
        #     contribution is sigma_bias^2 * A A^T, and it does not shrink
        #     with n - which is the whole point.
        try:
            drdb = -1.0 / sig_eff
            A = cov @ (J.T @ drdb)
            cov = cov + (math.radians(self.bearing_bias_sigma_deg) ** 2) * np.outer(A, A)
        except (ValueError, np.linalg.LinAlgError):
            pass

        # (b) A floor on how well ANY number of bearings can place an emitter.
        #     Systematic error that varies with azimuth (airframe reflection)
        #     is neither common-mode nor zero-mean-independent, so neither the
        #     term above nor 1/sqrt(n) averaging removes it. At range r, a
        #     calibration good to sigma_b radians simply cannot locate better
        #     than about r * sigma_b, and reporting a tighter ellipse than that
        #     is a claim the hardware cannot support.
        r_mean = float(np.mean(np.linalg.norm(self.xy - P_obs, axis=1)))
        floor_sigma_m = r_mean * math.radians(self.bearing_bias_sigma_deg)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, floor_sigma_m ** 2)
        cov = vecs @ np.diag(vals) @ vecs.T

        # Divergence guard. A bearing set that is inconsistent - two emitters
        # mistakenly associated, or severe multipath - can send the optimiser
        # to a solution thousands of kilometres away. Such a result is not a
        # fix, however confident the arithmetic looks, so keep the previous
        # estimate and declare the track unresolved.
        solution = np.asarray(sol.x, dtype=float)
        if (not np.all(np.isfinite(solution))
                or float(np.max(np.abs(solution - P_obs.mean(axis=0))))
                > MAX_SOLUTION_RANGE_M):
            self.resolved = False
            return False

        self.xy = solution
        self.P = 0.5 * (cov + cov.T)
        self._dirty = False

        # An ill-conditioned solve (near-collinear bearings) yields a covariance
        # that is arithmetically honest and operationally meaningless - tens of
        # kilometres, sometimes astronomically more. Range is simply not
        # observable from that geometry. Say so, rather than publishing a fix we
        # cannot defend: clamp the ellipse and mark the track unresolved.
        # Observability, not just arithmetic. Bearings spanning only a few
        # degrees pin down direction but say almost nothing about RANGE, and a
        # covariance computed from them understates the true error because the
        # problem is nearly degenerate along that axis. Demand real angular
        # spread before calling a fix resolved - otherwise the honest answer is
        # "we know roughly which way, not how far".
        azimuth_spread_deg = math.degrees(float(np.ptp(np.unwrap(np.sort(brg)))))
        sigmas = np.sqrt(np.clip(np.diag(self.P), 0.0, None))
        self.resolved = bool(np.all(np.isfinite(sigmas))
                             and np.max(sigmas) <= MAX_USEFUL_SIGMA_M
                             and azimuth_spread_deg >= MIN_AZIMUTH_SPREAD_DEG
                             and len(idx) >= 5)
        self.azimuth_spread_deg = azimuth_spread_deg
        if not self.resolved:
            self.P = np.eye(2) * MAX_USEFUL_SIGMA_M ** 2
        return self.resolved

    # ------------------------------------------------------------------
    def _ensure_solved(self) -> None:
        """Never report an EKF-only covariance: it is the overconfident one."""
        if self._dirty and len(self.obs_positions) >= 3:
            self.solve_batch()
            self._dirty = False

    def error_ellipse(self, confidence: float = 0.95) -> ErrorEllipse:
        self._ensure_solved()
        vals, vecs = np.linalg.eigh(self.P)
        vals = np.maximum(vals, 1e-9)
        k = CHI2.get(confidence, 5.991)
        major_idx = int(np.argmax(vals))
        minor_idx = 1 - major_idx
        return ErrorEllipse(
            semi_major_m=float(math.sqrt(k * vals[major_idx])),
            semi_minor_m=float(math.sqrt(k * vals[minor_idx])),
            orientation_deg=float(math.degrees(
                math.atan2(vecs[1, major_idx], vecs[0, major_idx]))) % 180.0,
            confidence=confidence,
        )

    def ellipse_area_m2(self, confidence: float = 0.95) -> float:
        return self.error_ellipse(confidence).area_m2()

    def contains(self, xy, confidence: float = 0.95) -> bool:
        """Is a point inside the confidence ellipse? Used to validate honesty."""
        self._ensure_solved()
        d = np.asarray(xy, dtype=float) - self.xy
        try:
            m = float(d @ np.linalg.inv(self.P) @ d)
        except np.linalg.LinAlgError:
            return False
        return m <= CHI2.get(confidence, 5.991)


class TrackStore:
    """Associates detections into tracks by frequency proximity.

    Phase 1 keys on frequency alone; two emitters sharing a channel are
    separated by position gating (Doc 11 Step 5.3).
    """

    def __init__(self, freq_tolerance_hz: float = 12_500.0,
                 split_distance_m: float = 1500.0,
                 bias_sigma_deg: float = 3.0,
                 min_gate_deg: float = 15.0):
        self.tracks: dict[str, EmitterTrack] = {}
        self.freq_tol = freq_tolerance_hz
        self.split_distance_m = split_distance_m
        self.bias_sigma_deg = bias_sigma_deg
        self.min_gate_deg = min_gate_deg
        self.confirm_n = 4          # consecutive gate failures before initiating
        self._miss_streak: dict[int, int] = {}
        self._next = 0

    def _new_track(self, det: Detection) -> EmitterTrack:
        self._next += 1
        t = EmitterTrack(f"E-{self._next:04d}", det.freq_hz, det.signal_class)
        self.tracks[t.track_id] = t
        return t

    def associate(self, det: Detection) -> EmitterTrack:
        candidates = [t for t in self.tracks.values()
                      if abs(t.freq_hz - det.freq_hz) <= self.freq_tol]
        if not candidates:
            return self._new_track(det)

        # Prefer a track whose current estimate is consistent with this bearing.
        best, best_err = None, None
        for t in candidates:
            if not t._bootstrapped or det.bearing_deg is None:
                return t
            dx = t.xy[0] - det.aircraft_pos_xy[0]
            dy = t.xy[1] - det.aircraft_pos_xy[1]
            pred = math.degrees(math.atan2(dy, dx)) % 360.0
            err = abs((pred - det.bearing_deg + 180) % 360 - 180)
            if best_err is None or err < best_err:
                best, best_err = t, err

        # The gate must include the SYSTEMATIC bearing error, not just the
        # random part. At high SNR the random sigma falls below a degree, while
        # calibration and airframe bias remain at a few degrees - so a gate
        # built from random sigma alone rejects perfectly good associations and
        # shatters one emitter into many phantom tracks, each converging
        # somewhere wrong with a confident little ellipse. That failure is far
        # worse than occasionally merging two real emitters, which shows up
        # honestly as a large ellipse and poor residuals.
        sigma_eff = math.hypot(det.bearing_sigma_deg or 5.0, self.bias_sigma_deg)
        gate_deg = max(self.min_gate_deg, 3.0 * sigma_eff)

        # Only split off a new track if the incumbent is actually well
        # determined. An immature track has a poor position estimate, so its
        # predicted bearing is not trustworthy enough to justify a split.
        if best_err is not None and best_err > gate_deg and best.resolved:
            # Track initiation requires CONFIRMATION. A single bearing that
            # fails the gate is far more likely to be multipath - a clean,
            # confident reflection off terrain - than a new emitter that
            # happens to share a channel. Spawning a track per outlier
            # manufactures phantom emitters with tight, wrong ellipses.
            # Demand several consecutive failures before believing it.
            key = self._freq_key(det.freq_hz)
            self._miss_streak[key] = self._miss_streak.get(key, 0) + 1
            if self._miss_streak[key] >= self.confirm_n:
                self._miss_streak[key] = 0
                return self._new_track(det)
            return None                      # outlier: counted, not tracked

        self._miss_streak[self._freq_key(det.freq_hz)] = 0
        return best

    def _freq_key(self, freq_hz: float) -> int:
        return int(freq_hz // max(self.freq_tol, 1.0))

    def update(self, det: Detection) -> EmitterTrack | None:
        t = self.associate(det)
        if t is None:                # rejected as an outlier
            return None
        t.update_from_detection(det)
        return t

    # ------------------------------------------------------------------
    # NOTE: a covariance-gated track merge was implemented here and REMOVED
    # after measurement. It merged tracks belonging to emitters 5 km apart -
    # their pre-merge covariances were poor enough to overlap - and the
    # combined batch solve, now fitting bearings to two different transmitters,
    # diverged to absurd positions. Proper duplicate suppression needs
    # multi-hypothesis association, not a distance gate. Kept as a comment so
    # the next person does not re-derive the same mistake.
    def _merge_consistent_DISABLED(self, t: EmitterTrack) -> None:
        """Disabled - see note above."""
        if not t.resolved:
            return
        for other in list(self.tracks.values()):
            if other is t or other.track_id not in self.tracks:
                continue
            if abs(other.freq_hz - t.freq_hz) > self.freq_tol:
                continue
            if not other.resolved:
                continue
            d = t.xy - other.xy
            S = t.P + other.P
            try:
                m2 = float(d @ np.linalg.inv(S) @ d)
            except np.linalg.LinAlgError:
                continue
            if m2 > CHI2[0.95]:
                continue                       # genuinely different places

            keep, drop = (t, other) if t.n_observations >= other.n_observations else (other, t)
            keep.obs_positions += drop.obs_positions
            keep.obs_sigmas_deg += drop.obs_sigmas_deg
            keep.obs_bearings_deg += drop.obs_bearings_deg
            keep.detections += drop.detections
            keep.n_observations += drop.n_observations
            if keep.truth_xy is None:
                keep.truth_xy = drop.truth_xy
            self.tracks.pop(drop.track_id, None)
            keep._dirty = True
            keep.solve_batch()
            if drop is t:
                return

    def best(self) -> EmitterTrack | None:
        """Most uncertain track that has been seen at least twice - the one
        most worth refining."""
        ready = [t for t in self.tracks.values() if t.n_observations >= 2]
        if not ready:
            return None
        return max(ready, key=lambda t: t.ellipse_area_m2())
