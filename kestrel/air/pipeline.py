"""The real edge pipeline: wideband IQ in, Detections out (Doc 01 §4).

    IQ -> PSD -> CFAR/wideband -> channelize -> features -> classify -> Detection

This is what runs on the Jetson. The only thing separating it from flight code
is where the IQ comes from: `SimulatedSdrSource` composes a scene from the
world model, `UhdSdrSource` reads a USRP. Everything downstream is identical,
which is the entire point of building it this way.

Note what this module does NOT do: it does not decide what may be recorded.
It produces Detections; the warrant gate alone decides their fate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kestrel.air.detector import (EDGE_GUARD_FRAC, channelize, features,
                                  find_candidates)
from kestrel.common.messages import Detection


@dataclass
class SweepSegment:
    """One tuning position: where to look and how wide."""
    center_hz: float
    rate_sps: float
    dwell_s: float = 0.2

    @property
    def lo_hz(self) -> float:
        return self.center_hz - self.rate_sps / 2

    @property
    def hi_hz(self) -> float:
        return self.center_hz + self.rate_sps / 2

    def contains(self, freq_hz: float) -> bool:
        return self.lo_hz <= freq_hz <= self.hi_hz


class SweepPlan:
    """A set of tuning positions covering the mission bands.

    Revisit time is the metric that matters: an emitter transmitting for four
    seconds every ten minutes is invisible if you return to its segment less
    often than it keys up (Doc 04 §7).
    """

    def __init__(self, segments: list[SweepSegment], n_aircraft: int = 1):
        self.segments = segments
        self.n_aircraft = max(n_aircraft, 1)
        self._i = 0

    @classmethod
    def from_bands(cls, bands: list[tuple[float, float]], rate_sps: float = 20e6,
                   dwell_s: float = 0.2, n_aircraft: int = 1,
                   overlap_frac: float = 0.08) -> "SweepPlan":
        """Tile the bands with overlapping segments.

        The overlap is not waste. Abutting segments put a hard seam in the
        middle of the spectrum, and an emitter that lands on one sits at the
        edge of both windows - exactly where the front end rolls off and where
        energy past Nyquist folds back, so it is detected as an artifact at the
        far edge of the band or missed altogether.

        How much overlap is enough: the worst case is a frequency exactly
        between two segment centres, at `rate x (1 - overlap) / 2`. For that to
        stay inside the detector's trustworthy middle (|offset| < 0.48 x rate,
        given `EDGE_GUARD_FRAC` = 2%) the overlap must exceed 4%. The default
        doubles that, which costs ~8% more segments - and revisit time is what
        that buys, so it is stated rather than buried.

        The band's own two ends need the same treatment: tiling naively from
        `lo + rate/2` leaves the first frequency of the band sitting exactly on
        the first window's edge, permanently inside the guard. The first centre
        is therefore pulled down so that `lo` starts out well inside it.
        """
        segs = []
        step = rate_sps * (1.0 - max(overlap_frac, 0.0))
        usable_half = rate_sps * (0.5 - EDGE_GUARD_FRAC)
        for lo, hi in bands:
            f = lo + 0.95 * usable_half
            while True:
                segs.append(SweepSegment(f, rate_sps, dwell_s))
                if f + 0.95 * usable_half >= hi:
                    break
                f += step
        return cls(segs, n_aircraft=n_aircraft)

    def revisit_time_s(self) -> float:
        return sum(s.dwell_s for s in self.segments) / self.n_aircraft

    def sector_for(self, index: int) -> list[SweepSegment]:
        """Interleaved division (Doc 06 §5): losing an aircraft degrades
        revisit time everywhere slightly, rather than going blind on one band."""
        return self.segments[index::self.n_aircraft]

    def next_segment(self, sector: list[SweepSegment] | None = None) -> SweepSegment:
        segs = sector if sector else self.segments
        seg = segs[self._i % len(segs)]
        self._i += 1
        return seg


# --------------------------------------------------------------------------
class EdgePipeline:
    def __init__(self, classifier, aircraft_id: str = "K1",
                 threshold_db: float = 8.0, nfft: int = 4096,
                 min_snr_db: float = 5.0, max_candidates_per_frame: int = 32):
        self.classifier = classifier
        self.aircraft_id = aircraft_id
        self.threshold_db = threshold_db
        self.nfft = nfft
        self.min_snr_db = min_snr_db
        self.max_candidates = max_candidates_per_frame

        # Health counters. These are the numbers that tell you whether the
        # funnel is behaving in flight; log them every sortie.
        self.frames = 0
        self.candidates_seen = 0
        self.candidates_dropped_budget = 0

    # ------------------------------------------------------------------
    def process_frame(self, iq: np.ndarray, segment: SweepSegment,
                      t_gpsdo_ns: int, aircraft_pos_xy, aircraft_alt_m: float = 120.0,
                      bearing_fn=None) -> list[tuple[Detection, np.ndarray]]:
        """One IQ frame -> classified Detections, each with its IQ snippet.

        The snippet travels with the Detection so the warrant gate can authorize
        capture of exactly the signal that was detected - never the wideband
        stream, which would collect everything on the air.
        """
        self.frames += 1
        cands = find_candidates(iq, segment.rate_sps, segment.center_hz,
                                nfft=self.nfft, threshold_db=self.threshold_db)
        self.candidates_seen += len(cands)

        # Strongest first, then cap. Under a burst of activity the GPU budget
        # is finite; dropping the weakest is better than falling behind and
        # losing whole frames.
        cands.sort(key=lambda c: -c.snr_db)
        if len(cands) > self.max_candidates:
            self.candidates_dropped_budget += len(cands) - self.max_candidates
            cands = cands[:self.max_candidates]

        out = []
        for c in cands:
            if c.snr_db < self.min_snr_db:
                continue
            snippet, snip_fs = channelize(iq, segment.rate_sps, segment.center_hz,
                                          c.freq_hz, c.bandwidth_hz)
            det = Detection(
                aircraft_id=self.aircraft_id,
                t_gpsdo_ns=t_gpsdo_ns,
                freq_hz=c.freq_hz,
                bandwidth_hz=c.bandwidth_hz,
                snr_db=c.snr_db,
                aircraft_pos_xy=(float(aircraft_pos_xy[0]), float(aircraft_pos_xy[1])),
                aircraft_alt_m=aircraft_alt_m,
                # Measured from the snippet, never assumed: this is what the
                # warrant is charged for.
                capture_duration_s=len(snippet) / max(snip_fs, 1.0),
            )
            if bearing_fn is not None:
                b = bearing_fn(c)
                if b is not None:
                    det.bearing_deg, det.bearing_sigma_deg = b
            out.append((det, snippet))
        return out

    # ------------------------------------------------------------------
    def health(self) -> dict:
        return {
            "frames": self.frames,
            "candidates_seen": self.candidates_seen,
            "candidates_per_frame": self.candidates_seen / max(self.frames, 1),
            "dropped_over_budget": self.candidates_dropped_budget,
        }


# --------------------------------------------------------------------------
class FeatureClassifier:
    """Rule-based classifier over the cheap features (Doc 04 §6).

    Not the trained network - that is Step 4 - but not a stub either: it makes
    a real decision from real IQ, so the pipeline can be exercised end to end
    against synthesized waveforms before any model exists. It abstains
    (`unknown`) readily, which is the safe direction: unknown is never
    authorized for capture.
    """

    model_version = "features-0.1"

    @staticmethod
    def _chirp_threshold(snr_db: float) -> float | None:
        """SNR-dependent decision boundary, read off measured separation.
        None means 'the feature no longer discriminates - abstain'."""
        if snr_db >= 25.0:
            return 0.55
        if snr_db >= 15.0:
            return 0.85
        if snr_db >= 10.0:
            return 1.15
        return None

    def classify(self, det: Detection, iq: np.ndarray | None = None
                 ) -> tuple[str, float]:
        if iq is None or len(iq) < 64:
            return "unknown", 0.0
        f = features(iq, max(det.bandwidth_hz * 4.0, 24_000.0))
        if not f:
            return "unknown", 0.0

        flat = f["spectral_flatness"]
        amp_var = f["amp_var"]
        papr = f["papr_db"]
        bw = det.bandwidth_hz

        # OFDM family: noise-like envelope, high PAPR, wide.
        if amp_var > 0.25 and papr > 8.0 and bw > 1e6:
            if bw > 15e6:
                return "nr5g", 0.88
            return ("wifi", 0.86) if bw > 8e6 else ("lte", 0.86)

        # Constant-envelope narrowband: the voice modes.
        if amp_var < 0.15 and bw < 60e3:
            if f["freq_var"] > 1e6:
                return "dmr", 0.87
            return "fm_voice", 0.86

        # LoRa before satcom: both are constant-envelope and overlap in
        # bandwidth, so amplitude cannot separate them. What distinguishes a
        # chirp is that its instantaneous frequency sweeps the whole channel -
        # freq_var is orders of magnitude higher than any fixed-carrier mode.
        if 60e3 <= bw <= 400e3 and amp_var < 0.35:
            # A chirp's instantaneous frequency ramps linearly, so its
            # derivative is nearly constant. Bandwidth and amplitude cannot
            # separate LoRa from a narrowband BPSK terminal; this can.
            #
            # But the feature is a second derivative of phase, so noise wrecks
            # it: measured LoRa/satcom separation is 0.10/1.41 at 40 dB SNR,
            # 0.86/1.42 at 12 dB, and gone by 8 dB. The threshold therefore
            # tracks SNR, and below 10 dB the honest answer is `unknown` -
            # which the warrant gate refuses, so abstaining costs nothing but
            # a missed capture, while guessing could authorize the wrong one.
            thr = self._chirp_threshold(det.snr_db)
            if thr is None:
                return "unknown", 0.4
            return ("lora", 0.87) if f["freq_ramp_ratio"] < thr else \
                   ("satcom_terminal", 0.86)

        return "unknown", 0.4
