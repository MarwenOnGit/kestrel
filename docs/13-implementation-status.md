# 13 — Implementation Status

**As of 2026-08-15.** What is built, what is measured, what is known to be
broken. Updated as work proceeds; do not let it drift from the code.

---

## 1. Where the plan stands

| Plan step (Doc 11) | Status |
|---|---|
| 0. Repo + message contracts | **Done** |
| 1. KestrelSim (detection-level) | **Done** |
| 2. Legal spine (token, gate, sealing, audit) | **Done** |
| 3. DSP chain (FFT/CFAR/channelize) | Not started — sim feeds detections directly |
| 4. Classifier (trained) | Stub only (`StubClassifier`, SNR-dependent confusion) |
| 5. Coarse geolocation | **Done** — batch MLE with systematic-error budget |
| 6. Autonomy agent | **Done** — behaviour tree, safety layer, info-gain planner |
| 7. Ground station | Partial — audit + fusion primitives; no web map yet |
| 8. Hardware bring-up | Blocked on hardware |

**44 tests pass, 1 xfail** (a recorded known limitation, `strict=True`, so it
fails the build if it starts unexpectedly passing).

Run everything: `python3 -m pytest tests/ -q` (~4 min, no hardware).

---

## 2. Measured results

All from `kestrel/sim`, 20 seeds unless noted. **Simulator numbers, not flight
numbers** — meaningless until validated against real measurements (Doc 09 §7).

### Geolocation, single isolated emitter
| Metric | Value |
|---|---|
| 95% ellipse containment | **93%** (CI 83–100%) against a nominal 95% |
| Median fix error | **4 m** |
| p95 fix error | 94 m |
| Tracks per emitter | 1.6 |

### Information-Gain Flight vs. baselines (`single_emitter`, 8 seeds)
| Planner | Median fix error |
|---|---|
| **infogain** | **453 m** |
| lawnmower | 1,640 m |
| random | 1,047 m |

A **3.6×** accuracy improvement from flight path alone, no extra hardware. This
is signature capability #1 doing what Doc 06 §4 claims.

### Legal invariants — every scenario, every seed
- `over_collection_count` = **0**
- `safety_violations` = **0**
- audit chain verifies after every sortie
- every recording maps to an authorized detection

---

## 3. Bugs found and fixed during implementation

Recorded because each one is a trap the next person would otherwise re-enter.

1. **Simulator truth leaked into the audit log.** `record_detection` stripped
   one truth field but not the two added later. Ground truth reaching a report
   would be indistinguishable from a measurement. Now uses `sanitized()`.

2. **Sequential EKF diverged.** Bearing-only EKF linearizes around its own
   estimate, so once it drifts it keeps confirming its error — 1,282 m off with
   a 97 m ellipse. Replaced with a periodic **batch maximum-likelihood solve**
   over all stored bearings, which re-linearizes from scratch.

3. **Information gain measured against the wrong baseline.** The planner scored
   candidates against the current position only, ignoring every observation
   already taken, so it oscillated instead of seeking new geometry. Now scores
   against the full observation history.

4. **The simulator was flattering us.** With only zero-mean independent errors,
   fixes converged to **1.3 m** — physically impossible for airborne bearing-only
   DF. Added the error sources that actually dominate: per-airframe calibration
   residual, azimuth-dependent airframe reflection, multipath outliers, and
   **time-correlated** GPS bias. (White-noise GPS also averaged away; real GPS
   error is correlated over minutes.)

5. **Covariance ignored systematic error.** A bias fits self-consistently and
   leaves almost no residual, so residual-based covariance stayed tiny while the
   fix was hundreds of metres off — containment collapsed to **3%**. Fixed by
   propagating a common bias through the Jacobian *and* flooring the ellipse at
   `range × calibration_sigma`. You cannot claim a 24 m fix at 1.7 km when your
   array is calibrated to 3°.

6. **Track fragmentation.** The association gate used random sigma only (<1° at
   high SNR), so a 3° systematic bias exceeded it constantly and shattered one
   emitter into ~5 phantom tracks, each confidently wrong. Gate now includes the
   systematic term, and track initiation requires **4 consecutive** gate
   failures so a single multipath bearing cannot spawn a track.

7. **Degenerate solves published as fixes.** Near-collinear bearings produced
   covariances up to 10^83 m² — arithmetically honest, operationally useless.
   Tracks now require real azimuth spread (≥20°) to be `resolved`; everything
   else is withheld rather than published.

8. **A "fix" that made things worse — reverted.** A covariance-gated track merge
   combined emitters 5 km apart and the joint solve diverged to 10^10 m. Removed,
   with a comment in `track.py` so it is not re-derived. A divergence guard was
   kept: any solve landing beyond 60 km is refused.

---

## 4. Known limitations (open items)

### 4.1 Multi-emitter data association — **the significant one**
In dense co-channel scenes (`border_crossing`), containment falls to ~62% and
the distant co-channel decoy is not reliably localized. Short early tracks
converge to wrong positions and get published as resolved.

*Why:* single-hypothesis nearest-neighbour association is not sufficient when
several emitters share a channel. *Fix:* multi-hypothesis association
(JPDA/MHT) plus TDOA from the swarm (Phase 2), which resolves ambiguity
geometrically instead of statistically. Recorded as a `strict` xfail so it
cannot be silently "fixed" by a threshold change.

### 4.2 Lawful users on authorized channels are collected
The farm-radio decoy transmits DMR inside the warrant's band and class, so its
traffic **is** recorded. That is the warrant working exactly as written, not a
gate failure — but it is the real-world minimization problem, and an argument
for writing narrower warrants. Fingerprinting (Phase 3) is what eventually
distinguishes devices on a shared channel.

### 4.3 Not yet real
- No DSP: detections come from the simulator, not from IQ.
- Classifier is a stub with a confusion model, not a trained network.
- Recorded IQ is a zero-filled placeholder — the sealing path is real, the
  contents are not.
- Single-aircraft geolocation only; no TDOA, no SADF, no swarm choreography.
- Ground station has no map yet.

### 4.4 Containment is conservative, not exact
93% measured against a nominal 95%, and the estimator errs toward large
ellipses. Conservative is the correct direction for this system, but it means
fixes are less precise than the hardware could support once calibration is
characterized properly on real data.

---

## 5. What to build next

1. **Step 3 — the DSP chain.** Replace simulated detections with FFT → CFAR →
   channelize over synthetic IQ. Biggest step toward reality.
2. **Step 7 — the ground map.** Cheap, and makes everything else demonstrable.
3. **Step 4 — the classifier.** Needs an IQ library; start recording early.
4. **Multi-hypothesis association** (§4.1) — or defer it to Phase 2 TDOA, which
   may make it moot. Decide deliberately rather than by drift.

Do **not** start Phase 2 hardware work until Steps 3, 4 and 7 are done in
simulation.

---

## 6. Honest caveat

Every number in §2 comes from a simulator whose error model I wrote, validated
against nothing but physics arguments. Doc 09 §7 exists for a reason: until
these figures are checked against a surveyed emitter and real hardware, treat
them as evidence the *software* works, not evidence the *system* works.
