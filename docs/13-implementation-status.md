# 13 — Implementation Status

**As of 2026-08-15 (Step 3 complete).** What is built, what is measured, what is known to be
broken. Updated as work proceeds; do not let it drift from the code.

---

## 1. Where the plan stands

| Plan step (Doc 11) | Status |
|---|---|
| 0. Repo + message contracts | **Done** |
| 1. KestrelSim (detection-level) | **Done** |
| 2. Legal spine (token, gate, sealing, audit) | **Done** |
| 3. DSP chain (FFT/CFAR/channelize) | **Done** — real IQ in, Detections out |
| 4. Classifier (trained) | Rule-based `FeatureClassifier` on real IQ; no trained network yet |
| 5. Coarse geolocation | **Done** — batch MLE with systematic-error budget |
| 6. Autonomy agent | **Done** — behaviour tree, safety layer, info-gain planner |
| 7. Ground station | Partial — audit + fusion primitives; no web map yet |
| 8. Hardware bring-up | Blocked on hardware |

**80 tests pass, 1 xfail** (a recorded known limitation, `strict=True`, so it
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
- Classifier is rule-based over hand-picked features, not a trained network.
  It abstains readily, which is safe but misses captures.
- The mission loop still uses the detection-level simulator; the IQ pipeline is
  exercised separately (`tests/test_pipeline.py`, `tools/dsp_demo.py`). Wiring
  the IQ path into `run_mission` is the next integration step.
- Recorded IQ in the mission loop is still a zero-filled placeholder — the
  sealing path is real, the contents are not.
- Single-aircraft geolocation only; no TDOA, no SADF, no swarm choreography.
- Ground station has no map yet.

### 4.4 Containment is conservative, not exact
93% measured against a nominal 95%, and the estimator errs toward large
ellipses. Conservative is the correct direction for this system, but it means
fixes are less precise than the hardware could support once calibration is
characterized properly on real data.

---

## 5. What to build next

1. **Wire the IQ pipeline into `run_mission`.** The DSP chain and the mission
   loop both work; they are not yet joined. This closes the last shortcut.
2. **Step 7 — the ground map.** Cheap, and makes everything else demonstrable.
3. **Step 4 — the trained classifier.** The IQ library now exists
   (`kestrel/sim/iq_synth.py` generates 12 classes), so training data is no
   longer blocked. Split by recording session, train across -5..+30 dB.
4. **Multi-hypothesis association** (§4.1) — or defer it to Phase 2 TDOA, which
   may make it moot. Decide deliberately rather than by drift.

Do **not** start Phase 2 hardware work until 1, 2 and 3 are done in simulation.

---

## 5a. Step 3 — the DSP chain (added 2026-08-15)

Real signal processing now replaces the geometry shortcut, at least in the
pipeline tests and the demo (`python3 tools/dsp_demo.py`):

    IQ -> Welch PSD -> noise floor -> CFAR + wideband -> channelize
       -> features -> classify -> Detection -> warrant gate

`kestrel/sim/iq_synth.py` synthesizes 12 signal classes as actual waveforms
(4FSK, pi/4-DQPSK, chirp, OFDM, GFSK, BPSK, FM, AM), composes them into a
wideband scene with link-budget-correct SNRs, and the pipeline is told nothing
about what is present. On the demo scene it finds all three in-band emitters
within 200 Hz, classifies each correctly, and the gate authorizes exactly the
two the warrant covers.

**Five more bugs found by measurement, continuing the list in §3:**

9.  **CFAR self-masking.** A 50 kHz signal is 205 bins wide, so CFAR's training
    cells landed *inside* it, the local background rose with the signal, and it
    became invisible to its own detector. CFAR cannot see anything wider than
    its training window. Fixed by running a global threshold against the robust
    noise floor alongside CFAR and taking the union — neither detector is
    sufficient alone.

10. **Bandwidth understated 6x.** Detection flags only the strong middle of a
    signal, so a 12.5 kHz DMR carrier was reported as 2 kHz. Now measured as
    the conventional 99%-power occupied bandwidth, with the frequency taken
    from the power centroid rather than the span midpoint.

11. **Duplicate candidates.** Two detectors firing on one signal expanded to
    identical spans; deduplication ran before expansion instead of after, so
    one emitter was reported three times.

12. **Hanning sidelobes.** At -31 dB, one strong emitter smeared across the
    band and masked weak neighbours — the exact failure this system exists to
    avoid. Switched to Blackman: wider main lobe, far lower sidelobes.
    Dynamic range beats frequency resolution in surveillance.

13. **No radio horizon.** Free-space loss alone gives a 5 W handheld at 300 km
    an SNR of +11 dB, so the simulator happily detected emitters from the far
    side of the country. Added a 4/3-earth horizon (about 51 km at 120 m
    altitude) with diffraction loss beyond it. This governs how much ground one
    aircraft can cover and had been silently missing since day one.

**And two synthesizer faults, which matter because a simulator that emits
signals no radio would produce tests the DSP against fiction:**

- OFDM symbols butted together left phase discontinuities that splattered
  across the whole span and lifted the measured floor by 10 dB. Real
  transmitters window their symbol edges; now so does this one.
- The interpolation filter's -75 dBc stopband left images every 50 kHz, and
  against a 97 dB signal the detector dutifully reported nine emitters that did
  not exist. Steepened to a Kaiser-16 filter.

**A useful negative result:** the LoRa-vs-satcom discriminator (how linearly
instantaneous frequency ramps) separates cleanly at 40 dB SNR (0.10 vs 1.41)
and degrades to uselessness by 8 dB (1.17 vs 1.46). The threshold is therefore
SNR-dependent and the classifier abstains below 10 dB. This is what Doc 05's
"measure accuracy per SNR bucket, not overall" looks like in practice, and it
is the concrete argument for a trained model over hand-written rules.

---

## 6. Honest caveat

Every number in §2 comes from a simulator whose error model I wrote, validated
against nothing but physics arguments. Doc 09 §7 exists for a reason: until
these figures are checked against a surveyed emitter and real hardware, treat
them as evidence the *software* works, not evidence the *system* works.
