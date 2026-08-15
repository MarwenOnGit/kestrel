# 11 — Phase 1 Implementation Plan

**Goal:** one aircraft that autonomously surveys, detects, classifies,
geolocates coarsely, records only what a signed warrant permits, and reports to
a live map — with a verifiable audit trail.

**Written to be executed by an engineer or a coding model without further
design decisions.** Every task states what to build, where, and how you know
it's done. Work top to bottom; tasks within a step can be parallelized.

**Rule for the whole plan: TDD.** Write the test first, watch it fail, then make
it pass. The tests are specified in each task.

---

## Step 0 — Repository and environment (½ day)

**0.1** Create the layout from Doc 03 §1.
```bash
cd /home/dexter/drone-project
mkdir -p schemas tests tools kestrel/{air,ground,models,sim,common} kestrel/ground/webmap
git init && git add -A && git commit -m "Initial project structure"
```

**0.2** Python environment per Doc 03 §2. Add `pyproject.toml` with `ruff` +
`pytest` config.

**0.3** Write the JSON schemas from Doc 01 §9 into `schemas/`:
`detection.schema.json`, `warrant_token.schema.json`, `geofix.schema.json`,
`recording_sidecar.schema.json`.

**0.4** `kestrel/common/messages.py` — dataclasses matching those schemas, with
`to_json`/`from_json` and validation.
✅ *Done when:* `pytest tests/test_messages.py` passes round-trip and rejects
malformed input.

---

## Step 1 — KestrelSim, detection-level (3–4 days)

Build the simulator **before** anything else (Doc 09).

**1.1** `kestrel/sim/world.py` — `Emitter`, `Aircraft`, `path_loss_db`,
`KestrelSim` exactly as in Doc 09 §3.
✅ *Test:* an emitter at 1 km with 37 dBm EIRP produces the SNR predicted by the
link budget ±1 dB.

**1.2** `kestrel/sim/scenarios/` — write `border_crossing.yaml` (Doc 09 §4) plus
`decoy_heavy.yaml` and `warrant_expiry_midflight.yaml`.
✅ *Test:* scenarios load and instantiate.

**1.3** `kestrel/sim/metrics.py` — the metrics from Doc 09 §5.
✅ *Test:* metrics computed against known ground truth give expected values on a
hand-built scenario.

**1.4** `kestrel/sim/run.py` — CLI runner with `--scenario`, `--seeds`,
`--assert-metrics`, `--render`.
✅ *Done when:* `python3 -m kestrel.sim.run --scenario border_crossing` completes
and prints metrics.

> Everything after this point is developed against the simulator. Do not wait
> for hardware.

---

## Step 2 — The legal spine (3–4 days) — **build this before the radio**

Deliberately early: if collection control is retrofitted, it is never trustworthy.

**2.1** `kestrel/common/crypto.py` — Ed25519 sign/verify, sealed-box helpers,
SHA-256 (libsodium via PyNaCl).

**2.2** `kestrel/ground/signer.py` — build and two-person-sign a `WarrantToken`
(Doc 08 §2). File-based keys for now; HSM in Phase 2.
✅ *Test:* `operator_id == co_signer_id` raises.

**2.3** `kestrel/air/warrant_gate.py` — `WarrantToken` + `WarrantGate` exactly as
Doc 08 §3.
✅ *Tests (all of Doc 08 §7):* expired token, unknown class, low confidence,
outside geofence, unauthorized frequency, excluded frequency, budget exhausted,
tampered signature.

**2.4** `kestrel/ground/audit.py` — hash-chained log with `append()` and
`verify()` (Doc 08 §5). Mirror it onboard.
✅ *Test:* corrupting any line makes `verify()` fail at that sequence number.

**2.5** `kestrel/air/sealer.py` — seal-to-ground-pubkey recording (Doc 08 §4).
✅ *Tests:* sealed file cannot be read without the private key; SHA-256 matches;
**structural test asserts no unseal capability exists anywhere in `kestrel/air/`**.

**2.6** `tests/test_structural.py` — the two architectural guarantees:
no unseal in `air/`, and no write path to the vault that bypasses the gate.
✅ *Done when:* both fail loudly if someone adds a bypass.

---

## Step 3 — DSP chain (4–5 days)

**3.1** `kestrel/air/sdr_source.py` — with a `SimulatedSdrSource` implementing the
same interface, so everything downstream is testable with no hardware
(Doc 04 §2).

**3.2** `kestrel/air/detector.py` — `welch_psd`, `noise_floor`, `cfar_detect`,
`bin_to_hz` (Doc 04 §3–4).
✅ *Tests (Doc 04 §9):* injected tone found at the right frequency; pure noise
yields zero detections.

**3.3** `channelize()` and `features()` (Doc 04 §5–6).
✅ *Test:* a channelized tone lands at baseband; power normalization confirmed.

**3.4** Sweep controller with the four strategies (Doc 04 §7); compute and log
revisit time.
✅ *Test:* revisit time matches the formula.

---

## Step 4 — Classifier (5–7 days, data-dominated)

**4.1** `kestrel/sim/iq_synth.py` — IQ-level synthesis for each class using GNU
Radio or direct NumPy modulation. Start with: `fm_voice`, `dmr`, `lte`, `lora`,
`wifi`, `noise/unknown`.

**4.2** Build the IQ library: synthetic + any real recordings you can make on the
bench. Split **by recording session**, never randomly (Doc 05 §1).

**4.3** `kestrel/models/classifier_def.py` — `SignalCNN` (Doc 05 §1).

**4.4** `kestrel/train/classifier.py` — training loop with the augmentation from
Doc 05, SNR range −5…+30 dB.
✅ *Target:* > 90% accuracy above 10 dB SNR on the held-out session.

**4.5** Confidence gate: `max(softmax) < 0.85 → unknown`.
✅ *Test:* a signal class absent from training is classified `unknown` at least
80% of the time.

**4.6** Export to ONNX. TensorRT compilation waits for hardware (Doc 05 §1).
✅ *Test:* ONNX and PyTorch outputs agree within tolerance.

---

## Step 5 — Coarse geolocation (3–4 days)

Phase 1 has one aircraft and no DF array, so: RSSI-trend + motion-based bearing.

**5.1** `kestrel/air/track.py` — `EmitterTrack` EKF with `update_bearing` and
`error_ellipse` (Doc 06 §6).
✅ *Test:* feeding true bearings from several positions converges to truth and
the ellipse shrinks.

**5.2** `kestrel/ground/fusion.py` — weighted least squares over bearing
observations, robust loss (Doc 07 §5).
✅ *Test:* truth falls inside the 95% ellipse in ≥19 of 20 randomized trials.
**This test is the honesty check for the whole system.**

**5.3** Emitter association: group detections into tracks by frequency +
bandwidth + proximity.
✅ *Test:* two emitters on the same frequency 2 km apart produce two tracks, not
one.

---

## Step 6 — Autonomy agent (4–5 days)

**6.1** `kestrel/air/safety.py` — the safety layer (Doc 06 §3).
✅ *Tests:* geofence clamp, altitude clamp, battery reserve → RTL, warrant expiry
→ RTL.

**6.2** `kestrel/air/planner.py` — `fisher_information_aoa`,
`expected_info_gain`, `max_info_gain_waypoint`, plus `LawnmowerPlanner` as the
baseline to beat (Doc 06 §4).
✅ *Test:* info-gain planner reaches a target ellipse area **≥3× faster** than
lawnmower, averaged over 20 seeds.

**6.3** `kestrel/air/agent.py` — the five-state behavior tree (Doc 06 §2).
✅ *Tests:* each state transition; safety veto applied last; `RETURN` on expiry.

**6.4** `kestrel/air/mavlink_io.py` — PX4 waypoint commands + telemetry, with a
simulated backend.
✅ *Test:* agent output produces valid MAVLink `SET_POSITION_TARGET` messages.

**6.5** Full-mission simulation test:
```python
def test_full_mission_no_over_collection():
    m = run_scenario("decoy_heavy", minutes=30)
    assert m["over_collection_count"] == 0
    assert m["detection_rate"] > 0.8
```
✅ *This is the Phase 1 acceptance test.*

---

## Step 7 — Ground station (3–4 days)

**7.1** `kestrel/ground/main.py` — receives `Detection` messages over ZeroMQ,
runs fusion, appends to the audit log.

**7.2** `kestrel/ground/webmap/` — Leaflet map showing aircraft position,
detections, and **error ellipses**, with quality flags visible.
✅ *Done when:* a simulated sortie renders live in a browser.

**7.3** `kestrel/ground/export.py` — two-person evidence export with provenance
manifest (Doc 08 §6).
✅ *Test:* same requester and approver raises; integrity mismatch raises.

**7.4** Sortie report generator: every fix with ellipse, method, GDOP, quality,
model versions, audit verification result.

---

## Step 8 — Hardware bring-up (1 week, gated on delivery)

**8.1** Assemble per Doc 02 §6. Run the §8 acceptance checks.
**8.2** Swap `SimulatedSdrSource` → real `SdrSource`. Verify zero overflows.
**8.3** Compile the classifier with TensorRT on the Jetson; verify < 2 ms.
**8.4** Bench test — **all 11 tests in Doc 10 §2**, compliance sign-off on 5–11.
**8.5** Validate the simulator against measured reality (Doc 09 §7); tune the
propagation model.
**8.6** Tethered flight — Doc 10 §4.
**8.7** Single-aircraft free flight — Doc 10 §5.

---

## Phase 1 Definition of Done

- [ ] Aircraft flies a full autonomous mission with **no operator input**
- [ ] Detects and classifies ≥4 signal classes at >90% accuracy above 10 dB SNR
- [ ] Produces geolocation fixes with **honest** error ellipses (19/20 containment)
- [ ] Records **only** warrant-authorized content — proven by bench tests and a
      real sortie's audit review
- [ ] Sealed recordings unreadable on the aircraft; exportable only with two people
- [ ] Audit chain verifies; every recording maps to an authorized detection
- [ ] Live map shows detections and fixes in real time
- [ ] Simulator validated against measured reality within agreed tolerance
- [ ] Full test suite green in CI, including all structural guarantees

**Estimated effort:** ~5–6 weeks of engineering plus hardware lead time. Steps
0–7 need no hardware at all — start immediately.

---

## Phase 2 preview (do not start until Phase 1 is done)

GPSDO + swarm time sync · 4-element DF array + MUSIC + calibration · SADF ·
3 aircraft + spectral choreography + recruitment · TDOA fusion · HSM key custody ·
information-gain flight validated in the air.

## Phase 3 preview

RF Weather Map (unsupervised baselines) · emitter fingerprinting ·
pattern-of-life database · lawful-intercept demodulation of authorized traffic ·
RL planner trained in KestrelSim · tethered persistent-overwatch variant.

---

## Ordering rationale (why this sequence)

1. **Simulator first** — everything else is developed and regression-tested
   against it; without it you're debugging RF, software, and flight at once.
2. **Legal spine second** — collection control retrofitted is collection control
   you cannot trust, and it constrains every interface downstream.
3. **DSP, then AI** — the classifier needs the channelizer's output shape fixed
   before training is meaningful.
4. **Autonomy before hardware** — the agent is pure logic; hardware only adds
   noise to its development.
5. **Hardware last** — by the time the radio arrives, everything downstream of it
   is already proven, so bring-up debugs one thing at a time.

Start at Step 0.1.
