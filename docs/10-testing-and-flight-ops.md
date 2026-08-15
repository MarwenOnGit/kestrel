# 10 — Testing and Flight Operations

Nothing flies until it has passed the stage below it. Checklists are meant to be
printed and signed.

---

## 1. The test ladder

```
  6. OPERATIONAL SORTIE       ← real mission, real warrant
  5. FREE FLIGHT (3 aircraft) ← swarm, TDOA, autonomy
  4. FREE FLIGHT (1 aircraft) ← autonomy over a known emitter
  3. TETHERED FLIGHT          ← payload under real vibration/EMI
  2. GROUND CALIBRATION       ← DF phase offsets, surveyed emitter
  1. BENCH TEST               ← shielded, known signal, no radiation
  0. SIMULATION               ← every commit, no hardware
```

**Never skip a rung.** Each one catches a class of failure the next cannot.

---

## 2. Rung 1 — Bench test (shielded)

**Setup:** signal generator → attenuators → shielded enclosure or direct
coaxial connection to the SDR. **No radiating antenna** — you must not transmit
into the air on frequencies you are not licensed for. Use a dummy load and
conducted injection.

| # | Test | Pass criterion |
|---|---|---|
| 1 | Stream 200 MHz for 10 min | Zero overflows |
| 2 | Inject tone at known freq | Detected within ±2 kHz |
| 3 | Sweep injected power down | Detection threshold matches predicted sensitivity ±3 dB |
| 4 | Inject each signal class | Classifier accuracy > 90% at SNR > 10 dB |
| 5 | Inject **authorized** class/freq | Recording created, sealed, audit entry present |
| 6 | Inject **unauthorized** freq | **No recording.** Metadata only. Reason logged |
| 7 | Inject with **expired** token | **No recording.** `token_expired` logged |
| 8 | Tamper with token, reload | Rejected, `TOKEN_REJECTED` logged, aircraft refuses to arm |
| 9 | Fill the vault | Recording stops, gap logged, ground alerted |
| 10 | Verify audit chain | `verify()` returns OK |
| 11 | Attempt to read a sealed file on the aircraft | **Impossible** — no unseal capability present |

Tests 5–11 are the legal spine. **Get them signed off by whoever owns
compliance**, not just by engineering.

---

## 3. Rung 2 — Ground calibration

### (a) DF array phase calibration
1. Place a test emitter at a **surveyed** position, known bearing, ≥10λ away.
2. Rotate the array (or drive it) through 360° in 10° steps.
3. At each step record measured vs. true bearing.
4. Fit and store per-channel phase offsets:
```bash
python3 tools/calibrate_array.py --scan scan.json --out config/array_cal_K1.json
```
5. Re-run; residual bearing error should be **< 3° RMS**.

**Recalibrate after any cable change, any antenna remount, and every 20 flight
hours.** Store the calibration date — it goes in the evidence manifest.

### (b) Attitude correction check
Point the aircraft north, then east, with the emitter fixed. Reported *true*
bearing must stay constant. If it rotates with the airframe, your yaw correction
is wrong (Doc 07 §2) — this is a common and dangerous bug because the fixes look
plausible.

### (c) Swarm time sync
```bash
python3 tools/check_swarm_time.py --nodes K1,K2,K3
```
Pairwise offset **< 50 ns**, stable for 10 minutes. Below this, TDOA is invalid.

### (d) Surveyed-emitter geolocation
Place the emitter at a known point; run the full chain on the ground with three
receivers spread out. **Truth must fall inside the reported 95% ellipse** in at
least 19 of 20 trials. If it doesn't, your uncertainty estimate is dishonest —
fix that before flying, because everything downstream trusts the ellipse.

---

## 4. Rung 3 — Tethered flight

Purpose: expose the payload to real vibration, EMI, and thermal conditions.

- [ ] Noise floor with motors running vs. stopped: **< 3 dB** difference
- [ ] Zero SDR overflows during full-throttle hover
- [ ] GPSDO stays locked throughout
- [ ] DF bearings on a ground test emitter remain within 5° of the ground-
      calibrated value
- [ ] Payload temperature < 80 °C after 20 min
- [ ] Battery draw matches the budget (Doc 02 §7)

Motor EMI raising the noise floor is the most common failure here. If you see it,
the fix is shielding, filtering, and physical separation — **not** turning up the
gain.

---

## 5. Rung 4 — Single-aircraft free flight

Fly a **known test emitter** at a surveyed location, in approved airspace, under
a test authorization.

- [ ] Autonomous takeoff → survey → contact → refine → report → RTL, **no
      operator input**
- [ ] Safety layer respected: geofence never breached, altitude bounds held
- [ ] Emitter detected and classified correctly
- [ ] SADF bearing error < 5° (Doc 07 §4)
- [ ] Reported ellipse contains the true position
- [ ] Warrant gate behaved: only authorized content recorded
- [ ] Audit chain verifies after landing
- [ ] Simulated link loss → mission continues, reports queue and flush

**Abort criteria (any one → land immediately):** loss of GPS lock, payload over
temperature, unexpected agent state, geofence warning, battery below reserve.

---

## 6. Rung 5 — Three-aircraft swarm

- [ ] Spectral choreography: sectors divided, union covers all segments
- [ ] Kill one aircraft's link mid-flight → survivors re-divide within 10 s
- [ ] TDOA fix on surveyed emitter within **50 m**, truth inside the ellipse
- [ ] Information-gain flight visibly improves geometry (log GDOP over time —
      it should fall)
- [ ] Recruitment works: one aircraft announces a contact, exactly one diverts
- [ ] Minimum separation (30 m) never violated
- [ ] Compare against a lawnmower baseline: **time-to-fix at least 3× better**

---

## 7. Rung 6 — Operational sortie checklist

### Pre-flight (T−60 min)
- [ ] Warrant token issued, **co-signed**, loaded, signature verified on each
      aircraft
- [ ] Token validity window covers the planned sortie **with margin**
- [ ] Geofence matches the warrant polygon (verify visually on the map)
- [ ] Aviation authorization current; airspace checked; NOTAM if required
- [ ] Array calibration current (< 20 flight hours)
- [ ] GPSDO locked on all aircraft, swarm time verified
- [ ] Vault empty and sealed; seal public key matches the ground HSM
- [ ] Model versions recorded in the audit log
- [ ] Mesh encryption keyed; own datalink band notched from the survey plan
- [ ] Battery health; payload mass within limits
- [ ] Abort procedure briefed; revocation message tested

### In-flight
- [ ] Monitor: link status, battery, GPSDO lock, gate decisions, vault usage
- [ ] Watch for `quality: degraded` fixes — do not act on them
- [ ] Abort on: warrant expiry approaching, GPS loss, thermal, airspace conflict

### Post-flight
- [ ] Aircraft disarmed; vault closed
- [ ] Audit chain verified on each aircraft (`audit.verify()`)
- [ ] Chain head hashes recorded and anchored
- [ ] Recordings transferred by cable, hashes re-verified
- [ ] Sortie report generated with all error ellipses and quality flags
- [ ] Low-confidence / high-novelty detections queued for analyst labeling
- [ ] Anything outside expectations written up

---

## 8. Continuous automated testing

Runs on every commit, no hardware:

```bash
pytest tests/ -v --cov=kestrel                       # unit
python3 -m kestrel.sim.run --all-scenarios --seeds 20 --assert-metrics
pytest tests/test_warrant_gate.py -v                 # legal spine
pytest tests/test_structural.py -v                   # no unseal in air/, no gate bypass
```

**Build-breaking assertions:**
- `over_collection_count == 0` in every scenario
- No unseal capability in `kestrel/air/`
- No write path to the vault that bypasses the warrant gate
- Audit chain verifies after every simulated sortie

---

## 9. Incident response

| Incident | Immediate action | Follow-up |
|---|---|---|
| Suspected over-collection | Ground the fleet; freeze vaults | Audit review; report to authorizing body; **destroy** unauthorized material per procedure |
| Aircraft lost / not recovered | Report location; assume hardware compromised | Rotate mesh keys; sealed data remains protected; document |
| Audit chain fails verification | Treat all sortie data as inadmissible | Forensic review before any further flights |
| Geofence breach | Land immediately | Review safety layer + PX4 params; do not fly until root-caused |
| GPSDO unlock mid-sortie | TDOA fixes from that period are **void** | Re-flag affected fixes in the database and in any report already issued |

**Reporting discipline:** if a fix was later invalidated, anyone who received it
must be told. Silent correction of distributed intelligence is not acceptable.

---

Next: [`11-phase1-implementation-plan.md`](11-phase1-implementation-plan.md).
