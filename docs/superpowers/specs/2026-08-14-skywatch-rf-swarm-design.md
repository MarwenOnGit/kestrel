# KESTREL — Design Specification

**v2.0 — 2026-08-14** (supersedes v1.0 "SkyWatch"; renamed and extended with the
autonomy layer, AI subsystem, and the five signature capabilities)

Autonomous airborne RF collection swarm for authorized spectrum monitoring,
emitter geolocation, and lawful signal collection.

> This is the formal specification. The **executable documentation** — build
> steps, code, and the implementation plan — lives in [`docs/`](../../).
> Start with [`docs/00-primer-rf-from-zero.md`](../../00-primer-rf-from-zero.md)
> if you are new to RF, and
> [`docs/11-phase1-implementation-plan.md`](../../11-phase1-implementation-plan.md)
> to begin building.

---

## 1. Purpose and scope

Three or more UAVs, operating with no human in the loop, survey a designated
area across 30 MHz – 8 GHz; detect, classify, and geolocate transmitters;
capture content only within a cryptographically signed warrant; and report to a
ground station in real time.

**In scope:** passive reception, detection, classification, direction finding,
geolocation, warrant-bounded recording, autonomous flight and swarm
coordination, chain-of-custody evidence handling.

**Explicitly out of scope, by design:** any transmit capability (no jamming, no
spoofing, no injection); decryption of protected communications; collection
outside a signed authorization; identification of persons (the system locates
*devices*).

---

## 2. Requirements

### Functional
- **F1** Detect signals across 30 MHz – 8 GHz with a swarm revisit time < 5 s.
- **F2** Classify ≥10 signal classes at >90% accuracy above 10 dB SNR, and emit
  `unknown` when uncertain.
- **F3** Geolocate emitters with a 95% error ellipse ≤ 50 m under good geometry.
- **F4** Record content **only** when a signed warrant token authorizes it.
- **F5** Operate a full sortie autonomously: takeoff → survey → contact →
  refine → report → return.
- **F6** Report detections and fixes live to a ground map, with uncertainty.
- **F7** Correlate emitters across sorties (pattern-of-life). *(Phase 3)*
- **F8** Produce a tamper-evident audit trail binding every recording to a
  warrant.

### Non-functional
- **N1** Payload ≤ 6 kg, ≤ 80 W (Doc 02 §7).
- **N2** Complete the mission with the datalink dead.
- **N3** Fail closed: any uncertainty reduces collection.
- **N4** Every geolocation output carries an honest error region.
- **N5** The warrant gate is deterministic and auditable line by line — no AI.
- **N6** Airborne code has no capability to decrypt its own recordings.

---

## 3. Architecture

Seven subsystems. Full detail in
[`docs/01-architecture.md`](../../01-architecture.md).

| | Subsystem | Purpose | Doc |
|---|---|---|---|
| **A** | RF Front End | Antennas, filters, LNAs, coherent SDR | [02](../../02-hardware-build.md) |
| **B** | Edge Processing | FFT → CFAR → classify → warrant gate → seal | [04](../../04-signal-processing.md), [05](../../05-ai-models.md) |
| **C** | Geolocation Sensing | AOA, SADF, TDOA measurement | [07](../../07-geolocation.md) |
| **D** | Datalink | Encrypted mesh; reports down, IQ stays aboard | [01 §6](../../01-architecture.md) |
| **E** | Ground Station | Signing, fusion, map, audit, export | [08](../../08-legal-and-audit.md) |
| **F** | Autonomy Agent | Behavior tree, info-gain planner, swarm gossip | [06](../../06-autonomy-agent.md) |
| **G** | KestrelSim | Digital twin: development, training, regression | [09](../../09-simulator.md) |

**Layering principle (F):** a hard-coded safety layer can veto a small
orchestrator, which emits *waypoints only* to a proven PX4 flight controller. No
learned model ever commands motors, and no learned model ever authorizes
collection.

**Processing model:** hybrid edge triage with triggered recording. Continuous
cheap detection; expensive AI only on candidates; content capture only on
warrant match. The funnel (800 MB/s → ~5 KB/s reported) is documented in
[Doc 04 §1](../../04-signal-processing.md).

---

## 4. The five signature capabilities

What distinguishes this design from a conventional SIGINT drone:

1. **Information-Gain Flight** — waypoints chosen to maximize the predicted
   reduction in geolocation uncertainty (Fisher information / D-optimality)
   rather than following a search grid. 3–5× faster convergence, costs
   microseconds of CPU, and is fully explainable. → [Doc 06 §4](../../06-autonomy-agent.md)
2. **Synthetic Aperture DF** — the aircraft's own motion synthesizes an aperture
   of tens of metres, giving VHF bearing accuracy a drone-sized array cannot
   reach. The aperture is free; you were flying anyway. → [Doc 07 §4](../../07-geolocation.md)
3. **RF Weather Map** — an unsupervised per-tile, per-hour model of "normal"
   spectrum. Novelty needs no labels, so previously unseen emitters are caught
   rather than misclassified. → [Doc 05 §2](../../05-ai-models.md)
4. **Sealed Collection** — recordings are encrypted to a ground-held key; the
   aircraft cannot read its own take. The warrant is a signed, time-boxed,
   geofenced capability token. Legal scope becomes a property of the
   mathematics. → [Doc 08 §2–4](../../08-legal-and-audit.md)
5. **Spectral Choreography** — decentralized gossip divides spectrum and sky
   across the swarm and recruits aircraft onto contacts, covering 8 GHz with
   radios that see 400 MHz each. Survives any aircraft dropping out. → [Doc 06 §5](../../06-autonomy-agent.md)

---

## 5. Interface contracts

`Detection`, `WarrantToken`, `GeoFix`, and the recording sidecar are specified in
[`docs/01-architecture.md` §9](../../01-architecture.md) and enforced by JSON
schemas in [`schemas/`](../../../schemas/).

---

## 6. Error handling

Full table in [`docs/01-architecture.md` §10](../../01-architecture.md).
Governing rule: **every failure path reduces collection and is logged.** Lost
link → keep working, queue reports. Lost GPS → TDOA void, not merely degraded.
Invalid token → record nothing. Uncertain classification → `unknown`, never
recorded.

---

## 7. Testing

Six-rung ladder, no rung skipped:
simulation → shielded bench → ground calibration → tethered → single free
flight → swarm. Detail and checklists in
[`docs/10-testing-and-flight-ops.md`](../../10-testing-and-flight-ops.md).

**Build-breaking assertions** (CI, every commit):
`over_collection_count == 0` in every scenario · no unseal capability in
`kestrel/air/` · no vault write path bypassing the warrant gate · audit chain
verifies after every simulated sortie.

---

## 8. Build phases

- **Phase 1** — one aircraft: survey, classify, coarse geolocation,
  warrant-gated recording, live map, audit. *First flyable deliverable.*
- **Phase 2** — the swarm: GPSDO time sync, DF array, SADF, TDOA fusion,
  spectral choreography, information-gain flight in the air.
- **Phase 3** — full intelligence: RF Weather Map, fingerprinting,
  pattern-of-life, authorized demodulation, RL planner.

Step-by-step plan for Phase 1:
[`docs/11-phase1-implementation-plan.md`](../../11-phase1-implementation-plan.md).

---

## 9. Open questions (blocking, non-technical)

Answers change data schemas, not just paperwork. Detail in
[`docs/08-legal-and-audit.md` §8](../../08-legal-and-audit.md):

1. Governing statutory authority → `WarrantToken` fields and retention rules.
2. Minimization requirements → real-time onboard stop rules.
3. Retention and certified destruction procedure.
4. Key custody — must be a different role from flight operations.
5. Aviation authorization for autonomous BVLOS operation.
6. Post-hoc notification/disclosure obligations.

Build against the provisional schema in Doc 08 §2 until 1–4 are answered.

---

*End of specification v2.0.*
