# 01 — Architecture

The whole system, subsystem by subsystem, with the interface contracts that let
each be built and tested independently.

---

## 1. Design principles

1. **Small specialists, not one big brain.** No monolithic model flies the
   aircraft. A tiny orchestrator calls proven components. This is what makes
   full autonomy fit on limited hardware.
2. **Scope enforced by mathematics, not policy.** The warrant is a signed
   capability token; content is sealed to a key the drone does not hold.
3. **Fail closed.** Any doubt — invalid token, expired warrant, lost GPS,
   uncertain classification — results in *less* collection, never more.
4. **The link is a luxury.** Every aircraft completes its mission with the
   datalink dead. Reports queue; scope enforcement is local.
5. **Honest uncertainty.** Every geolocation output carries an error region.
   The system never reports a point it cannot defend.
6. **Metadata is cheap, content is precious.** Survey the whole band as
   metadata; capture content only inside the warrant.

---

## 2. System diagram

```
   AIRCRAFT 1                AIRCRAFT 2                AIRCRAFT 3 … N
 ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
 │ A  RF FRONT END    │   │      (identical)   │   │      (identical)   │
 │   antennas → filter│   │                    │   │                    │
 │   → LNA → SDR      │   │                    │   │                    │
 │        ↓ IQ        │   │                    │   │                    │
 ├────────────────────┤   │                    │   │                    │
 │ B  EDGE PROCESSING │   │                    │   │                    │
 │   FFT → detect     │   │                    │   │                    │
 │   → AI classify    │   │                    │   │                    │
 │   → WARRANT GATE   │   │                    │   │                    │
 │     ├ match → seal │   │                    │   │                    │
 │     │        + rec │   │                    │   │                    │
 │     └ else → count │   │                    │   │                    │
 ├────────────────────┤   │                    │   │                    │
 │ C  GEO SENSING     │   │                    │   │                    │
 │   AOA array, SADF  │   │                    │   │                    │
 │   GPSDO timestamps ├───┼── shared time ─────┼───┤ → TDOA baselines   │
 ├────────────────────┤   │                    │   │                    │
 │ F  AUTONOMY AGENT  │   │                    │   │                    │
 │   orchestrator     │◄──┼── gossip: tasking, ┼──►│  recruitment        │
 │   info-gain planner│   │   contacts, health │   │                    │
 │   safety veto      │   │                    │   │                    │
 │        ↓ waypoints │   │                    │   │                    │
 │   PX4 flight ctrl  │   │                    │   │                    │
 ├────────────────────┤   │                    │   │                    │
 │ D  DATALINK (mesh, encrypted) ──────────────────────────────┐        │
 └────────────────────┘   └────────────────────┘   └───────────┼────────┘
                                                               │
                                       ┌───────────────────────▼──────────┐
                                       │ E  GROUND STATION                │
                                       │   warrant token signer (HSM)     │
                                       │   live map + error ellipses      │
                                       │   geolocation fusion             │
                                       │   hash-chained audit log         │
                                       │   pattern-of-life database       │
                                       │   evidence export (2-person)     │
                                       └──────────────────────────────────┘
```

Subsystems **A–E** are the sensing/legal spine. **F** is the autonomy layer.
**G** (KestrelSim, Doc 09) is ground-only: the digital twin used to develop and
train everything before it flies.

---

## 3. Subsystem A — RF Front End ("the ears")

**Purpose:** capture radio energy across 30 MHz – 8 GHz cleanly.

**Chain:** `Antenna → band filter → LNA → SDR → IQ`

**Two antenna groups per aircraft** (no single antenna spans this range):

| Group | Range | Type | Serves |
|---|---|---|---|
| Low | 30–520 MHz | Wideband discone or blade | Handheld VHF/UHF voice, marine VHF, DMR/TETRA |
| High | 0.7–8 GHz | Log-periodic / Vivaldi / spiral | Cellular, ISM, LoRa, WiFi/BT, drone links, L/S/Ku satcom |

Plus, for direction finding, a **4-element circular array** in the UHF band
(Doc 02 §4). Bearing comes from comparing elements, so geometry must be known
to millimetres and stay rigid in flight.

**SDR:** Ettus **USRP X410** — 4 coherent channels, up to 400 MHz instantaneous
bandwidth, 1 MHz–8 GHz. (**USRP B210** — 2 channels, 56 MHz — is the lighter
Phase-1 option.) Coherence is mandatory: see Doc 00 §6.

**Contract A→B:** IQ stream over UHD/10 GbE. Parameters: `center_freq_hz`,
`sample_rate_sps`, `n_channels`, `gain_db`, plus a GPSDO-derived timestamp on
every buffer.

---

## 4. Subsystem B — Edge Processing ("the brain")

**Purpose:** find every signal, identify it, decide if the warrant permits
capture, act — in real time, onboard.

**Hardware:** NVIDIA **Jetson AGX Orin** (GPU for AI) + the SDR's FPGA (for the
cheap always-on work) + encrypted **NVMe SSD**.

**Pipeline, per aircraft, continuously:**

1. **Tune/sweep** the assigned band segment (assigned by Spectral Choreography,
   Doc 06 §5).
2. **FFT → spectrum.** On the FPGA where possible — it's the highest-rate,
   dumbest work.
3. **Energy tripwire.** Cheap CFAR-style threshold against a rolling noise-floor
   estimate. Produces candidate detections. Runs always; costs almost nothing.
   *(This two-tier design is what keeps the power budget survivable.)*
4. **Classify** each candidate — GPU neural net (Doc 05): FM voice, DMR, TETRA,
   LTE/5G, LoRa, WiFi/BT, DJI/OcuSync, satcom terminal, or `unknown`.
5. **Novelty check** against the RF Weather Map (Doc 05 §4): is this normal for
   this tile at this hour?
6. **WARRANT GATE** (Doc 08 §3). Compare against the signed token.
   - **Match** → record IQ for *that channel only*, **sealed** to the ground
     public key, tagged with warrant ID, time, position, bearing.
   - **No match** → **metadata-only** record. No content. Ever.
7. **Emit** a `Detection` to the datalink queue and to the local autonomy agent.

**Why the gate sits after the AI:** the AI may be uncertain; `unknown` is not
`authorized`. Uncertainty must never widen collection.

---

## 5. Subsystem C — Geolocation Sensing ("the where")

Onboard: measure. On the ground: fuse. Three measurement techniques (Doc 07):

- **AOA** — bearing from the physical 4-element array.
- **SADF (Synthetic Aperture DF)** — use the aircraft's own motion plus GPSDO
  phase coherence to synthesize an aperture tens of metres long, far exceeding
  the airframe. This is how a 1.2 m drone gets useful VHF bearings.
- **TDOA** — nanosecond-accurate arrival timestamps across ≥3 aircraft.

**Contract:** every detection carries `t_gpsdo_ns`, `aircraft_pos_lla`,
`aircraft_attitude`, and optional `bearing_deg ± sigma`.

---

## 6. Subsystem D — Datalink ("the voice")

- **MANET mesh radio** (Doodle Labs / Silvus class) — self-healing between
  aircraft and to ground.
- **Encrypted** (AES-256) and **authenticated** — nobody can inject false
  detections into an evidentiary system.
- **Downlinked live:** `Detection` JSON + small spectrum thumbnails. Kilobytes.
- **Stays onboard:** sealed IQ recordings (gigabytes), retrieved by cable after
  landing to preserve chain of custody.
- **Aircraft-to-aircraft:** gossip messages for tasking, contact recruitment, and
  health (Doc 06 §5).

---

## 7. Subsystem E — Ground Station ("command")

- **Warrant token signer** — builds the authorized target list from the legal
  authorization; two operators co-sign via HSM; token is signed, time-boxed and
  geofenced (Doc 08 §2).
- **Live map** — detections and fixes with error ellipses, in real time.
- **Geolocation fusion** — combines AOA + SADF + TDOA into maximum-likelihood
  fixes (Doc 07 §5).
- **Hash-chained audit log** — append-only; every detection and every recording
  bound to a warrant ID.
- **Pattern-of-life database** — emitters correlated across sorties and days.
- **Evidence export** — two-person, HSM-gated decryption of sealed recordings;
  itself logged.

---

## 8. Subsystem F — Autonomous Mission Agent

The layer that removes the operator. Full detail in Doc 06. In brief:

```
   ┌─────────────────────────────────────────────┐
   │  SAFETY LAYER  (hard-coded, can veto all)   │  geofence, altitude,
   └───────────────▲─────────────────────────────┘  battery, separation,
                   │ vetoes                          lost-link RTL
   ┌───────────────┴─────────────────────────────┐
   │  MISSION AGENT (orchestrator, ~50 KB)       │  behavior tree:
   │   SURVEY → CONTACT → REFINE → REPORT        │  sense-decide-act loop
   └───────────────┬─────────────────────────────┘
                   │ waypoints only
   ┌───────────────▼─────────────────────────────┐
   │  PX4 FLIGHT CONTROLLER (proven, classical)  │  actually flies
   └─────────────────────────────────────────────┘
```

The agent **never** commands motors. It emits waypoints; PX4 flies them; the
safety layer can override both. This separation is what makes autonomy
certifiable.

---

## 9. Interface contracts

Stable contracts = independently buildable subsystems. Full JSON schemas in
[`schemas/`](../schemas/).

**`Detection`** (B → D → E, and B → F):
```json
{
  "detection_id": "uuid",
  "aircraft_id": "K1",
  "t_gpsdo_ns": 1755180000123456789,
  "freq_hz": 462562500,
  "bandwidth_hz": 12500,
  "snr_db": 18.4,
  "signal_class": "dmr",
  "class_confidence": 0.93,
  "novelty_score": 0.71,
  "bearing_deg": 47.2,
  "bearing_sigma_deg": 3.1,
  "aircraft_pos_lla": [31.2, 34.8, 120.0],
  "warrant_match": true,
  "warrant_id": "W-2026-0142",
  "recording_ref": "sealed/K1/1755180000.iqz or null"
}
```

**`WarrantToken`** (E → A, pre-flight, signed — Doc 08 §2):
```json
{
  "warrant_id": "W-2026-0142",
  "issued_by": "…", "co_signer": "…",
  "freq_ranges_hz": [[462000000, 468000000]],
  "signal_classes": ["dmr", "fm_voice"],
  "area_polygon": [[31.1,34.7],[31.3,34.7],[31.3,34.9],[31.1,34.9]],
  "valid_from": "2026-08-14T04:00:00Z",
  "valid_until": "2026-08-14T16:00:00Z",
  "max_recording_seconds": 3600,
  "seal_pubkey": "base64…",
  "signature": "base64…"
}
```

**`GeoFix`** (E internal → map/database):
```json
{
  "emitter_id": "E-0093",
  "lat": 31.2043, "lon": 34.8117,
  "error_ellipse": {"semi_major_m": 42.0, "semi_minor_m": 18.5, "orientation_deg": 73.0, "confidence": 0.95},
  "method": "tdoa+aoa", "n_aircraft": 3,
  "contributing_detections": ["uuid", "…"]
}
```

**`AudioSealedRecording`** sidecar (onboard):
```json
{
  "recording_ref": "sealed/K1/1755180000.iqz",
  "warrant_id": "W-2026-0142",
  "detection_ref": "uuid",
  "start_ns": 0, "stop_ns": 0,
  "gps_track": [[31.2,34.8,120.0,1755180000]],
  "sha256": "…",
  "sealed_to_pubkey": "base64…"
}
```

---

## 10. Failure modes and responses

| Failure | Response |
|---|---|
| Datalink lost | Continue mission; queue reports; flush on reconnect. Gate is local, so scope holds. |
| GPS / GPSDO lost | Disable TDOA (timing invalid); AOA + survey continue; fixes flagged `degraded`; PX4 falls back to its own failsafe. |
| NVMe full or failing | Stop recording, keep metadata, alert ground, **log the gap** — never a silent loss. |
| Warrant token invalid / expired / outside geofence | Record **nothing**; metadata only; alert. Fail closed. |
| Classifier uncertain | `signal_class: unknown` → counted, never recorded. |
| Aircraft lost or captured | Sealed recordings are unreadable without the ground HSM key. |
| One aircraft drops from swarm | Others re-divide spectrum and sky via gossip; TDOA degrades to AOA-only if <3 remain. |
| Autonomy agent misbehaves | Safety layer vetoes; PX4 failsafe; operator abort always available on the mesh. |

---

## 11. Compute and power budget (per aircraft, indicative)

| Consumer | Power | Notes |
|---|---|---|
| SDR (X410) | ~25 W | Dominated by wideband ADC |
| Jetson AGX Orin | 15–40 W | Configurable; run at 30 W cap |
| Mesh radio | ~10 W | Transmit duty dependent |
| GPSDO | ~2 W | |
| NVMe | ~5 W peak | Only while recording |
| **Payload total** | **~60–80 W** | Sized against airframe endurance in Doc 02 §7 |

The two-tier triage (cheap tripwire always on, GPU classifier only on candidate)
is what keeps the Jetson near its floor instead of its ceiling.

---

Next: [`02-hardware-build.md`](02-hardware-build.md).
