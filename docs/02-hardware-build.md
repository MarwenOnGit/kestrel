# 02 — Hardware Build

Bill of materials, antenna construction, wiring, assembly, and the checks that
tell you it worked. Assumes hardware budget is not a constraint, but flags where
weight and power genuinely are.

---

## 1. Bill of materials — per aircraft

### Sensing
| Item | Recommended | Why this one |
|---|---|---|
| SDR | **Ettus USRP X410** | 4 coherent RX channels, 1 MHz–8 GHz, up to 400 MHz BW. Coherence enables DF/TDOA. |
| SDR (light alt.) | Ettus USRP B210 | 2 channels, 56 MHz BW. Phase-1 only; cheaper/lighter. |
| Clock | **GPSDO** (Ettus GPSDO for X410, or Jackson Labs LN) | Nanosecond swarm time → TDOA accuracy. |
| Low-band antenna | Wideband **discone** 30–520 MHz, or blade | The VHF/UHF voice mission. Physically large. |
| High-band antenna | **Log-periodic** or **Vivaldi**, 0.7–8 GHz | Cellular / ISM / satcom / drone links. |
| DF array | 4× matched UHF elements, circular, phase-matched cables | Bearing measurement. Geometry must be rigid. |
| LNAs | Mini-Circuits ZX60-series or similar, NF < 1.5 dB | Sensitivity floor. One per chain, at the antenna. |
| Filters | Band-pass per group + notch for local broadcast | Prevents desensitization by strong local TX. |
| Cables | Phase-matched, low-loss (LMR-100/RG-405), equal lengths | **Unequal cable lengths destroy DF accuracy.** |

### Compute and storage
| Item | Recommended | Why |
|---|---|---|
| Edge computer | **NVIDIA Jetson AGX Orin 64 GB** | GPU for classifier; 10 GbE for USRP; 15–60 W configurable. |
| Storage | 2 TB **NVMe SSD**, hardware-encrypted | Sealed IQ recordings. |
| Carrier board | Vendor carrier with 10 GbE + M.2 | Must have 10 GbE to feed X410. |

### Flight and link
| Item | Recommended | Why |
|---|---|---|
| Airframe | Heavy-lift multirotor, **5–15 kg payload**, e.g. Freefly Alta X class | Must carry the VHF antenna + full receiver. |
| Flight controller | **Pixhawk 6X running PX4** | Proven, autonomous waypoint flight, robust failsafes. |
| Companion link | Jetson ↔ Pixhawk over serial, **MAVLink** | Agent sends waypoints, reads telemetry. |
| Mesh radio | **Doodle Labs Helix / Silvus StreamCaster** | Encrypted self-healing MANET, aircraft-to-aircraft. |
| Battery | Sized for ~80 W payload + lift, LiPo/Li-ion | See §7. |
| Vibration isolation | Damped payload plate | Vibration corrupts phase → corrupts DF. |

### Ground station
| Item | Recommended |
|---|---|
| Workstation | Linux laptop/rack, RTX GPU (training + fusion) |
| Ground mesh radio | Matching Doodle/Silvus node + directional antenna |
| **HSM** | YubiHSM 2 or Nitrokey HSM — holds warrant signing key **and** seal private key |
| Map | QGIS, or the bundled Leaflet web map |
| Test emitter | Signal generator + attenuators + **shielded dummy load** |

---

## 2. Critical assembly rules (violate these and nothing works)

1. **Equal, phase-matched cables on all DF channels.** A 1 cm length mismatch at
   450 MHz is a ~5° bearing error. Cut them together, measure them together.
2. **LNA at the antenna, not at the radio.** Amplify before cable loss.
3. **Rigid array geometry.** If elements flex in flight, bearings wander. Mount
   on a stiff plate, not on booms that vibrate.
4. **Isolate the payload from motor noise.** Drone ESCs are RF-noisy. Shield the
   payload bay, ferrite the power leads, and keep the SDR away from ESCs.
5. **Separate power rails.** Never run the SDR off the same unfiltered rail as
   motors. Use a dedicated regulated supply with filtering.
6. **Ground the shield once.** Ground loops inject hum.
7. **Never transmit.** This is a receive-only payload. No TX antenna port should
   be connected to an amplifier at all.

---

## 3. RF chain wiring

```
  LOW BAND (30–520 MHz)
    Discone ──► BPF 30–520 ──► LNA (NF<1.5dB) ──► [X410 RX0]

  HIGH BAND (0.7–8 GHz)
    Log-periodic ──► BPF 0.7–8G ──► LNA ──► [X410 RX1]

  DF ARRAY (UHF, 4 elements — Phase 2)
    E1 ─ BPF ─ LNA ─ phase-matched cable ─► [X410 RX0]
    E2 ─ BPF ─ LNA ─ phase-matched cable ─► [X410 RX1]
    E3 ─ BPF ─ LNA ─ phase-matched cable ─► [X410 RX2]
    E4 ─ BPF ─ LNA ─ phase-matched cable ─► [X410 RX3]
         (all cables cut to identical length)

  CLOCK
    GPS antenna ──► GPSDO ──► 10 MHz REF + PPS ──► X410 REF IN / PPS IN

  DATA
    X410 ──10 GbE──► Jetson AGX Orin ──M.2──► encrypted NVMe
                          │
                          ├─ USB/serial ─► Pixhawk 6X (MAVLink)
                          └─ Ethernet ───► Mesh radio
```

**Note:** in Phase 2 the DF array consumes all four channels, so low/high band
survey and DF are time-multiplexed (the agent switches modes). If you want both
simultaneously, add a second X410 — the design supports it.

---

## 4. Building the DF array (Phase 2)

A **4-element uniform circular array (UCA)** for UHF ~450 MHz.

1. **Element spacing.** Radius r ≈ 0.25–0.4 λ. At 450 MHz, λ = c/f = 300/450 =
   0.667 m, so r ≈ 0.17–0.27 m. Choose **r = 0.20 m** → array diameter 0.40 m.
   *(λ in metres = 300 / frequency in MHz. Memorize this.)*
2. **Elements.** Four identical quarter-wave monopoles (λ/4 = 16.7 cm at
   450 MHz) on a ground plane, or four small matched dipoles.
3. **Geometry.** Mount at 0°, 90°, 180°, 270° on a rigid carbon plate. Measure
   positions to ±2 mm and **record them** — the DF math needs exact geometry.
4. **Cables.** Cut four identical lengths. Verify with a VNA that electrical
   length matches within a few degrees of phase.
5. **Calibrate.** See Doc 10 §3 — you must measure and store per-channel phase
   offsets; they are never zero in practice.

> **Why the array is small and the aperture isn't.** A 0.4 m array at VHF gives
> poor bearing accuracy on its own. **Synthetic Aperture DF** (Doc 07 §4) uses
> the aircraft's flight path to synthesize an aperture of tens of metres,
> recovering the accuracy a physically huge array would give. The physical array
> handles fast bursts; SADF handles anything lasting long enough to fly across.

---

## 5. Antenna placement on the airframe

- **Low-band discone**: below the airframe, on a mast, as far from the frame as
  practical. The airframe itself distorts patterns.
- **High-band log-periodic**: downward-looking, gimbal-mounted if you want steer-
  able gain (optional; adds mass but greatly improves satcom pickup).
- **DF array**: horizontal plate underneath, level, centered — off-center arrays
  need extra calibration and pick up more airframe reflection.
- **GPS antenna**: on top, clear sky view, far from the mesh radio.
- **Mesh radio antennas**: separated from the sensing antennas by as much
  distance as possible. **Your own datalink is the loudest emitter in the sky**
  — notch-filter its band out of your survey, and log it as self-interference.

---

## 6. Assembly steps

1. Bench-assemble the payload plate: Jetson, X410, GPSDO, NVMe, mesh radio.
2. Wire power with a dedicated filtered regulator; fuse every rail.
3. Connect X410 ↔ Jetson via 10 GbE; verify link at full rate.
4. Connect GPSDO 10 MHz + PPS to X410; verify lock (Doc 03 §5).
5. Attach antennas with filters and LNAs; verify with a VNA that VSWR is sane
   (< 2:1 across band).
6. Connect Jetson ↔ Pixhawk serial; verify MAVLink heartbeat.
7. Mount plate to airframe on vibration isolators.
8. **Weigh everything.** Compare against §7.
9. Run the bench acceptance tests (Doc 10 §2) **before** the first flight.

---

## 7. Weight and power budget

| Item | Mass (approx) | Power |
|---|---|---|
| USRP X410 + enclosure | 1.6 kg | 25 W |
| Jetson AGX Orin + carrier | 0.7 kg | 30 W (capped) |
| GPSDO | 0.3 kg | 2 W |
| NVMe + heatsink | 0.1 kg | 5 W peak |
| Mesh radio + antennas | 0.4 kg | 10 W |
| Low-band antenna + mast | 0.6 kg | — |
| High-band antenna | 0.2 kg | — |
| DF array + 4 LNAs/filters | 0.9 kg | 2 W |
| Cabling, plate, isolators | 0.8 kg | — |
| **Total payload** | **≈ 5.6 kg** | **≈ 74 W** |

Against a 5–15 kg-payload heavy-lift multirotor this leaves margin. Expect
**25–40 min** endurance. If you need more time on station, the **tethered**
variant (power over tether, fixed position) gives unlimited endurance for
persistent overwatch of a known crossing — worth building one of your three
aircraft this way.

---

## 8. Acceptance checks before you fly anything

- [ ] X410 streams IQ to Jetson at target rate with **zero overflows** for 10 min
- [ ] GPSDO reports `LOCKED`; PPS present; time offset stable
- [ ] All four DF channels show correlated phase on a common injected test signal
- [ ] Noise floor with motors **off** vs **on** differs by < 3 dB (if worse, your
      shielding is inadequate — fix before flying, or you will hunt ghosts)
- [ ] MAVLink heartbeat from Pixhawk seen by Jetson
- [ ] Mesh link to ground at planned range
- [ ] NVMe encryption active; seal public key installed
- [ ] Payload mass within airframe limit with 20% margin

---

Next: [`03-software-setup.md`](03-software-setup.md).
