# KESTREL

**An autonomous airborne RF collection swarm for authorized spectrum monitoring,
emitter geolocation, and lawful signal collection.**

*A kestrel hovers in place and hunts by seeing trails that are invisible to
everything else on the field. That is the whole design brief.*

---

## What this is

Three or more drones fly over an area that authorities already have cause to
monitor. Each carries a wideband software-defined radio, a small AI computer,
and a precise clock. Flying with **no operator in the loop**, the swarm:

1. listens across 30 MHz – 8 GHz (handheld radios, phones, satellite terminals,
   trackers, drone links),
2. identifies what each transmission *is* using onboard AI,
3. calculates **where on the ground** each transmitter sits, by comparing what
   several aircraft hear at the same instant,
4. records only what a cryptographically signed warrant permits, in a form the
   drone itself cannot read back,
5. reports live to a ground map, and builds a multi-day picture of who transmits,
   when, and from where.

A human sets the legal scope before launch and releases evidence after landing.
Everything in between is autonomous.

## What this is not

Not a jammer, not a spoofer, not a decryption tool, not a way to operate outside
a warrant. There is no transmit-side capability in this design at all. See
[`docs/08-legal-and-audit.md`](docs/08-legal-and-audit.md).

---

## What is built, and what has been measured

The design is ambitious; the implementation is Phase 1, **in simulation**,
plus a first contact with real signals. Every number below says where it
came from.

- **DSP chain on IQ.** Welch PSD, CFAR combined with a global threshold,
  channelization, 99%-power bandwidth, 12 synthesized waveform classes. A
  sortie's detections are found in samples, not derived from geometry.
  Eighteen bugs found by measurement are written up with their physics in
  [Doc 13](docs/13-implementation-status.md).
- **Real satellite downlinks** ([Doc 15](docs/15-real-signals.md)). On 92
  public recordings, re-modulated to IQ: detection in 90–92/92, median
  frequency error 65 Hz. AX.25 frames from 10 satellites decode with valid
  FCS, OPS-SAT among them. The rule-based classifier **failed**: it labelled
  every satellite DMR or FM voice above the gate's threshold. That is recorded,
  mitigated in the warrant, and turned into requirements for the trained
  model.
- **Legal spine, attacked** ([Doc 14](docs/14-threat-model.md)). Warrants
  are Ed25519-signed, time-boxed and geofenced. Recordings are sealed to the
  ground *and* signed by the airframe, so they are both unreadable and
  attributable. The audit chain carries signed checkpoints checked against a
  downlink. Warrant limits survive reboots and refuse clock rollback. Each
  threat has a test, and each residual risk is stated.
- **Geolocation and autonomy.** Bearing-only batch MLE with a
  systematic-error budget; information-gain flight measured at 3.6× lower
  median fix error than a lawnmower search (simulator, 8 seeds). Ideas 2, 3
  and 5 below are design only.

---

## The five ideas that make it different

| # | Idea | Why it matters |
|---|---|---|
| 1 | **Information-Gain Flight** | The swarm flies toward *maximum expected uncertainty reduction*, not a search grid. 3.6× lower fix error than a grid in simulation; *built*. → [`06`](docs/06-autonomy-agent.md) |
| 2 | **Synthetic Aperture DF** | The aircraft's own motion synthesizes an antenna array far larger than the airframe — VHF direction-finding a drone shouldn't be able to do. *Design only.* → [`07`](docs/07-geolocation.md) |
| 3 | **RF Weather Map** | An unsupervised, per-tile, per-hour model of "normal." Anomalies need no labels, so novel emitters are caught. *Design only.* → [`05`](docs/05-ai-models.md) |
| 4 | **Sealed Collection** | Recordings are encrypted to a ground-held key; the drone cannot read its own take. Each recording is signed by the airframe. The warrant is a signed, time-boxed, geofenced token; on expiry the vault closes. *Built, threat-modelled.* → [`08`](docs/08-legal-and-audit.md), [`14`](docs/14-threat-model.md) |
| 5 | **Spectral Choreography** | Aircraft divide spectrum and sky by gossip and *recruit* each other onto contacts, covering 8 GHz with radios that see 400 MHz each. *Design only.* → [`06`](docs/06-autonomy-agent.md) |

---

## Documentation — read in this order

| Doc | Contents |
|---|---|
| [`00-primer-rf-from-zero.md`](docs/00-primer-rf-from-zero.md) | **Start here if you have never done radio work.** Every concept from scratch. |
| [`01-architecture.md`](docs/01-architecture.md) | The whole system, subsystem by subsystem, with interface contracts. |
| [`02-hardware-build.md`](docs/02-hardware-build.md) | Bill of materials, wiring, antenna construction, assembly steps. |
| [`03-software-setup.md`](docs/03-software-setup.md) | Exact commands to bring up the ground station and each aircraft. |
| [`04-signal-processing.md`](docs/04-signal-processing.md) | The DSP chain: sweep, FFT, detect, channelize. With code. |
| [`05-ai-models.md`](docs/05-ai-models.md) | Classifier, RF Weather Map, fingerprinting. Training → shrinking → deploying. |
| [`06-autonomy-agent.md`](docs/06-autonomy-agent.md) | The autonomous mission agent, information-gain planner, swarm choreography. |
| [`07-geolocation.md`](docs/07-geolocation.md) | AOA, TDOA, synthetic aperture DF, fusion math. |
| [`08-legal-and-audit.md`](docs/08-legal-and-audit.md) | Warrant tokens, sealed collection, hash-chained audit, chain of custody. |
| [`09-simulator.md`](docs/09-simulator.md) | KestrelSim — the digital twin you develop and train against. |
| [`10-testing-and-flight-ops.md`](docs/10-testing-and-flight-ops.md) | Bench → replay → tethered → free flight. Checklists. |
| [`11-phase1-implementation-plan.md`](docs/11-phase1-implementation-plan.md) | **The step-by-step build plan.** Start executing here. |
| [`12-getting-started-and-risks.md`](docs/12-getting-started-and-risks.md) | How to start, and what will go wrong, written before implementation. |
| [`13-implementation-status.md`](docs/13-implementation-status.md) | **What is built, what is measured, every bug found.** |
| [`14-threat-model.md`](docs/14-threat-model.md) | The legal spine, attacked: what is fixed, what is open. |
| [`15-real-signals.md`](docs/15-real-signals.md) | First contact with real satellite downlinks: what held, what broke. |
| [`docs/superpowers/specs/`](docs/superpowers/specs/) | The formal design specification. |

## Build phases

- **Phase 1 — One aircraft.** Survey, detect, classify, record under warrant,
  report to a map. Proves the RF chain *and* the legal controls end to end.
- **Phase 2 — The swarm.** Add the precise clock, DF array, three aircraft,
  TDOA/AOA fusion, information-gain flight.
- **Phase 3 — Full autonomy + intelligence.** RF Weather Map, fingerprinting,
  pattern-of-life database, lawful-intercept demodulation, RL positioning.

Each phase ends in a flight test and a sign-off.

## Quick status

**Phase 1 in progress, in simulation.** Steps 0–3, 5 and 6 of
[`docs/11-phase1-implementation-plan.md`](docs/11-phase1-implementation-plan.md)
are built: simulator, legal spine, DSP chain, geolocation, autonomy agent — and
the DSP chain now flies inside the mission loop, so a sortie's detections are
found in IQ samples rather than derived from geometry. Still open: the trained
classifier (Step 4), the ground map (Step 7), and everything needing hardware
(Step 8).

    pip install -r requirements.txt                              # numpy, scipy, pynacl, pytest
    python3 -m pytest tests/ -q                                  # see Doc 13 §1 for counts
    python3 tools/real_signals.py tests/data                     # real satellite downlinks
    python3 -m kestrel.sim.run --scenario border_crossing --iq   # fly a sortie on IQ
    python3 tools/dsp_demo.py                                    # the DSP funnel, one frame

Read [`docs/13-implementation-status.md`](docs/13-implementation-status.md) for
what is measured, what is known broken, and every bug found so far, then
[`14`](docs/14-threat-model.md) and [`15`](docs/15-real-signals.md).

## Before you spend money

Four questions in [`docs/08-legal-and-audit.md`](docs/08-legal-and-audit.md)
§7 must be answered by your legal and aviation authorities first. They change
data schemas, not just paperwork.
