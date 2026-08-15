# 12 — Getting Started, and What Will Go Wrong

Written before implementation begins, so it can be checked against reality later.
Two parts: **how to start** (§1–4) and **what will hurt** (§5–12), ranked by how
much damage each does and when it bites.

---

## 1. The first two weeks (do these in parallel)

### Track A — Software (you, today, no hardware)
**Day 1: the vertical slice.** Before building any subsystem properly, wire the
thinnest possible end-to-end path with every stage stubbed:

```
synthetic IQ → FFT/detect → classifier(stub: always "dmr")
  → warrant gate (real) → sealer (real) → report JSON → print to console
```

One day of work. It is throwaway-quality and that's fine. What it buys you: the
shape of every interface, discovered by building rather than by guessing. Almost
every integration problem you would otherwise hit in month three shows up here
instead, when it's free to fix.

Then thicken one stage at a time, keeping the whole path runnable at every
commit. **Never have a period where the system doesn't run end to end.**

**Days 2–10:** Steps 0–2 of the implementation plan (repo, simulator, legal
spine). All laptop work.

### Track B — Hardware intuition (in parallel, ~$50)
Buy an **RTL-SDR dongle** this week. It is nothing like the X410 and that's not
the point. Spend three evenings with GQRX or SDR++:
- Watch the FM broadcast band. See what a strong signal looks like.
- Find a local analog voice repeater. Watch it key up and drop.
- Look at the 433 MHz ISM band. Watch car remotes and weather sensors burst.
- Deliberately overload the front end with too much gain and watch fake signals
  appear across the spectrum.

That last one teaches you more about RF than a week of reading. You cannot debug
this system without an intuition for what spectrum *looks like*, and that
intuition is cheap to buy now and expensive to lack later.

### Track C — Bureaucracy (start immediately — longest lead time)
File the requests today, because they take months and everything waits on them:
- Legal authority template + the six questions in Doc 08 §8
- Aviation authorization for autonomous BVLOS operation
- Key custody decision (who holds the seal private key)
- Spectrum/receive authorization if your jurisdiction requires it

**This is very likely your critical path — not the engineering.** Teams
consistently discover this in month five.

---

## 2. Order of construction, and why

Doc 11 gives the task list. The reasoning behind the sequence:

1. **Simulator first.** Without it you debug RF, software, and flight
   simultaneously, and you will not know which one is lying to you.
2. **Legal spine second.** Retrofitted collection control is control you cannot
   defend, and it constrains every interface downstream.
3. **DSP third.** Fixes the data shapes the classifier will consume.
4. **AI fourth.** Needs the DSP output and needs real recordings.
5. **Autonomy fifth.** Pure logic; hardware only adds noise to its development.
6. **Hardware last.** By then everything downstream is proven, so bring-up
   debugs one variable at a time.

---

## 3. Team shape

This project spans RF engineering, DSP, machine learning, flight systems,
cryptography, and law. **Nobody has all six.** Realistic minimum:

| Role | Owns | Can't be shared with |
|---|---|---|
| RF/DSP engineer | Subsystems A, C, calibration | — |
| Software/ML engineer | Subsystems B, F, G | — |
| Flight/safety engineer | PX4, airframe, flight ops | Should not also own autonomy logic |
| Compliance owner | Subsystem E legal, key custody | **Must be independent of flight ops** |

If you are a team of one or two, build Phase 1 only, and treat the swarm as a
later program. A solo Phase 1 is realistic. A solo swarm is not.

---

## 4. Rules that will save you months

1. **Everything runs in simulation before it flies.** No exceptions.
2. **Log obsessively, from day one.** You cannot attach a debugger to an aircraft
   at 120 m. Structured logs are your *only* instrument. Budget real effort here
   early — it feels premature and it isn't.
3. **Record everything you ever receive** (subject to the warrant gate). Real IQ
   is the scarcest resource in the project. Every bench test, every flight —
   keep it, label it, index it. Your classifier is made of this.
4. **Version and sign every model.** You will need to state which model produced
   a classification, months later, possibly in court.
5. **One variable at a time during bring-up.** New antenna *and* new firmware
   *and* new flight profile = an unsolvable debugging problem.
6. **Never let a test go yellow.** `over_collection_count == 0` is binary.

---

## 5. Problem tier 1 — the ones most likely to derail you

### 5.1 The drone is a jammer, and it's jamming you
**Severity: critical. Bites at: first tethered flight.**

Motor ESCs, power converters, and digital electronics radiate broadband noise.
Mounting a sensitive receiver 30 cm from four ESCs is, from an RF standpoint,
absurd — and it's what you're doing. Expect the noise floor to rise **10–20 dB**
when motors spin up. That is a 10–100× loss in sensitivity: emitters you detected
at 5 km on the bench vanish at 1 km in the air.

Worse: it is frequency-dependent and lumpy, so it looks like signals. You will
chase ghosts.

**Mitigations, in order of effectiveness:** physical separation (antenna on a
mast, as far from the frame as tolerable) · shielded payload enclosure ·
ferrites and pi-filters on every power lead · separate regulated rail for the
SDR · low-noise ESCs · characterize the noise signature and subtract it as a
known interferer.

**Do not skip the "motors off vs. motors on" measurement in Doc 02 §8.** If the
delta is >3 dB, fix it before flying. Teams that defer this end up rebuilding the
payload bay.

### 5.2 Your own datalink deafens your receiver
**Severity: high. Bites at: first integrated test.**

The mesh radio transmits watts, centimetres from a receiver trying to hear
picowatts. Even out-of-band, it will desensitize the front end.

**Mitigations:** maximum physical separation and orthogonal polarization ·
notch filters for the mesh band · time-division (don't receive while
transmitting — costs throughput but works) · put the mesh in a band far from your
priority targets · **always** notch your own band out of the survey plan, or
every sortie "discovers" your datalink.

### 5.3 The classifier works in the lab and fails in the field
**Severity: high. Bites at: first real-data evaluation, ~month 3.**

You will train on synthetic + clean bench recordings, hit 96% accuracy, feel
great, then see 60% on real airborne data. This is the **domain gap**, and it is
the single most predictable disappointment in the project.

Causes: synthetic signals lack real hardware impairments · bench recordings have
high SNR and no Doppler · real signals are weak, multipath-distorted, Doppler-
shifted by aircraft motion, and interfered with · real class distribution is
wildly imbalanced (99% boring).

**Mitigations:** train across −5…+30 dB SNR from the start · aggressive
augmentation (frequency offset, phase, Doppler, multipath, interference) · get
real recordings as early as possible, even poor ones · **split by recording
session, never randomly** — random splits will report fake accuracy and you will
believe it · measure accuracy per SNR bucket, not just overall · budget real time
for a labeling workflow, because you will relabel repeatedly.

**Plan for the classifier to be mediocre for a long time.** The architecture
tolerates this: `unknown` is safe, and the RF Weather Map catches what the
classifier misses. Don't let a weak classifier block progress on everything else.

### 5.4 Confident, wrong bearings
**Severity: high. Bites at: ground calibration, then again in flight.**

DF is the subsystem most likely to lie to you convincingly. Sources of error:
- **Uncalibrated channel phase** — cables, amplifiers, connectors. Never zero.
- **Airframe reflections** — the drone's own body distorts the antenna pattern,
  producing azimuth-dependent bias that is not constant.
- **Multipath** — terrain and buildings reflect; you get a strong, clean, wrong
  bearing pointing at a cliff face.
- **Attitude correction bugs** — mixing up yaw conventions, degrees/radians,
  clockwise/counterclockwise, magnetic/true north. The fixes still look
  plausible, which is why this bug survives.

**Mitigations:** calibrate on a surveyed emitter and recalibrate often ·
characterize bias vs. azimuth on a rotating mount, store the correction table ·
robust loss / RANSAC in fusion so one multipath bearing can't move the fix ·
test attitude correction explicitly (point the airframe different directions at a
fixed emitter; true bearing must not move) · **validate ellipse containment** —
if truth isn't inside the 95% ellipse 19 times in 20, your uncertainty is
dishonest and every downstream decision is compromised.

### 5.5 Aviation approval, not engineering, sets your schedule
**Severity: critical to the program, zero to the code. Bites at: month 4–8.**

Autonomous BVLOS multi-aircraft operation is among the hardest approvals to
obtain. Even for government operators it involves safety cases, airworthiness
arguments, and often a pilot-in-command requirement per aircraft — which
partially defeats the autonomy premise.

**Mitigations:** start the conversation in week one · design the safety layer to
be *presentable* as a safety case (this is a real reason the safety layer is
hard-coded and inspectable) · have a tethered variant ready, since tethered
operation often falls under far lighter rules and can deliver operational value
while approvals grind · keep a manual-supervision mode available.

---

## 6. Problem tier 2 — expensive but survivable

### 6.1 Sample drops
Zero overflows on a workstation ≠ zero overflows on a Jetson under thermal load
with the mesh radio saturating the same bus. Every dropped sample is a hole in
time that silently invalidates TDOA. **Mitigations:** jumbo frames + kernel
buffer tuning (Doc 03 §3) · CPU affinity and real-time priority for the
streaming thread · monitor and *log* the overflow count as a first-class health
metric, and mark affected fixes as suspect.

### 6.2 Thermal
Jetson at 30 W plus a USRP at 25 W in a sealed enclosure, in the sun, with no
forward airspeed while hovering. It will throttle, and throttling silently halves
your processing rate — which looks like a software bug. **Mitigations:** design
airflow early · use motor downwash · monitor temperature and degrade gracefully
(reduce bandwidth, not correctness) · test on a hot day, not just in the lab.

### 6.3 Timing is harder than it reads
GPSDO cold-start lock takes minutes. Lock can drop in flight. Holdover drift
degrades TDOA quietly. And the operating areas you care about are exactly where
GPS jamming and spoofing are most likely.

**Mitigations:** monitor lock continuously and **void** (not degrade) TDOA on
loss · log holdover duration and widen error estimates with it · consider
chip-scale atomic clocks for holdover if the budget allows · verify swarm
pairwise offset before every sortie · have a GPS-denied fallback mode (AOA and
SADF still work with degraded position knowledge; TDOA does not).

### 6.4 Geometry beats hardware
You can spend heavily on receivers and still get 500 m fixes because three
aircraft drifted into a line. GDOP swings accuracy by an order of magnitude.
**Mitigation:** this is precisely what information-gain flight exists for — but
verify it's actually working by logging GDOP over time. It should trend down
during REFINE. If it doesn't, your planner has a bug, and the fixes will still
*look* fine.

### 6.5 Multi-agent bugs that appear once an hour
Gossip protocols fail in ways unit tests don't catch: network partition →
split-brain where two aircraft claim the same sector, or none claim a sector ·
oscillation where aircraft flip between contacts, never refining any · a
recruitment stampede that abandons survey coverage · stale peer entries keeping
a dead aircraft's sector "covered."

**Mitigations:** run thousands of randomized simulated sorties with injected
partitions and aircraft failures · add hysteresis/dwell minimums to every state
transition · assert invariants continuously in sim (sector union = all segments;
no aircraft in CONTACT for more than N seconds) · make the division function
deterministic from the live-peer set (as designed) so recovery needs no
negotiation.

### 6.6 Endurance is brutal
30–40 minutes, minus transit, minus reserve, leaves perhaps 15–20 minutes on
station. Ambitions will exceed this constantly.
**Mitigations:** stage from close to the area of interest · build one tethered
aircraft for persistent overwatch · treat battery as a budgeted resource in the
planner (the travel-cost term exists for this) · hot-swap batteries and fly
relays.

### 6.7 Storage and data management
2 TB fills fast, and a season of sorties produces a data-management problem
nobody planned for: indexing, retention deadlines, certified destruction, and
finding the one recording an analyst needs eight months later.
**Mitigation:** design the indexing and retention tooling in Phase 1, when there
are ten files instead of ten thousand.

---

## 7. Problem tier 3 — quiet, chronic, corrosive

- **Calibration decay.** Everything drifts — cables, amplifiers, oscillators,
  mounts. Without a scheduled recalibration regime, accuracy degrades so slowly
  that nobody notices until fixes are consistently wrong. Put calibration date in
  the evidence manifest so it becomes visible.
- **The demo trap.** A polished demo in ideal conditions creates expectations
  that operational reality won't meet, and the gap gets blamed on the engineers.
  Demo with the error ellipses visible and the failure modes shown.
- **Scope creep against the gate.** Someone will ask "while we're up there, can
  it also just check…". Each request is individually reasonable; collectively
  they dissolve the warrant discipline that makes the system lawful. The gate is
  a technical control precisely so the answer can be "the token doesn't allow it"
  rather than a negotiation.
- **Over-trusting the fix.** Operators will start treating a coordinate as
  ground truth and stop reading the ellipse. Make degraded fixes visually
  unmistakable, and never let a bare coordinate leave the system.
- **Model drift.** New area, new season, new hardware → accuracy silently drops.
  Monitor per-class confidence distributions over time; a shifting distribution
  is the early warning.
- **Institutional memory loss.** Six months in, nobody remembers why the CFAR
  threshold is 8 dB or why channel 3 has a 12° offset. Write the *why* down at
  the moment you decide it.

---

## 8. Problems specific to this design's ambitions

Being fair about where the creative ideas are hardest:

| Idea | Where it will fight you |
|---|---|
| **Information-gain flight** | Needs a decent initial track to compute gain against. Cold start (huge covariance) makes the gradient meaningless — you need a bootstrap heuristic for the first few observations. Also, the planner optimizes for *one* emitter; multi-emitter arbitration is genuinely unsolved in the current design. |
| **Synthetic aperture DF** | Requires coherent phase over seconds. Target oscillator drift, aircraft position error, and any timing glitch destroy coherence. Expect it to work beautifully in sim and take months to work in the air. Have AOA as the fallback. |
| **RF Weather Map** | Needs baseline sorties *before* operations, and "normal" varies more than you expect (weekday/weekend, season, harvest, holidays). Expect a high false-anomaly rate initially. It improves with data, so deploy it early even when it's noisy — but don't let anyone act on it alone. |
| **Sealed collection** | Operationally awkward: analysts can't triage in the field, and every export needs two people. Expect pressure to weaken it. Don't. The awkwardness *is* the control. |
| **Spectral choreography** | Interleaved sector division makes revisit-time reasoning harder to explain to operators, and recruitment logic is the most likely source of emergent misbehavior. Keep a "dumb" fixed-division mode as a fallback you can switch to when debugging. |

---

## 9. The two failure modes that kill projects like this

**1. Trying to build the whole thing at once.** The swarm, the AI, the autonomy,
and the legal system are each a real project. Attempted simultaneously, nothing
reaches usable quality and there's no working artifact to show at month six.
*Counter:* Phase 1 is deliberately small and genuinely useful on its own. Ship it.

**2. Discovering that the boring parts dominate.** People join this project
excited about AI and autonomy, then spend eight months on EMI, calibration,
dropped samples, and thermal management. That is not a detour — **that is the
project.** The AI is a few weeks of work sitting on top of a mountain of RF
engineering.
*Counter:* budget accordingly, staff accordingly, and set expectations with
stakeholders on day one.

---

## 10. Leading indicators you're in trouble

Check monthly:

- [ ] The end-to-end path hasn't run in over a week → integration debt accruing
- [ ] Simulator hasn't been validated against real measurements in a month → the
      sim is drifting into fiction
- [ ] Anyone quotes classifier accuracy without an SNR breakdown → self-deception
- [ ] Error ellipse containment hasn't been measured recently → dishonest output
- [ ] Nobody has verified the audit chain on real sortie data → the legal spine
      is theoretical
- [ ] `over_collection_count` was ever nonzero and got "explained" → stop
      everything
- [ ] Calibration is more than 20 flight hours old → bearings are decaying
- [ ] The compliance owner reports to flight operations → structural conflict
- [ ] Somebody proposed skipping a test-ladder rung to hit a date → it will cost
      more than the date is worth

---

## 11. What I'd actually do in month one

1. Vertical slice, day one (§1).
2. Simulator to the point where a fake aircraft finds a fake emitter and plots an
   ellipse. Feels like a toy. It is the most valuable asset you'll build.
3. Warrant gate with its full test suite. Get compliance to review the *tests*,
   not the prose.
4. RTL-SDR evenings, building intuition.
5. Legal and aviation requests filed.
6. One real recording of one real signal class, start to finish, through the
   whole pipeline. Even a bad one. It will reveal five assumptions you didn't
   know you had made.

Do **not** buy the X410 in month one. Buy it when the software is waiting on it —
otherwise it sits in a box while you write Python, and you'll have spent the
budget before learning what you actually need.

---

## 12. A realistic timeline

| Period | Milestone | Confidence |
|---|---|---|
| Month 1 | Simulator + legal spine + vertical slice | High |
| Month 2 | DSP chain, classifier v1 on synthetic data | High |
| Month 3 | Autonomy agent flying in sim; Phase 1 software complete | Medium-high |
| Month 4 | Hardware bring-up; bench tests pass | Medium (EMI risk) |
| Month 5–6 | Tethered, then single free flight | Medium (approval risk) |
| Month 7–9 | Phase 2: DF array, GPSDO, three aircraft | Low-medium |
| Month 10–12 | Phase 3 features; operational sorties | Low |

The confidence column matters more than the dates. Anything below "high" will
slip, and the slips will come from **approvals and EMI**, not from software.
