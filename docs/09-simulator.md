# 09 — KestrelSim (the digital twin)

**Build this first.** It is the highest-leverage component in the project.

---

## 1. Why the simulator comes before the hardware

You cannot develop an autonomous swarm by flying it. Real sorties cost hours,
batteries, airspace approvals, and crashes. Worse, they aren't *repeatable* — you
can never re-run the exact same RF conditions to see whether your change helped.

KestrelSim gives you:
- **Repeatability.** Same seed, same scenario, every time.
- **Speed.** 30 simulated minutes in seconds. Thousands of sorties overnight.
- **Ground truth.** You know exactly where the emitters are, so you can measure
  geolocation error honestly instead of guessing.
- **Free failures.** Crash the swarm a million times at no cost.
- **RL training environment** for Phase 3 (Doc 06 §4).
- **Regression testing.** Every commit re-runs the full mission.

A rule that will save the project: **every feature is developed against the
simulator and merely *validated* on hardware.**

---

## 2. What it must model (and what it must not)

| Model | Fidelity needed | Why |
|---|---|---|
| Emitter behavior (on/off, duty cycle, power, mode) | **High** | Drives revisit-time and detection logic |
| Path loss + terrain occlusion | **Medium-high** | Determines what's detectable from where |
| Noise floor + interference | Medium | Drives CFAR tuning |
| Aircraft motion | Medium | Waypoint following, wind, battery drain |
| GPSDO timing error | **High** | TDOA accuracy is entirely about this |
| Antenna patterns | Medium | Bearing bias, nulls |
| Multipath | Low-medium | Enough to test outlier rejection |
| Full IQ waveforms | **Only when testing DSP** | Expensive; use "detection-level" sim for autonomy |

**Two fidelity levels — this is the key design decision:**

- **Detection-level sim (fast).** Skips waveforms entirely; produces `Detection`
  objects directly from geometry and a propagation model. Runs ~1000× real time.
  Use for autonomy, swarm, planner, and RL training.
- **IQ-level sim (slow).** Synthesizes actual waveforms. Use for classifier
  training data, DSP validation, and TDOA/SADF math.

Most development uses the fast one.

---

## 3. Core simulator

```python
# kestrel/sim/world.py
import numpy as np

C = 299_792_458.0

class Emitter:
    """A transmitter on the ground with realistic on/off behavior."""
    def __init__(self, xy, freq_hz, mode, eirp_dbm=37,
                 duty=0.02, burst_s=4.0, period_s=600.0, mobile=False):
        self.xy, self.freq_hz, self.mode = np.array(xy, float), freq_hz, mode
        self.eirp_dbm, self.duty = eirp_dbm, duty
        self.burst_s, self.period_s, self.mobile = burst_s, period_s, mobile
        self.fingerprint = np.random.randn(64)      # unique per physical radio
        self.fingerprint /= np.linalg.norm(self.fingerprint)

    def is_transmitting(self, t):
        return (t % self.period_s) < self.burst_s

    def step(self, dt):
        if self.mobile:
            self.xy += np.random.randn(2) * 0.5 * dt     # slow drift

class Aircraft:
    def __init__(self, ac_id, xy, alt=120.0, speed=15.0):
        self.id, self.xy, self.alt, self.speed = ac_id, np.array(xy, float), alt, speed
        self.battery_pct = 100.0
        self.clock_error_ns = np.random.randn() * 15.0   # GPSDO residual error

    def step(self, dt, waypoint):
        d = np.array(waypoint) - self.xy
        n = np.linalg.norm(d)
        if n > 1e-6:
            self.xy += d/n * min(self.speed*dt, n)
        self.battery_pct -= dt * (100.0 / (35*60))       # ~35 min endurance
        self.clock_error_ns += np.random.randn() * 0.5   # slow drift


def path_loss_db(d_m, freq_hz, terrain=None):
    """Free-space path loss + optional terrain occlusion penalty."""
    if d_m < 1.0: d_m = 1.0
    fspl = 20*np.log10(d_m) + 20*np.log10(freq_hz) - 147.55
    if terrain is not None and terrain.is_occluded(...):
        fspl += terrain.excess_loss_db
    return fspl


class KestrelSim:
    def __init__(self, emitters, aircraft, noise_floor_dbm=-110.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.emitters, self.aircraft = emitters, aircraft
        self.noise_floor, self.t = noise_floor_dbm, 0.0

    def step(self, dt, waypoints):
        self.t += dt
        for e in self.emitters: e.step(dt)
        for a in self.aircraft: a.step(dt, waypoints[a.id])
        return self.generate_detections()

    def generate_detections(self):
        out = []
        for a in self.aircraft:
            for e in self.emitters:
                if not e.is_transmitting(self.t): continue
                d = np.linalg.norm(e.xy - a.xy)
                rx_dbm = e.eirp_dbm - path_loss_db(d, e.freq_hz)
                snr = rx_dbm - self.noise_floor
                if snr < 5.0: continue                     # below detection
                # bearing error scales inversely with SNR — this is real physics
                sigma = np.radians(max(0.5, 30.0/np.sqrt(max(snr, 1))))
                true_b = np.arctan2(e.xy[1]-a.xy[1], e.xy[0]-a.xy[0])
                out.append(Detection(
                    aircraft_id=a.id, freq_hz=e.freq_hz, snr_db=snr,
                    signal_class=e.mode,
                    bearing_deg=np.degrees(true_b + self.rng.normal(0, sigma)),
                    bearing_sigma_deg=np.degrees(sigma),
                    t_gpsdo_ns=int((self.t + d/C)*1e9 + a.clock_error_ns),
                    aircraft_pos=a.xy.copy(),
                    truth_emitter=e))                      # ground truth, sim only
        return out
```

**Note the two details that make it useful:** bearing error scales with SNR (so
the planner learns that flying closer helps), and each aircraft carries an
independent clock error (so TDOA accuracy degrades realistically).

---

## 4. Scenarios

Define missions as data, so they're reproducible and shareable:

```yaml
# kestrel/sim/scenarios/border_crossing.yaml
name: "Night crossing, VHF handhelds + satcom"
area: {min_xy: [0, 0], max_xy: [8000, 8000]}
duration_s: 1800
noise_floor_dbm: -112
emitters:
  - {xy: [2100, 3400], freq_hz: 462.5e6, mode: dmr,  eirp_dbm: 37, burst_s: 6, period_s: 240}
  - {xy: [2350, 3510], freq_hz: 462.5e6, mode: dmr,  eirp_dbm: 37, burst_s: 4, period_s: 240}
  - {xy: [5900, 1200], freq_hz: 1626e6,  mode: satcom_terminal, eirp_dbm: 45, burst_s: 60, period_s: 1800}
  - {xy: [4000, 4000], freq_hz: 868e6,   mode: lora, eirp_dbm: 14, burst_s: 0.5, period_s: 60, mobile: true}
decoys:
  - {xy: [1000, 7000], freq_hz: 462.5e6, mode: dmr, eirp_dbm: 37, note: "legitimate farm radio"}
aircraft: 3
warrant:
  freq_ranges_hz: [[462.0e6, 468.0e6]]
  signal_classes: [dmr, fm_voice]
```

**Include decoys.** A scenario where every signal is a target teaches nothing. The
legitimate farm radio on the same frequency is exactly the case where your warrant
gate, geofence, and reporting honesty get tested.

**Scenario library to build:**
`border_crossing`, `maritime_smuggling`, `urban_dense_spectrum`,
`mountain_occlusion`, `jamming_present`, `aircraft_failure`,
`gps_denied`, `warrant_expiry_midflight`, `decoy_heavy`.

---

## 5. Metrics — how you know you're improving

```python
# kestrel/sim/metrics.py
def evaluate(sim_run):
    return {
        "detection_rate":        found_emitters / total_emitters,
        "time_to_first_fix_s":   ...,
        "median_fix_error_m":    ...,     # vs. ground truth
        "p95_fix_error_m":       ...,
        "final_ellipse_area_m2": ...,
        "revisit_time_s":        ...,
        "false_fix_rate":        ...,     # fixes >500 m from any real emitter
        "classifier_accuracy":   ...,
        "over_collection_count": 0,       # MUST be zero, always
        "battery_efficiency":    fixes_per_kwh,
    }
```

`over_collection_count` must be **zero in every scenario, every run**. Treat any
nonzero value as a build-breaking failure, not a tuning issue.

---

## 6. Using the sim to train the RL planner (Phase 3)

```python
# kestrel/sim/gym_env.py
class KestrelEnv(gym.Env):
    """Observation: own position, peer positions, track means + covariances,
       battery, sector assignment.
       Action:      next waypoint (continuous, bounded to a local ring).
       Reward:      information gain − travel cost − time penalty
                    − LARGE penalty for any safety violation."""

    def step(self, action):
        dets = self.sim.step(dt=1.0, waypoints=self._to_waypoints(action))
        self._update_tracks(dets)
        reward = (self._info_gain()
                  - 0.001*self._distance_travelled()
                  - 0.01
                  - 100.0*self._safety_violations())
        return self._obs(), reward, self._done(), {}
```

Train with PPO or SAC. The resulting policy is **tens of kilobytes** and runs in
microseconds on the Jetson — again, the cost is training, not inference.

**Two non-negotiables when deploying an RL policy:**
1. The safety layer (Doc 06 §3) still wraps it. RL output is a *suggestion*.
2. It must beat the information-gain planner on the frozen scenario suite before
   it flies. Keep the analytic planner as the permanent fallback.

---

## 7. Validating the simulator against reality

A simulator you haven't validated is a confident liar. After the first real
sorties:

1. Fly a known test emitter at a surveyed location.
2. Replay the same geometry in KestrelSim.
3. Compare: detection ranges, SNR vs. distance, bearing error distribution,
   fix error.
4. Tune `path_loss_db`, `noise_floor`, and bearing-sigma models until simulated
   and measured distributions agree.
5. **Re-validate after any hardware change.**

Track the sim-vs-real gap as a first-class metric. When it grows, trust the sim
less — and say so in reports.

---

## 8. Running it

```bash
# Single scenario, with visualization
python3 -m kestrel.sim.run --scenario border_crossing --render

# Regression suite (every commit)
python3 -m kestrel.sim.run --all-scenarios --seeds 20 --assert-metrics

# Planner comparison
python3 -m kestrel.sim.compare --planners infogain,lawnmower,random --seeds 50
```

The middle command belongs in CI. It is what stops a clever refactor from
silently making the swarm worse — or, far more seriously, from breaking the
over-collection guarantee.

---

Next: [`10-testing-and-flight-ops.md`](10-testing-and-flight-ops.md).
