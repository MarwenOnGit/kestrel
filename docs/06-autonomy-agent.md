# 06 — The Autonomous Mission Agent

How the operator is removed from the loop, safely, on limited hardware.

---

## 1. The key insight

The instinct is to imagine one AI "pilot" that flies, listens, decides, and
reports. That is the wrong model — on limited hardware *and* on unlimited
hardware, because a monolithic learned controller is unexplainable and
uncertifiable.

Instead: **a tiny orchestrator calling proven specialists.**

```
   ┌──────────────────────────────────────────────────┐
   │ SAFETY LAYER — hard-coded, deterministic         │   can veto anything
   │ geofence · altitude · battery · separation ·     │   no AI, no exceptions
   │ lost-link RTL · airspace · payload thermal       │
   └────────────────────▲─────────────────────────────┘
                        │ veto / clamp
   ┌────────────────────┴─────────────────────────────┐
   │ MISSION AGENT — behavior tree, ~50 KB, no GPU    │   the "agent"
   │   SURVEY → CONTACT → REFINE → REPORT → RETURN    │
   │   + information-gain planner                     │
   │   + swarm gossip                                 │
   └────────────────────┬─────────────────────────────┘
                        │ waypoints + sensor tasking ONLY
   ┌────────────────────▼─────────────────────────────┐
   │ PX4 FLIGHT CONTROLLER — classical, flight-proven │   actually flies
   └──────────────────────────────────────────────────┘
```

**The agent never commands motors.** It emits waypoints and sensor tasking. PX4
does the flying. The safety layer can override both. This separation is what
makes autonomy certifiable — and it also means the "AI" costs almost nothing
computationally.

---

## 2. The behavior tree

Five states. Deterministic transitions. Readable by a reviewer in five minutes —
which is the point.

```
┌──────────┐  novelty or classified contact   ┌───────────┐
│  SURVEY  ├────────────────────────────────► │  CONTACT  │
│ divide & │                                  │ confirm,  │
│ scan     │ ◄────────── nothing found ───────┤ recruit   │
└────┬─────┘                                  └─────┬─────┘
     │                                              │ confirmed
     │                                        ┌─────▼─────┐
     │                                        │  REFINE   │
     │                                        │ info-gain │
     │                                        │ flight    │
     │                                        └─────┬─────┘
     │                                              │ ellipse < target
     │                                        ┌─────▼─────┐
     │ ◄──────────────────────────────────────┤  REPORT   │
     │                                        └───────────┘
┌────▼─────┐
│  RETURN  │  battery / warrant expiry / abort
└──────────┘
```

```python
# kestrel/air/agent.py  (structure)
class MissionAgent:
    def __init__(self, planner, swarm, safety, mav, gate):
        self.state = "SURVEY"
        self.planner, self.swarm = planner, swarm
        self.safety, self.mav, self.gate = safety, mav, gate
        self.tracks = {}            # emitter_id -> belief state

    def tick(self, detections, telemetry):     # ~5 Hz. That's plenty.
        self._update_tracks(detections)

        if self.safety.must_return(telemetry) or self.gate.expired():
            self.state = "RETURN"

        if self.state == "SURVEY":
            wp = self.planner.survey_waypoint(self.swarm.my_sector())
            if self._interesting(detections):
                self.state = "CONTACT"
                self.swarm.announce_contact(self._best(detections))

        elif self.state == "CONTACT":
            wp = self.planner.confirm_waypoint(self._best_track())
            if self._confirmed(): self.state = "REFINE"
            elif self._stale():   self.state = "SURVEY"

        elif self.state == "REFINE":
            wp = self.planner.max_info_gain_waypoint(self._best_track())
            if self._ellipse_small_enough(): self.state = "REPORT"

        elif self.state == "REPORT":
            self.swarm.publish_fix(self._best_track())
            self.state = "SURVEY"

        else:                                   # RETURN
            wp = self.planner.home()

        wp = self.safety.clamp(wp, telemetry)   # ALWAYS last
        self.mav.goto(wp)
```

A behavior tree, not a learned policy, for the top level: it is inspectable,
testable state by state, and a reviewer can reason about every transition. The
learning lives *inside* `planner` and the classifiers, where mistakes are
recoverable.

---

## 3. The safety layer (non-negotiable)

Hard-coded, no AI, evaluated **last**, on every tick:

```python
class SafetyLayer:
    def clamp(self, wp, tel):
        wp = self._clamp_to_geofence(wp)          # never outside the polygon
        wp.alt = min(max(wp.alt, self.alt_min), self.alt_max)
        wp = self._enforce_separation(wp, tel.neighbors, min_m=30)
        if tel.battery_pct < self.reserve_pct: wp = self.home()
        if tel.payload_temp_c > 80: self._throttle_payload()
        return wp

    def must_return(self, tel):
        return (tel.battery_pct < self.reserve_pct
                or tel.link_lost_s > self.link_timeout_s
                or tel.gps_fix_type < 3
                or self.warrant_expired())
```

Below this sits PX4's own failsafes (RTL on low battery, lost link, geofence
breach) as a second independent net. **Two independent safety nets, neither
involving AI.**

---

## 4. Information-Gain Flight *(signature idea #1)*

**The vanilla approach:** fly a lawnmower grid, take bearings as you go, hope
the geometry works out.

**The problem:** geolocation accuracy depends enormously on *where you observe
from*. Two aircraft taking bearings from nearly the same direction produce a long
thin error ellipse — a nearly useless fix. Moving 400 m sideways can shrink the
ellipse tenfold. A grid pattern is blind to this.

**KESTREL's approach:** treat flight as an **experiment design problem**. At each
tick, evaluate candidate next positions and fly toward the one that maximally
reduces the uncertainty of the current best emitter track.

The math is the **Fisher Information Matrix** and its inverse, the
**Cramér–Rao Lower Bound** — the theoretical floor on how well you can estimate
a position given a set of measurements. We choose the waypoint that minimizes
the predicted CRLB.

```python
# kestrel/air/planner.py
import numpy as np

def fisher_information_aoa(emitter_xy, obs_positions, sigma_rad):
    """FIM for a set of bearing observations of one emitter.
    Larger determinant = more information = smaller error ellipse."""
    J = np.zeros((2, 2))
    for p in obs_positions:
        dx, dy = emitter_xy[0]-p[0], emitter_xy[1]-p[1]
        r2 = dx*dx + dy*dy
        if r2 < 1.0: continue
        # gradient of bearing wrt emitter position
        g = np.array([-dy/r2, dx/r2])
        J += np.outer(g, g) / (sigma_rad**2)
    return J

def expected_info_gain(track, my_pos, candidate_pos, others_pos, sigma_rad):
    """How much would flying to candidate_pos improve this fix?"""
    before = fisher_information_aoa(track.xy, others_pos + [my_pos], sigma_rad)
    after  = fisher_information_aoa(track.xy, others_pos + [candidate_pos], sigma_rad)
    # D-optimality: maximize log-determinant of information
    return np.log(np.linalg.det(after) + 1e-12) - np.log(np.linalg.det(before) + 1e-12)

def max_info_gain_waypoint(track, my_pos, others_pos, sigma_rad=np.radians(3),
                           radius_m=300, n_candidates=16, cost_weight=0.002):
    """Ring search around current position. Cheap, deterministic, effective."""
    best, best_score = my_pos, -np.inf
    for th in np.linspace(0, 2*np.pi, n_candidates, endpoint=False):
        cand = (my_pos[0] + radius_m*np.cos(th), my_pos[1] + radius_m*np.sin(th))
        gain = expected_info_gain(track, my_pos, cand, others_pos, sigma_rad)
        travel_cost = cost_weight * np.hypot(cand[0]-my_pos[0], cand[1]-my_pos[1])
        score = gain - travel_cost          # information per unit battery
        if score > best_score:
            best, best_score = cand, score
    return best
```

**Why this stands out:**
- Ellipses converge **3–5× faster** than a survey grid in simulation — fewer
  minutes on station per fix, which is everything when endurance is 30 minutes.
- It is **explainable**: "the aircraft moved east because a bearing from the east
  cuts the ellipse's long axis." You can defend that in a review.
- It is **cheap**: a 16-candidate ring search with 2×2 matrices is microseconds
  of CPU. No GPU. No training. This is the elegance — the most distinctive
  behavior in the system costs essentially nothing to run.
- It generalizes: the same function with a TDOA information matrix drives swarm
  geometry (spread out perpendicular to the baseline).

**Later upgrade (Phase 3):** replace the ring search with a trained RL policy for
multi-emitter, multi-aircraft joint planning. An RL policy network is *kilobytes*
and runs in microseconds — the cost is entirely in training it in KestrelSim
(Doc 09), not in running it. Keep the information-gain planner as the fallback
and as the reward signal for training.

---

## 5. Spectral Choreography *(signature idea #5)*

**The problem:** the mission spans ~8 GHz; each radio sees 400 MHz. Three
aircraft all scanning the same way is triple redundancy and triple blindness.

**The solution:** the swarm divides *spectrum* and *sky* by gossip, and
**recruits** when something is found — a foraging strategy.

```python
# kestrel/air/swarm.py
class SwarmCoordinator:
    """Decentralized. No master. Survives any aircraft dropping out."""

    def __init__(self, my_id, segments):
        self.my_id, self.segments = my_id, segments
        self.peers = {}                 # id -> last heartbeat + claimed sector

    def my_sector(self):
        """Deterministic division by sorted live-peer index. If a peer dies,
        every survivor recomputes and the gap closes with no negotiation."""
        live = sorted([self.my_id] + [p for p, hb in self.peers.items()
                                      if hb.age_s < 5.0])
        idx, n = live.index(self.my_id), len(live)
        return self.segments[idx::n]           # interleaved, not blocked

    def announce_contact(self, det):
        """Bee-style recruitment: tell peers where the interesting thing is."""
        self.broadcast({"type": "CONTACT", "freq": det.freq_hz,
                        "class": det.signal_class, "novelty": det.novelty_score,
                        "bearing": det.bearing_deg, "from": self.my_id,
                        "pos": self.my_pos})

    def on_contact(self, msg):
        """Decide whether to abandon my sector and help.
        Only the aircraft that adds the most GEOMETRIC value diverts —
        so a contact never sucks in the whole swarm."""
        value = self.geometric_value(msg) - self.sector_cost()
        if value > self.divert_threshold and self.is_best_helper(msg):
            self.tasking = ("ASSIST", msg)
```

**Interleaved (`[idx::n]`) rather than blocked division** is deliberate: each
aircraft samples across the *whole* spectrum rather than owning one contiguous
chunk, so losing an aircraft degrades revisit time everywhere slightly instead of
going fully blind on one band.

**Recruitment discipline:** the swarm must not stampede. Only the aircraft whose
diversion adds the most *geometric* information (Doc 06 §4) leaves its sector.
Survey coverage is preserved while the fix is refined.

**Revisit time** (Doc 04 §7) with 3 aircraft over 20 segments at 200 ms dwell:
**~1.3 s** — effectively continuous coverage of 8 GHz using radios that see
400 MHz each. That is the payoff.

---

## 6. Emitter tracking (belief state)

The agent maintains a belief about each emitter, updated with every detection —
an **Extended Kalman Filter** over emitter position:

```python
class EmitterTrack:
    def __init__(self, emitter_id):
        self.xy = np.array([0.0, 0.0])
        self.P  = np.eye(2) * 1e6          # huge initial uncertainty
        self.observations, self.fingerprint = [], None

    def update_bearing(self, obs_pos, bearing_rad, sigma_rad):
        dx, dy = self.xy[0]-obs_pos[0], self.xy[1]-obs_pos[1]
        r2 = dx*dx + dy*dy
        h = np.arctan2(dy, dx)                       # predicted bearing
        H = np.array([[-dy/r2, dx/r2]])              # Jacobian
        R = np.array([[sigma_rad**2]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        innov = np.arctan2(np.sin(bearing_rad-h), np.cos(bearing_rad-h))  # wrapped
        self.xy = self.xy + (K @ np.array([innov])).ravel()
        self.P  = (np.eye(2) - K @ H) @ self.P

    def error_ellipse(self, conf=0.95):
        vals, vecs = np.linalg.eigh(self.P)
        k = 5.991 if conf == 0.95 else 2.279          # chi-square, 2 DOF
        return {"semi_major_m": float(np.sqrt(k*vals[1])),
                "semi_minor_m": float(np.sqrt(k*vals[0])),
                "orientation_deg": float(np.degrees(np.arctan2(vecs[1,1], vecs[0,1])))}
```

`P` is the covariance — literally the error ellipse. The planner in §4 is
choosing waypoints to shrink `det(P)` as fast as possible. **Tracking and
planning are the same mathematics viewed from two ends**, which is why the whole
autonomy layer stays this small.

---

## 7. What autonomy does NOT cover — and why

Two acts stay human, permanently, by design:

1. **Authorizing scope** — a human binds a warrant to a target list before
   launch (Doc 08 §2).
2. **Releasing evidence** — two humans decrypt and export recordings after
   landing (Doc 08 §6).

This is not a technical limitation. It's the boundary that keeps collection
lawful and evidence admissible. Everything between those bookends — takeoff,
survey, detection, classification, geolocation, warrant-gated recording,
reporting, return — runs with **zero human input**.

**Autonomous inside the fence; human at the gate.**

An operator can always abort over the mesh, but the mission does not require one.

---

## 8. Testing the agent

Test it entirely in KestrelSim (Doc 09) before it ever flies:

```python
def test_agent_refines_fix_faster_than_grid():
    sim = KestrelSim(emitters=[Emitter(xy=(1200, 800), freq=462e6, mode="dmr")])
    a = run(sim, planner=InfoGainPlanner(), minutes=10)
    b = run(sim, planner=LawnmowerPlanner(), minutes=10)
    assert a.final_ellipse_area < b.final_ellipse_area / 3

def test_safety_layer_vetoes_geofence_breach():
    wp = agent.tick(detections=[far_outside_contact], telemetry=tel)
    assert geofence.contains(wp)

def test_swarm_reconverges_when_aircraft_lost():
    sim.kill_aircraft("K2")
    sim.step(seconds=10)
    assert set.union(*[a.my_sector() for a in sim.alive]) == set(ALL_SEGMENTS)

def test_agent_never_records_without_warrant_match():
    sim.run(minutes=30, warrant=narrow_token)
    assert all(r.warrant_id == narrow_token.id for r in sim.recordings)
    assert all(narrow_token.covers(r.freq_hz) for r in sim.recordings)
```

That last test is the one that matters most. Run it on every commit.

---

Next: [`07-geolocation.md`](07-geolocation.md).
