# 07 — Geolocation

Turning "a signal exists" into "it is at this point on the ground, with this much
uncertainty."

---

## 1. Three techniques, deliberately combined

| Technique | Needs | Good for | Weak when |
|---|---|---|---|
| **AOA** (bearing from array) | Antenna array on one aircraft | Continuous signals, fast | Small array = coarse at VHF |
| **SADF** (synthetic aperture) | Motion + coherent phase | Long signals, **excellent at VHF** | Signal must persist while you fly |
| **TDOA** (arrival time) | ≥3 aircraft + GPSDO | **Short bursts**, wide bandwidth | Needs tight time sync, poor geometry hurts |

They are complementary in exactly the right way: **TDOA catches the brief
transmissions, SADF nails the long ones, AOA gives an instant coarse answer to
bootstrap both.** Fusion (§5) combines whatever is available.

---

## 2. AOA — bearing from the array

With a 4-element uniform circular array, the wave arrives at each element at a
slightly different phase. Standard estimator: **MUSIC** (MUltiple SIgnal
Classification).

```python
# kestrel/air/aoa.py
import numpy as np

def music_bearing(X, array_xy, wavelength_m, n_signals=1, grid_deg=1.0):
    """X: (n_elements, n_snapshots) complex IQ, time-aligned across elements.
       array_xy: (n_elements, 2) element positions in metres.
       Returns (bearing_deg, spectrum) — peak of the MUSIC spectrum."""
    R = (X @ X.conj().T) / X.shape[1]              # covariance
    w, V = np.linalg.eigh(R)
    En = V[:, :-n_signals]                         # noise subspace
    k = 2*np.pi / wavelength_m

    angles = np.arange(0, 360, grid_deg)
    spectrum = np.zeros(len(angles))
    for i, a in enumerate(angles):
        u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        steer = np.exp(1j * k * (array_xy @ u))    # expected phase per element
        proj = En.conj().T @ steer
        spectrum[i] = 1.0 / (np.real(proj.conj() @ proj) + 1e-12)
    return float(angles[np.argmax(spectrum)]), spectrum
```

**Calibration is mandatory.** Real arrays have per-channel phase offsets from
cables, amplifiers, and the airframe. Measure them (Doc 10 §3) and apply:
```python
X_cal = X * np.exp(-1j * stored_phase_offsets[:, None])
```
Uncalibrated arrays produce confident, wrong bearings — the worst failure mode
in the system.

**Attitude correction.** The array bearing is relative to the *airframe*. Rotate
by the aircraft's yaw (and correct for roll/pitch tilt) to get a true bearing:
```python
true_bearing = (array_bearing + yaw_deg) % 360
```
Getting this wrong is the second most common source of confident-but-wrong fixes.
Test it explicitly.

---

## 3. TDOA — arrival time across the swarm

Three aircraft hear the same burst. The **time difference** between each pair
defines a hyperbola of possible source positions; the hyperbolas intersect at
the emitter.

**Step 1 — measure the time differences by cross-correlation:**
```python
def tdoa_estimate(iq_a, iq_b, fs, t_a_ns, t_b_ns, upsample=16):
    """Returns the arrival-time difference in seconds."""
    from scipy.signal import correlate, resample_poly
    a = resample_poly(iq_a, upsample, 1)          # sub-sample resolution
    b = resample_poly(iq_b, upsample, 1)
    xc = correlate(a, b, mode="full")
    lag = (np.argmax(np.abs(xc)) - (len(b) - 1)) / (fs * upsample)
    clock_delta = (t_a_ns - t_b_ns) * 1e-9        # GPSDO-known offset
    return lag + clock_delta
```

**Step 2 — solve for position** (nonlinear least squares over the hyperbolas):
```python
from scipy.optimize import least_squares
C = 299_792_458.0

def solve_tdoa(receiver_positions, tdoas, ref_idx=0):
    """receiver_positions: list of (x, y, z) metres, local ENU frame.
       tdoas: measured (t_i - t_ref) seconds for each i != ref_idx."""
    P = np.array(receiver_positions, dtype=float)
    ref = P[ref_idx]

    def residuals(xy):
        p = np.array([xy[0], xy[1], 0.0])          # assume emitter on ground
        d_ref = np.linalg.norm(p - ref)
        out = []
        for i, td in tdoas.items():
            d_i = np.linalg.norm(p - P[i])
            out.append((d_i - d_ref)/C - td)
        return out

    x0 = P[:, :2].mean(axis=0)                     # start at swarm centroid
    sol = least_squares(residuals, x0, method="lm")
    return sol.x, sol
```

**Step 3 — the error ellipse**, from the Jacobian at the solution:
```python
def tdoa_covariance(sol, sigma_t=20e-9):
    J = sol.jac
    cov = np.linalg.inv(J.T @ J) * (C*sigma_t)**2
    vals, vecs = np.linalg.eigh(cov)
    return {"semi_major_m": float(np.sqrt(5.991*vals[1])),
            "semi_minor_m": float(np.sqrt(5.991*vals[0])),
            "orientation_deg": float(np.degrees(np.arctan2(vecs[1,1], vecs[0,1])))}
```

### The two things that dominate TDOA accuracy

**(a) Timing.** 30 cm per nanosecond. 20 ns of clock error ≈ 6 m of position
error. GPSDO discipline is not optional.

**(b) Geometry — GDOP.** *Geometric Dilution of Precision.* Three aircraft in a
line give terrible fixes perpendicular to that line; three in a wide triangle
around the emitter give excellent ones. GDOP can swing accuracy by **an order of
magnitude** with identical hardware.

```
   BAD geometry (collinear)        GOOD geometry (spread)
        K1  K2  K3                        K1
        ●   ●   ●                        ╱  ╲
            ┊                          K2 ── K3
            ┊  ← huge uncertainty          ●  ← tight ellipse
            ★  along this axis             ★
```

**This is exactly why Information-Gain Flight (Doc 06 §4) exists.** The swarm
actively flies itself into good geometry rather than hoping for it. Hardware
buys you the floor; geometry buys you the accuracy.

---

## 4. SADF — Synthetic Aperture Direction Finding *(signature idea #2)*

**The problem.** At VHF (30–300 MHz) the wavelength is 1–10 m. Good bearing
accuracy needs an array a meaningful fraction of a wavelength across — metres
wide. You cannot hang a 5 m array under a drone. So conventional wisdom says
drones are poor at VHF DF, which is unfortunate because **VHF handhelds are the
primary target**.

**The insight.** You don't need all your antennas to exist at the same *time* —
only at known *positions*. A single antenna, carried along a 50 m flight path,
sampling coherently, is mathematically equivalent to a 50 m antenna array. This
is the same principle as synthetic aperture radar, applied to passive DF.

**What makes it possible here:** the GPSDO gives phase coherence across the whole
flight path, and RTK-grade GPS gives centimetre knowledge of where each sample
was taken.

```python
# kestrel/air/sadf.py
def synthetic_aperture_bearing(samples, positions, wavelength_m,
                               grid_deg=0.5, ref_phase=None):
    """samples:   complex IQ, one per snapshot along the flight path
       positions: (N,2) aircraft position in metres (ENU) for each sample
       Coherently sums along candidate directions — the peak is the bearing."""
    k = 2*np.pi / wavelength_m
    angles = np.arange(0, 360, grid_deg)
    power = np.zeros(len(angles))
    for i, a in enumerate(angles):
        u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        # expected phase at each position for a plane wave from direction a
        expected = np.exp(-1j * k * (positions @ u))
        power[i] = np.abs(np.sum(samples * np.conj(expected)))**2
    return float(angles[np.argmax(power)]), power
```

**Effective aperture** = flight path length. Fly 100 m across while the target
transmits, and at 150 MHz (λ = 2 m) you have a **50-wavelength aperture** — a
bearing resolution far beyond anything a physical drone array can achieve.

**Requirements and honest limits:**
- The emitter must transmit long enough to fly a useful baseline (a few seconds
  for a voice transmission — which is typical for the handheld traffic we target).
- Emitter must be roughly stationary during the pass.
- Phase must stay coherent — **GPSDO mandatory**, and the aircraft's own motion
  must be known precisely (RTK GPS strongly recommended).
- Frequency drift in the *target's* oscillator limits integration time; estimate
  and compensate it, or cap the aperture.

**Curved paths beat straight ones.** A gentle arc gives a 2-D aperture and
resolves the ambiguity a straight line leaves. The planner should prefer arcs
during REFINE — another place where flight path *is* the sensor design.

> **Why this is the standout capability.** It inverts the usual trade-off. A
> conventional system needs a big airframe for a big array; KESTREL gets the big
> array from *motion*, so a modest drone achieves VHF DF performance normally
> associated with fixed ground sites. The aperture is free — you were flying
> anyway.

---

## 5. Fusion — combining everything

The ground station merges every measurement of an emitter into one
maximum-likelihood estimate:

```python
# kestrel/ground/fusion.py
def fuse(track_observations):
    """Weighted nonlinear least squares over mixed measurement types."""
    def residuals(xy):
        r = []
        for o in track_observations:
            if o.type == "bearing":                     # AOA or SADF
                pred = np.degrees(np.arctan2(xy[1]-o.pos[1], xy[0]-o.pos[0]))
                err = (pred - o.bearing_deg + 180) % 360 - 180
                r.append(err / o.sigma_deg)
            elif o.type == "tdoa":
                d1 = np.linalg.norm(xy - o.pos_a[:2])
                d2 = np.linalg.norm(xy - o.pos_b[:2])
                r.append(((d1-d2)/C - o.tdoa_s) / o.sigma_s)
            elif o.type == "rssi":                      # weak, low weight
                r.append((o.predicted_range(xy) - o.range_est_m) / o.sigma_m)
        return r

    sol = least_squares(residuals, x0=initial_guess(track_observations))
    return sol.x, covariance_from(sol)
```

**Weighting is what makes fusion work.** Each measurement contributes in inverse
proportion to its variance, so a 0.5° SADF bearing dominates a 15° coarse AOA
automatically, and a bad RSSI estimate can't drag the answer. Never fuse by
averaging.

**Outlier rejection:** multipath (signals bouncing off terrain) produces
occasional wildly wrong bearings. Use RANSAC or a robust loss (`soft_l1`) — with
a plain quadratic loss, one multipath bearing can move the fix hundreds of
metres.

---

## 6. Reporting honestly

Every fix that leaves this system carries:

```json
{
  "lat": 31.2043, "lon": 34.8117,
  "error_ellipse": {"semi_major_m": 42.0, "semi_minor_m": 18.5,
                    "orientation_deg": 73.0, "confidence": 0.95},
  "method": "tdoa+sadf", "n_aircraft": 3, "n_observations": 47,
  "gdop": 2.3, "quality": "good"
}
```

**Rules for any output that reaches an operator or a court:**
- Never report a bare coordinate. Always the ellipse.
- If GDOP is poor or observations are few, mark `quality: degraded` and say so
  prominently on the map.
- If GPSDO lock was lost, TDOA results are **void**, not merely degraded.
- Distinguish *emitter location* from *person location*. The system locates a
  transmitting device. That is all it locates, and reports must say so.

---

## 7. Expected performance (indicative, good conditions)

| Configuration | Typical 95% error |
|---|---|
| 1 aircraft, AOA only, UHF | 200–800 m |
| 1 aircraft, SADF, VHF, 100 m aperture | 50–150 m |
| 3 aircraft, TDOA, good GDOP | 20–60 m |
| 3 aircraft, TDOA + SADF fused, refined by info-gain flight | **10–30 m** |

The last row is the design target and depends as much on **flight geometry** as
on hardware — which is the whole argument for Doc 06.

---

Next: [`08-legal-and-audit.md`](08-legal-and-audit.md).
