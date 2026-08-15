# 04 — Signal Processing

How raw IQ becomes a list of detections, in real time, on limited hardware.
This is the highest-data-rate part of the system and therefore where the
engineering care goes.

---

## 1. The problem, stated numerically

At 200 MHz bandwidth with 16-bit complex samples:
`200e6 × 2 × 2 bytes = 800 MB/s`.

You cannot store that (a 2 TB disk fills in 40 minutes), you cannot downlink it,
and you cannot run a neural network on all of it. So the pipeline is a
**funnel**: each stage is cheaper per sample than the last is expensive, and
each stage throws away almost everything.

```
 800 MB/s   IQ from SDR
    │  FFT + averaging               (FPGA — nearly free)
 ~2 MB/s    power spectra
    │  CFAR energy tripwire          (CPU — cheap)
 ~50/s      candidate detections
    │  channelize + classify         (GPU — expensive, but rare)
 ~50/s      classified detections
    │  warrant gate                  (trivial)
 ~1/s       authorized recordings    (NVMe)
 ~5 KB/s    reports to ground
```

**The funnel is the whole design.** Every optimization is about pushing work
earlier (cheaper) or making later stages rarer.

---

## 2. Stage 1 — Acquisition

```python
# kestrel/air/sdr_source.py
import uhd, numpy as np

class SdrSource:
    """Streams timestamped IQ frames from a USRP."""

    def __init__(self, center_hz, rate_sps, gain_db=40, channels=(0,)):
        self.usrp = uhd.usrp.MultiUSRP("type=x4xx")
        self.usrp.set_clock_source("gpsdo")
        self.usrp.set_time_source("gpsdo")
        for ch in channels:
            self.usrp.set_rx_rate(rate_sps, ch)
            self.usrp.set_rx_freq(uhd.types.TuneRequest(center_hz), ch)
            self.usrp.set_rx_gain(gain_db, ch)
        self.channels = list(channels)
        self.rate = rate_sps
        self.center = center_hz

        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = self.channels
        self.streamer = self.usrp.get_rx_stream(st_args)
        self.md = uhd.types.RXMetadata()

    def frames(self, frame_len=65536):
        buf = np.zeros((len(self.channels), frame_len), dtype=np.complex64)
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        self.streamer.issue_stream_cmd(cmd)
        while True:
            n = self.streamer.recv(buf, self.md)
            if self.md.error_code != uhd.types.RXMetadataErrorCode.none:
                # 'overflow' means we are too slow — must be zero in production
                yield None, self.md.error_code
                continue
            # GPSDO-derived timestamp: the basis of all TDOA
            t_ns = int(self.md.time_spec.get_real_secs() * 1e9)
            yield buf[:, :n].copy(), t_ns
```

**Rules:**
- Overflows must be **zero**. An overflow is a hole in time and silently
  destroys TDOA validity.
- Always carry `t_ns` with the data. A sample without a timestamp is useless for
  geolocation.
- Set gain deliberately: too low and you miss weak signals; too high and strong
  local signals cause intermodulation (fake signals that aren't there).

---

## 3. Stage 2 — Spectrum and noise floor

```python
# kestrel/air/detector.py
import numpy as np

def welch_psd(iq, nfft=4096, averages=16):
    """Average several FFTs to get a stable power spectrum (dB)."""
    win = np.hanning(nfft).astype(np.float32)
    n = min(averages, len(iq) // nfft)
    acc = np.zeros(nfft, dtype=np.float32)
    for k in range(n):
        seg = iq[k*nfft:(k+1)*nfft] * win
        acc += np.abs(np.fft.fftshift(np.fft.fft(seg)))**2
    psd = acc / max(n, 1)
    return 10*np.log10(psd + 1e-20)

def noise_floor(psd_db, percentile=25):
    """Robust noise floor: signals are the minority of bins, so a low
    percentile approximates 'quiet'. Median would be biased by wide signals."""
    return np.percentile(psd_db, percentile)
```

> **Why a percentile and not a mean.** The mean is dragged up by every strong
> signal, so a busy band would raise its own floor and hide weak emitters —
> exactly the ones we care about. The 25th percentile stays anchored to quiet
> bins.

---

## 4. Stage 3 — CFAR energy tripwire

**CFAR = Constant False Alarm Rate.** Instead of a fixed threshold (which breaks
whenever conditions change), compare each bin against its *local neighbourhood*.
This keeps the false-alarm rate constant as the noise floor drifts with
altitude, temperature, and terrain.

```python
def cfar_detect(psd_db, guard=4, train=32, threshold_db=8.0):
    """Cell-averaging CFAR. Returns indices of bins that exceed their
    local background by threshold_db."""
    n = len(psd_db)
    lin = 10**(psd_db/10.0)
    csum = np.concatenate([[0.0], np.cumsum(lin)])
    hits = []
    half = guard + train
    for i in range(half, n - half):
        lo = csum[i-half] - csum[i-half-0]          # left training start
        left  = csum[i-guard] - csum[i-half]
        right = csum[i+half+1] - csum[i+guard+1]
        bg = (left + right) / (2.0*train)
        if lin[i] > bg * 10**(threshold_db/10.0):
            hits.append(i)
    return _group_adjacent(hits)

def _group_adjacent(idx, max_gap=2):
    """Adjacent hot bins are ONE signal, not many. Group them."""
    if not idx: return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= max_gap: cur.append(i)
        else: groups.append(cur); cur = [i]
    groups.append(cur)
    return groups
```

Each group becomes a **candidate detection** with:
- `freq_hz` = center of the group, mapped from bin index
- `bandwidth_hz` = group width × bin spacing
- `snr_db` = peak power − noise floor

Bin→frequency mapping:
```python
def bin_to_hz(bin_idx, nfft, rate_sps, center_hz):
    return center_hz + (bin_idx - nfft//2) * (rate_sps / nfft)
```

**Tuning `threshold_db`:** start at 8 dB. Lower → more sensitive, more false
alarms → more GPU wakeups → more power. This single number is your main
sensitivity/power dial and should be adaptive (Doc 06 §4 lets the agent tune it
by mission phase).

---

## 5. Stage 4 — Channelize

Cut the candidate out of the wideband stream so the classifier sees only it:

```python
from scipy.signal import resample_poly

def channelize(iq, rate_sps, center_hz, target_hz, target_bw_hz, out_rate=None):
    """Shift the signal of interest to baseband, filter, decimate."""
    out_rate = out_rate or max(target_bw_hz*4, 48000)
    n = np.arange(len(iq))
    # frequency shift
    shifted = iq * np.exp(-2j*np.pi*(target_hz-center_hz)/rate_sps * n)
    # decimate (resample_poly low-pass filters as it decimates)
    up, down = 1, max(1, int(rate_sps // out_rate))
    return resample_poly(shifted, up, down), rate_sps/down
```

Output: a short, narrow IQ snippet — typically 1024–8192 samples. This is what
the AI classifier consumes (Doc 05) and what gets sealed if authorized.

---

## 6. Stage 5 — Feature extraction

Cheap features that (a) help the classifier and (b) are enough on their own for
many decisions:

```python
def features(iq_snippet, fs):
    x = iq_snippet / (np.std(iq_snippet) + 1e-12)   # normalize power away
    inst_amp   = np.abs(x)
    inst_phase = np.unwrap(np.angle(x))
    inst_freq  = np.diff(inst_phase) * fs / (2*np.pi)
    return {
        "papr_db": 20*np.log10(np.max(inst_amp)/(np.mean(inst_amp)+1e-12)),
        "amp_var": float(np.var(inst_amp)),          # AM-ish?
        "freq_var": float(np.var(inst_freq)),        # FM-ish?
        "kurtosis": float(np.mean(inst_amp**4)/(np.mean(inst_amp**2)**2)),
        "spec_flatness": _flatness(x),               # noise-like (OFDM) vs tonal
        "symbol_rate_hz": _estimate_symbol_rate(x, fs),
    }
```

**Normalize power away.** Otherwise the classifier learns "loud = close" instead
of "shape = type," and fails on distant emitters. This is the single most common
mistake in RF machine learning.

---

## 7. Sweep strategy

The X410 sees 400 MHz at once; the mission spans ~8 GHz. So you must sweep — and
sweeping means you are *blind* to everything you're not looking at.

| Strategy | Behavior | Use when |
|---|---|---|
| **Linear sweep** | Step through all bands evenly | Baseline survey |
| **Priority-weighted** | Dwell longer on bands with history of activity | Normal operations |
| **Revisit-driven** | Return to a band before its emitters typically re-key | Known target pattern |
| **Swarm-divided** | Each aircraft owns different segments | **Default — see Doc 06 §5** |

**Revisit time** is the metric that matters: if a smuggler's radio transmits for
4 seconds every 10 minutes, and your revisit time on that band is 12 minutes,
you will miss them. Compute it explicitly:

```
revisit_time = n_segments × dwell_per_segment / n_aircraft
```

With 3 aircraft dividing 20 segments at 200 ms dwell → 1.3 s revisit. That's the
payoff of Spectral Choreography: **the swarm turns a sweep into near-continuous
coverage.**

---

## 8. Where each stage runs

| Stage | Hardware | Rate | Cost |
|---|---|---|---|
| Acquisition | X410 → 10 GbE | 800 MB/s | fixed |
| FFT / PSD | X410 FPGA (preferred) or Jetson GPU | continuous | low |
| Noise floor + CFAR | Jetson CPU | continuous | low |
| Channelize | Jetson CPU/GPU | per candidate | medium |
| Classify | Jetson GPU (TensorRT) | per candidate | high — keep rare |
| Warrant gate | Jetson CPU | per candidate | trivial |
| Seal + write | Jetson CPU → NVMe | per authorized hit | medium |

**Optimization order if you run out of headroom:** (1) push FFT to the FPGA,
(2) raise the CFAR threshold, (3) shrink the classifier input, (4) batch
classifier calls, (5) reduce instantaneous bandwidth. Do **not** solve it by
skipping the warrant gate or by recording more and sorting later.

---

## 9. Tests for this document's code

```python
# tests/test_detector.py
def test_cfar_finds_injected_tone():
    fs, n = 1e6, 65536
    t = np.arange(n)/fs
    noise = (np.random.randn(n) + 1j*np.random.randn(n)) * 0.1
    tone  = np.exp(2j*np.pi*100e3*t)              # +100 kHz from center
    psd = welch_psd(noise + tone)
    groups = cfar_detect(psd)
    assert len(groups) == 1
    f = bin_to_hz(np.mean(groups[0]), 4096, fs, 0.0)
    assert abs(f - 100e3) < 2e3

def test_cfar_quiet_band_is_quiet():
    noise = (np.random.randn(65536) + 1j*np.random.randn(65536)) * 0.1
    assert cfar_detect(welch_psd(noise)) == []   # no false alarms in pure noise
```

Every stage gets tests like these, run against synthetic IQ. You should be able
to validate the entire DSP chain on a laptop with no radio attached.

---

Next: [`05-ai-models.md`](05-ai-models.md).
