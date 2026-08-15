"""Detection: wideband IQ in, candidate signals out (Doc 04 §3-5).

This is the top of the funnel and the highest-rate code in the system, so it is
written to be vectorized and boring. Everything here runs continuously; the
expensive AI downstream runs only on what survives.

    IQ -> Welch PSD -> noise floor -> CFAR -> group -> candidates
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
def welch_psd(iq: np.ndarray, nfft: int = 4096, averages: int = 16,
              overlap: float = 0.5) -> np.ndarray:
    """Averaged power spectrum in dB, DC-centred.

    Averaging trades frequency-domain variance for time resolution. A single
    FFT of noise has ~5.6 dB standard deviation, which would set off any
    threshold; averaging 16 brings that to ~1.4 dB.
    """
    if len(iq) < nfft:
        iq = np.pad(iq, (0, nfft - len(iq)))
    # Blackman-Harris, not Hanning. Hanning's first sidelobe is only -31 dB, so
    # one strong emitter smears across the band and masks weak ones nearby -
    # the very situation this system exists to handle. Blackman-Harris pays
    # ~1.5x wider main lobe for -92 dB sidelobes, which is the right trade for
    # surveillance: dynamic range beats frequency resolution here.
    win = np.blackman(nfft).astype(np.float32)
    win_power = np.sum(win ** 2)

    step = max(int(nfft * (1.0 - overlap)), 1)
    starts = range(0, max(len(iq) - nfft + 1, 1), step)
    acc = np.zeros(nfft, dtype=np.float64)
    n = 0
    for s in starts:
        if n >= averages:
            break
        seg = iq[s:s + nfft] * win
        acc += np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
        n += 1
    psd = acc / max(n, 1) / win_power
    return 10.0 * np.log10(psd + 1e-20)


def noise_floor(psd_db: np.ndarray, percentile: float = 25.0) -> float:
    """Robust noise-floor estimate.

    A low percentile, not the mean: signals are a minority of bins, but a wide
    strong one drags the mean up and the band then hides its own weak emitters
    - exactly the ones worth finding.
    """
    return float(np.percentile(psd_db, percentile))


# --------------------------------------------------------------------------
def cfar_detect(psd_db: np.ndarray, guard: int = 4, train: int = 32,
                threshold_db: float = 8.0) -> list[list[int]]:
    """Cell-averaging CFAR (Constant False Alarm Rate).

    Compares each bin against its own local background rather than a fixed
    threshold, so the false-alarm rate stays put as the noise floor drifts with
    altitude, temperature and terrain. Vectorized: this runs on every frame.

    Returns groups of adjacent hot bins - one group is one signal.
    """
    n = len(psd_db)
    half = guard + train
    if n < 2 * half + 1:
        return []

    lin = np.power(10.0, psd_db / 10.0)
    csum = np.concatenate([[0.0], np.cumsum(lin)])

    i = np.arange(half, n - half)
    left = csum[i - guard] - csum[i - half]
    right = csum[i + half + 1] - csum[i + guard + 1]
    background = (left + right) / (2.0 * train)

    hot = lin[i] > background * (10.0 ** (threshold_db / 10.0))
    return _group_adjacent(i[hot].tolist())


def _group_adjacent(idx: list[int], max_gap: int = 2) -> list[list[int]]:
    """Adjacent hot bins are ONE signal, not many."""
    if not idx:
        return []
    groups, cur = [], [idx[0]]
    for k in idx[1:]:
        if k - cur[-1] <= max_gap:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    return groups


# --------------------------------------------------------------------------
def bin_to_hz(bin_idx: float, nfft: int, rate_sps: float, center_hz: float) -> float:
    return center_hz + (bin_idx - nfft // 2) * (rate_sps / nfft)


def hz_to_bin(freq_hz: float, nfft: int, rate_sps: float, center_hz: float) -> float:
    return (freq_hz - center_hz) * nfft / rate_sps + nfft // 2


# --------------------------------------------------------------------------
@dataclass
class Candidate:
    """A signal found in the spectrum, before classification."""
    freq_hz: float
    bandwidth_hz: float
    snr_db: float
    peak_db: float
    noise_floor_db: float
    bin_lo: int
    bin_hi: int


def wideband_detect(psd_db: np.ndarray, nf_db: float,
                    threshold_db: float = 8.0) -> list[list[int]]:
    """Global threshold against the robust noise floor.

    CFAR alone cannot see a signal wider than its own training window: the
    training cells land inside the signal, the local background rises with it,
    and a 200-bin-wide emitter becomes invisible to its own detector. A plain
    threshold against the global floor has no such blind spot. Neither detector
    is sufficient alone - CFAR handles a tilted or drifting floor, this handles
    wide signals - so run both and take the union.
    """
    return _group_adjacent(np.flatnonzero(psd_db > nf_db + threshold_db).tolist())


def _merge_overlapping(groups: list[list[int]]) -> list[list[int]]:
    if not groups:
        return []
    spans = sorted((g[0], g[-1]) for g in groups)
    out = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo <= out[-1][1] + 2:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [list(range(lo, hi + 1)) for lo, hi in out]


def _occupied_span(psd_db: np.ndarray, lo: int, hi: int, nf_db: float,
                   edge_db: float = 3.0, max_expand: int = 4096) -> tuple[int, int]:
    """Expand a detected core outward to where the signal meets the noise floor.

    Detection flags only the strong middle of a signal; reporting that as the
    bandwidth understates it several-fold and breaks every downstream
    assumption (channelizer width, classifier input, occupancy statistics).
    """
    edge = nf_db + edge_db
    n = len(psd_db)
    i = lo
    steps = 0
    while i > 0 and psd_db[i - 1] > edge and steps < max_expand:
        i -= 1
        steps += 1
    j = hi
    steps = 0
    while j < n - 1 and psd_db[j + 1] > edge and steps < max_expand:
        j += 1
        steps += 1
    return i, j


def _power_containment_span(psd_db: np.ndarray, lo: int, hi: int,
                            fraction: float = 0.99) -> tuple[int, int]:
    """Occupied bandwidth by the conventional definition: the narrowest span
    holding `fraction` of the signal power.

    Expanding to a fixed dB-above-floor instead measures how far the skirts
    remain detectable, which for a shaped signal is several times its real
    occupied bandwidth.
    """
    w = np.power(10.0, psd_db[lo:hi + 1] / 10.0)
    total = float(np.sum(w))
    if total <= 0 or len(w) < 3:
        return lo, hi
    c = np.cumsum(w) / total
    tail = (1.0 - fraction) / 2.0
    i = int(np.searchsorted(c, tail))
    j = int(np.searchsorted(c, 1.0 - tail))
    return lo + min(i, len(w) - 1), lo + min(max(j, i), len(w) - 1)


def find_candidates(iq: np.ndarray, rate_sps: float, center_hz: float,
                    nfft: int = 4096, threshold_db: float = 8.0,
                    averages: int = 16) -> list[Candidate]:
    """Full detection stage: IQ -> list of candidate signals."""
    psd = welch_psd(iq, nfft=nfft, averages=averages)
    nf = noise_floor(psd)

    # Expand FIRST, then merge. Two detectors firing on the same signal expand
    # to the same span, so merging before expansion leaves duplicates.
    raw = (cfar_detect(psd, threshold_db=threshold_db)
           + wideband_detect(psd, nf, threshold_db=threshold_db))
    spans = [_occupied_span(psd, g[0], g[-1], nf) for g in raw]
    groups = _merge_overlapping([list(range(a, b + 1)) for a, b in spans])

    out = []
    for g in groups:
        lo, hi = g[0], g[-1]
        lo, hi = _power_containment_span(psd, lo, hi, fraction=0.99)
        peak_bin = int(lo + int(np.argmax(psd[lo:hi + 1])))
        # Centre on the power centroid, not the span midpoint: asymmetric
        # skirts would otherwise bias the reported frequency.
        w = np.power(10.0, psd[lo:hi + 1] / 10.0)
        centroid = float(np.sum(np.arange(lo, hi + 1) * w) / max(np.sum(w), 1e-20))
        bin_hz = rate_sps / nfft
        # Signal power is spread across the occupied bins; SNR must compare
        # integrated signal power against noise in the same bandwidth, or wide
        # signals are systematically under-reported.
        occupied = np.power(10.0, psd[lo:hi + 1] / 10.0).sum()
        noise_in_bw = np.power(10.0, nf / 10.0) * (hi - lo + 1)
        snr = 10.0 * math.log10(max(occupied - noise_in_bw, 1e-12)
                                / max(noise_in_bw, 1e-20))
        out.append(Candidate(
            freq_hz=bin_to_hz(centroid, nfft, rate_sps, center_hz),
            bandwidth_hz=(hi - lo + 1) * bin_hz,
            snr_db=snr,
            peak_db=float(psd[peak_bin]),
            noise_floor_db=nf,
            bin_lo=lo, bin_hi=hi,
        ))
    return out


# --------------------------------------------------------------------------
def channelize(iq: np.ndarray, rate_sps: float, center_hz: float,
               target_hz: float, target_bw_hz: float,
               out_rate_sps: float | None = None) -> tuple[np.ndarray, float]:
    """Shift a signal of interest to baseband, low-pass, decimate.

    The classifier sees only this narrow snippet, never the wideband stream.
    """
    from scipy.signal import resample_poly

    out_rate = out_rate_sps or max(target_bw_hz * 4.0, 24_000.0)
    n = np.arange(len(iq))
    shifted = iq * np.exp(-2j * np.pi * (target_hz - center_hz) / rate_sps * n)

    decim = max(int(rate_sps // out_rate), 1)
    if decim > 1:
        # resample_poly low-pass filters as it decimates, which is the
        # anti-aliasing we need - decimating without it folds neighbouring
        # channels on top of the one we want.
        shifted = resample_poly(shifted, 1, decim)
    return shifted.astype(np.complex64), rate_sps / decim


# --------------------------------------------------------------------------
def features(iq: np.ndarray, fs: float) -> dict[str, float]:
    """Cheap modulation features. Power is normalized away deliberately: a
    classifier trained on absolute level learns 'loud = close' instead of
    'shape = type' and fails at exactly the range where it matters."""
    if len(iq) < 8:
        return {}
    x = np.asarray(iq, dtype=np.complex64)
    x = x / (np.std(x) + 1e-12)

    amp = np.abs(x)
    phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)

    spec = np.abs(np.fft.fft(x)) ** 2 + 1e-20
    geo = np.exp(np.mean(np.log(spec)))
    arith = np.mean(spec)

    # How smoothly the instantaneous frequency changes. A chirp sweeps
    # linearly, so its frequency DERIVATIVE is nearly constant and this ratio
    # collapses (~0.1); modes whose frequency jumps between symbols sit far
    # higher (~0.4-1.4). This is what separates LoRa from other
    # constant-envelope modes of similar bandwidth, where amplitude features
    # are useless.
    d_inst = np.diff(inst_freq) if len(inst_freq) > 2 else np.array([0.0])
    ramp_ratio = float(np.std(d_inst) / (np.std(inst_freq) + 1e-12))

    return {
        "freq_ramp_ratio": ramp_ratio,
        "papr_db": float(20 * np.log10(np.max(amp) / (np.mean(amp) + 1e-12))),
        "amp_var": float(np.var(amp)),
        "freq_var": float(np.var(inst_freq)),
        "kurtosis": float(np.mean(amp ** 4) / (np.mean(amp ** 2) ** 2 + 1e-12)),
        "spectral_flatness": float(geo / arith),      # ~1 noise-like (OFDM), <<1 tonal
        "zero_cross_rate": float(np.mean(np.abs(np.diff(np.sign(x.real)))) / 2.0),
    }
