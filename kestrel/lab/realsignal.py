"""Turn a real recorded satellite pass into IQ the pipeline can eat (Doc 15).

Public amateur-satellite recordings are almost all FM-discriminator audio: the
ground station has already demodulated the RF. KESTREL's pipeline wants
complex IQ. `fm_remodulate` puts the audio back on a carrier, which recovers
the transmitted waveform - real bit timing, real symbol shaping, real filter
response of the satellite's radio and of the receiver, real noise from the
pass - but NOT the original RF impairments (Doppler drift, fading, adjacent
signals). Results built on it are labelled "re-modulated", never "RF".

The one free parameter is the FM deviation, because the audio's absolute
scale is not recorded. It is set from the signal's own peak level and a
stated nominal deviation, and every result reports which value was used.
"""
from __future__ import annotations

import wave

import numpy as np


def load_wav(path: str, max_s: float | None = None) -> tuple[np.ndarray, float]:
    """Mono float audio in [-1, 1) and its sample rate. Stereo is taken as the
    left channel: these recordings are discriminator audio, not I/Q pairs."""
    w = wave.open(path)
    fs = float(w.getframerate())
    n = w.getnframes() if max_s is None else min(w.getnframes(), int(max_s * fs))
    raw = w.readframes(n)
    if w.getsampwidth() != 2:
        raise ValueError(f"{path}: only 16-bit PCM is supported")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if w.getnchannels() == 2:
        x = x[0::2]
    return x, fs


def fm_remodulate(audio: np.ndarray, fs: float, deviation_hz: float = 3000.0,
                  peak_percentile: float = 99.5) -> np.ndarray:
    """Complex baseband FM whose instantaneous frequency is the audio.

    The audio is DC-removed (DC in discriminator output is just carrier
    offset) and scaled so its `peak_percentile` magnitude maps to
    `deviation_hz`."""
    x = audio - np.mean(audio)
    peak = np.percentile(np.abs(x), peak_percentile) + 1e-12
    inst_freq = x / peak * deviation_hz
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    return np.exp(1j * phase).astype(np.complex64)


def upsample_fft(x: np.ndarray, factor: int) -> np.ndarray:
    """Band-limited integer upsampling by zero-padding the spectrum. Exact for
    a periodic signal; the wrap-around at the ends is irrelevant here because
    frames are taken from the middle of a burst."""
    n = len(x)
    X = np.fft.fft(x)
    m = n * factor
    Y = np.zeros(m, dtype=np.complex128)
    h = n // 2
    Y[:h] = X[:h]
    Y[-(n - h):] = X[h:]
    return (np.fft.ifft(Y) * factor).astype(np.complex64)


def embed_in_scene(bb: np.ndarray, fs_in: float, factor: int, offset_hz: float,
                   snr_db: float, rng: np.random.Generator,
                   noise_bw_hz: float = 20_000.0) -> tuple[np.ndarray, float]:
    """Place the baseband signal at `offset_hz` inside a window `factor` times
    wider, over complex white noise. `snr_db` is defined over `noise_bw_hz`
    (about one amateur-satellite FSK channel), which is how a receiver's
    detector sees it - not over the whole window.

    Returns (iq, rate_sps)."""
    rate = fs_in * factor
    up = upsample_fft(bb, factor)
    t = np.arange(len(up)) / rate
    sig = up * np.exp(2j * np.pi * offset_hz * t).astype(np.complex64)
    p_sig = np.mean(np.abs(sig) ** 2)
    n0 = p_sig / (10 ** (snr_db / 10)) / noise_bw_hz          # noise PSD, W/Hz
    sigma = np.sqrt(n0 * rate / 2)
    noise = sigma * (rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig)))
    return (sig + noise).astype(np.complex64), rate


def busiest_window(audio: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    """The `seconds`-long stretch with the most signal variance - the burst,
    not the silence either side of it."""
    n = int(seconds * fs)
    if len(audio) <= n:
        return audio
    hop = max(n // 8, 1)
    best, best_v = 0, -1.0
    for i in range(0, len(audio) - n, hop):
        v = float(np.var(audio[i:i + n]))
        if v > best_v:
            best, best_v = i, v
    return audio[best:best + n]
