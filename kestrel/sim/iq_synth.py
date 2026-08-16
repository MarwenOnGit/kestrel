"""IQ-level synthesis (Doc 09 §2, "IQ-level sim").

Generates actual waveforms for each signal class so the DSP chain and, later,
the classifier have something real to work on. Slower than the detection-level
simulator by orders of magnitude - use it for DSP validation and for building
classifier training data, not for autonomy development.

Every generator returns unit-power complex baseband at the given sample rate.
"""
from __future__ import annotations

import math

import numpy as np


def _unit_power(x: np.ndarray) -> np.ndarray:
    p = np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12
    return (x / p).astype(np.complex64)


# --------------------------------------------------------------------------
# Analog
# --------------------------------------------------------------------------
def fm_voice(n: int, fs: float, deviation_hz: float = 2500.0,
             audio_hz: float = 800.0, rng=None) -> np.ndarray:
    """Narrowband FM. Constant envelope, ~12.5 kHz occupied."""
    rng = rng or np.random.default_rng()
    t = np.arange(n) / fs
    # Two tones plus wobble: a single tone looks nothing like speech.
    audio = (np.sin(2 * np.pi * audio_hz * t)
             + 0.5 * np.sin(2 * np.pi * (audio_hz * 2.3) * t)
             + 0.3 * rng.standard_normal(n))
    audio = audio / (np.max(np.abs(audio)) + 1e-12)
    phase = 2 * np.pi * deviation_hz * np.cumsum(audio) / fs
    return _unit_power(np.exp(1j * phase))


def am_voice(n: int, fs: float, audio_hz: float = 700.0, depth: float = 0.7,
             rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    t = np.arange(n) / fs
    audio = np.sin(2 * np.pi * audio_hz * t) + 0.3 * rng.standard_normal(n)
    audio /= (np.max(np.abs(audio)) + 1e-12)
    return _unit_power((1.0 + depth * audio).astype(np.complex64))


# --------------------------------------------------------------------------
# Digital voice
# --------------------------------------------------------------------------
def _symbols(n: int, fs: float, symbol_rate: float, levels: np.ndarray, rng):
    sps = max(int(round(fs / symbol_rate)), 1)
    n_sym = int(np.ceil(n / sps)) + 1
    syms = rng.choice(levels, size=n_sym)
    return np.repeat(syms, sps)[:n], sps


def dmr(n: int, fs: float, symbol_rate: float = 4800.0, rng=None) -> np.ndarray:
    """DMR / C4FM-style 4-level FSK. Deviations +/-1944, +/-648 Hz."""
    rng = rng or np.random.default_rng()
    levels = np.array([-1944.0, -648.0, 648.0, 1944.0])
    dev, _ = _symbols(n, fs, symbol_rate, levels, rng)
    return _unit_power(np.exp(1j * 2 * np.pi * np.cumsum(dev) / fs))


def p25(n: int, fs: float, rng=None) -> np.ndarray:
    """P25 phase 1 C4FM: same family as DMR, 4800 baud, different deviations."""
    rng = rng or np.random.default_rng()
    levels = np.array([-1800.0, -600.0, 600.0, 1800.0])
    dev, _ = _symbols(n, fs, 4800.0, levels, rng)
    return _unit_power(np.exp(1j * 2 * np.pi * np.cumsum(dev) / fs))


def tetra(n: int, fs: float, symbol_rate: float = 18_000.0, rng=None) -> np.ndarray:
    """TETRA pi/4-DQPSK: differential phase steps of +/-pi/4, +/-3pi/4."""
    rng = rng or np.random.default_rng()
    steps = np.array([np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4])
    d, _ = _symbols(n, fs, symbol_rate, steps, rng)
    return _unit_power(np.exp(1j * np.cumsum(d) / max(int(round(fs / symbol_rate)), 1)))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def lora(n: int, fs: float, bandwidth_hz: float = 125_000.0, sf: int = 9,
         rng=None) -> np.ndarray:
    """LoRa chirp spread spectrum: linear up-chirps with cyclic shifts.

    Very distinctive in a waterfall, and a common covert tracker.
    """
    rng = rng or np.random.default_rng()
    n_chip = 2 ** sf
    samples_per_chirp = max(int(fs * n_chip / bandwidth_hz), 8)
    out = []
    while sum(len(c) for c in out) < n:
        shift = rng.integers(0, n_chip)
        t = np.arange(samples_per_chirp) / fs
        k = bandwidth_hz / (samples_per_chirp / fs)          # chirp rate
        f0 = -bandwidth_hz / 2 + shift * bandwidth_hz / n_chip
        inst = (f0 + k * t) % bandwidth_hz - bandwidth_hz / 2
        out.append(np.exp(1j * 2 * np.pi * np.cumsum(inst) / fs))
    return _unit_power(np.concatenate(out)[:n])


def ofdm(n: int, fs: float, occupied_hz: float, n_sub: int = 64,
         rng=None) -> np.ndarray:
    """Generic OFDM - stands in for LTE/5G/WiFi.

    Noise-like in the time domain (high PAPR, spectral flatness near 1), which
    is exactly the feature that separates it from constant-envelope voice modes.
    """
    rng = rng or np.random.default_rng()
    used = max(int(n_sub * min(occupied_hz / fs, 1.0)), 4)
    cp_len = max(n_sub // 8, 1)
    ramp = max(cp_len // 4, 1)

    # Raised-cosine edge window. Butting OFDM symbols together leaves a phase
    # discontinuity at every boundary, and that splatters energy across the
    # entire band - enough to lift the measured noise floor by 10 dB and hide
    # weak emitters that are nowhere near this signal. Real transmitters window
    # their symbol edges for exactly this reason, so the synthesizer must too,
    # or the DSP is being tested against a signal no radio would emit.
    edge = 0.5 * (1 - np.cos(np.pi * np.arange(1, ramp + 1) / (ramp + 1)))

    blocks = []
    total = 0
    while total < n + ramp:
        grid = np.zeros(n_sub, dtype=complex)
        qpsk = (rng.choice([-1, 1], used) + 1j * rng.choice([-1, 1], used)) / np.sqrt(2)
        grid[:used // 2] = qpsk[:used // 2]
        grid[-(used - used // 2):] = qpsk[used // 2:]
        sym = np.fft.ifft(grid) * n_sub
        blk = np.concatenate([sym[-cp_len:], sym])          # cyclic prefix
        blk[:ramp] *= edge
        blk[-ramp:] *= edge[::-1]
        blocks.append(blk)
        total += len(blk) - ramp

    # Overlap-add across the ramps so the windowing does not punch gaps.
    stride = len(blocks[0]) - ramp
    buf = np.zeros(stride * len(blocks) + ramp, dtype=complex)
    for k, blk in enumerate(blocks):
        buf[k * stride:k * stride + len(blk)] += blk
    return _unit_power(buf[:n])


def lte(n, fs, rng=None):
    return ofdm(n, fs, occupied_hz=fs * 0.9, n_sub=128, rng=rng)


def nr5g(n, fs, rng=None):
    return ofdm(n, fs, occupied_hz=fs * 0.95, n_sub=256, rng=rng)


def wifi(n, fs, rng=None):
    return ofdm(n, fs, occupied_hz=fs * 0.8, n_sub=64, rng=rng)


def ocusync(n, fs, rng=None):
    return ofdm(n, fs, occupied_hz=fs * 0.85, n_sub=128, rng=rng)


def bluetooth(n: int, fs: float, symbol_rate: float = 1e6, rng=None) -> np.ndarray:
    """GFSK, 2-level."""
    rng = rng or np.random.default_rng()
    dev, _ = _symbols(n, fs, symbol_rate, np.array([-160e3, 160e3]), rng)
    return _unit_power(np.exp(1j * 2 * np.pi * np.cumsum(dev) / fs))


def satcom_terminal(n: int, fs: float, symbol_rate: float = 50_000.0,
                    rng=None) -> np.ndarray:
    """Narrowband BPSK burst, as from an L-band handheld terminal."""
    rng = rng or np.random.default_rng()
    syms, _ = _symbols(n, fs, symbol_rate, np.array([-1.0, 1.0]), rng)
    return _unit_power(syms.astype(np.complex64))


GENERATORS = {
    "fm_voice": fm_voice, "am_voice": am_voice,
    "dmr": dmr, "p25": p25, "tetra": tetra,
    "lora": lora, "lte": lte, "nr5g": nr5g, "wifi": wifi,
    "bluetooth": bluetooth, "ocusync": ocusync,
    "satcom_terminal": satcom_terminal,
}


def generate(signal_class: str, n: int, fs: float, rng=None) -> np.ndarray:
    if signal_class not in GENERATORS:
        raise KeyError(f"no generator for '{signal_class}'. "
                       f"Have: {sorted(GENERATORS)}")
    return GENERATORS[signal_class](n, fs, rng=rng)


# --------------------------------------------------------------------------
# Scene composition
# --------------------------------------------------------------------------
def compose_scene(n: int, rate_sps: float, center_hz: float,
                  signals: list[dict], noise_floor_dbm: float = -110.0,
                  rng=None) -> np.ndarray:
    """Build one wideband IQ frame containing several signals plus noise.

    signals: [{"freq_hz":…, "signal_class":…, "snr_db":…, "bandwidth_hz":…}, …]

    This is what the SDR would hand the Jetson, and what the DSP chain must
    pull detections out of without being told what is present.
    """
    rng = rng or np.random.default_rng()

    # Noise power is the reference; every SNR is relative to it.
    noise_p = 1.0
    scene = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) \
        * np.sqrt(noise_p / 2.0)
    scene = scene.astype(np.complex64)

    for s in signals:
        offset = s["freq_hz"] - center_hz
        if abs(offset) > rate_sps / 2:
            continue                              # outside the tuned window
        bw = s.get("bandwidth_hz", 12_500.0)

        # Front-end roll-off. A real receiver's usable bandwidth is smaller
        # than its sample rate: the anti-alias filter is already falling by the
        # time you reach Nyquist. Without modelling that, a signal sitting on
        # the tuning boundary is placed exactly at fs/2, where its own
        # bandwidth wraps around and it appears at FULL STRENGTH at both edges
        # of the band - an 80 dB emitter that does not exist, in two places at
        # once. The detector cannot be blamed for finding what the
        # synthesizer put there.
        edge_frac = abs(offset) / (rate_sps / 2)
        if edge_frac > 0.9:
            rolloff_db = 60.0 * (edge_frac - 0.9) / 0.1
        else:
            rolloff_db = 0.0

        # Generate at a rate matched to the signal, then interpolate up to the
        # wideband rate - generating narrowband modes directly at 20 Msps is
        # both wasteful and numerically poor.
        gen_fs = max(bw * 4.0, 48_000.0)
        gen_fs = min(gen_fs, rate_sps)
        n_gen = int(np.ceil(n * gen_fs / rate_sps)) + 16
        base = generate(s["signal_class"], n_gen, gen_fs, rng=rng)

        if gen_fs < rate_sps:
            from scipy.signal import resample_poly
            up = int(round(rate_sps / gen_fs))
            # A steep Kaiser interpolation filter. The default gives only
            # ~-75 dBc stopband, which against a 97 dB signal leaves images
            # every gen_fs across the span - the detector duly finds them and
            # reports emitters that do not exist. Those artifacts are a
            # property of the synthesizer, not of radio, so they must not
            # reach the DSP under test.
            base = resample_poly(base, up, 1, window=("kaiser", 16.0))
        base = base[:n]
        if len(base) < n:
            base = np.pad(base, (0, n - len(base)))

        amp = math.sqrt(noise_p * (10.0 ** ((s["snr_db"] - rolloff_db) / 10.0)))
        t = np.arange(n)
        scene = scene + (amp * base * np.exp(2j * np.pi * offset / rate_sps * t)
                         ).astype(np.complex64)

    return scene.astype(np.complex64)
