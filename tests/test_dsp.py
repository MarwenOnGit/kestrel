"""DSP chain tests (Doc 04 §9). No hardware, no simulator geometry - just
signals in, detections out."""
from __future__ import annotations

import numpy as np
import pytest

from kestrel.air.detector import (bin_to_hz, cfar_detect, channelize, features,
                                  find_candidates, noise_floor, welch_psd)
from kestrel.sim import iq_synth

FS = 1e6
N = 262_144


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def noise(n=N, rng=None, power=1.0):
    rng = rng or np.random.default_rng(0)
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n))
            * np.sqrt(power / 2)).astype(np.complex64)


# ----------------------------------------------------------------- basics
def test_welch_psd_shape_and_noise_level():
    psd = welch_psd(noise())
    assert psd.shape == (4096,)
    # Averaging must tame the variance, or CFAR fires on noise alone.
    assert np.std(psd) < 3.0


def test_noise_floor_is_robust_to_strong_narrowband_signals():
    """The case the percentile estimator exists for.

    Several loud narrow emitters must not lift the measured floor, or a busy
    channel raises its own threshold and hides the weak signals we care about.
    A mean-based estimator moves many dB here.
    """
    n = N
    t = np.arange(n) / FS
    quiet = noise(n)
    loud = quiet.copy()
    for f in (-300e3, -120e3, 90e3, 250e3):
        loud = loud + 40.0 * np.exp(2j * np.pi * f * t).astype(np.complex64)

    nf_quiet = noise_floor(welch_psd(quiet))
    nf_loud = noise_floor(welch_psd(loud))
    assert abs(nf_loud - nf_quiet) < 1.5


def test_noise_floor_does_rise_under_a_band_filling_emitter():
    """A documented LIMIT, not a bug.

    No percentile floor can be robust to a signal occupying most of the band -
    there are simply not enough quiet bins left to measure. A wideband emitter
    (or, in the field, the aircraft's own datalink or ESC noise) genuinely
    raises the measured floor and genuinely costs sensitivity.

    The engineering answers are physical, not statistical: filter the offending
    band out, notch your own datalink (Doc 02 §5), and estimate the floor per
    sweep segment rather than across the whole span. This test pins the
    behaviour so nobody later "fixes" the estimator to hide it.
    """
    n = N
    quiet = noise(n)
    loud = quiet + 30 * iq_synth.ofdm(n, FS, occupied_hz=FS * 0.5,
                                      rng=np.random.default_rng(1))
    psd_loud = welch_psd(loud)
    nf_quiet = noise_floor(welch_psd(quiet))

    occupancy = float(np.mean(psd_loud > nf_quiet + 3.0))
    assert occupancy > 0.8, "expected this emitter to fill the band"
    assert noise_floor(psd_loud) - nf_quiet > 3.0, (
        "a band-filling emitter must show up as a raised floor; if it no "
        "longer does, the estimator is concealing lost sensitivity")


def test_cfar_finds_an_injected_tone_at_the_right_frequency():
    n = N
    t = np.arange(n) / FS
    sig = noise(n) + 5.0 * np.exp(2j * np.pi * 100e3 * t)
    groups = cfar_detect(welch_psd(sig))
    assert len(groups) == 1
    f = bin_to_hz(float(np.mean(groups[0])), 4096, FS, 0.0)
    assert abs(f - 100e3) < 2e3


def test_cfar_reports_nothing_in_pure_noise():
    """False alarms cost GPU wakeups and therefore flight time."""
    total = 0
    for seed in range(8):
        total += len(cfar_detect(welch_psd(noise(rng=np.random.default_rng(seed)))))
    assert total == 0


def test_cfar_separates_two_signals_but_groups_one():
    n = N
    t = np.arange(n) / FS
    sig = (noise(n)
           + 5.0 * np.exp(2j * np.pi * -200e3 * t)
           + 5.0 * np.exp(2j * np.pi * 150e3 * t))
    assert len(cfar_detect(welch_psd(sig))) == 2


# ------------------------------------------------------------- candidates
def test_find_candidates_reports_frequency_bandwidth_and_snr():
    scene = iq_synth.compose_scene(
        N, FS, center_hz=462.5e6,
        signals=[{"freq_hz": 462.520e6, "signal_class": "dmr",
                  "snr_db": 25.0, "bandwidth_hz": 12_500.0}],
        rng=np.random.default_rng(2))
    cands = find_candidates(scene, FS, 462.5e6)
    assert len(cands) == 1
    c = cands[0]
    assert abs(c.freq_hz - 462.520e6) < 4_000
    assert 3_000 < c.bandwidth_hz < 60_000
    assert c.snr_db > 15.0


def test_an_emitter_on_the_tuning_boundary_produces_no_phantom_at_the_far_edge():
    """A signal at exactly fs/2 wraps around the spectrum.

    Found in a swarm sortie: an emitter sitting on the seam between two sweep
    segments was placed at Nyquist, wrapped, and was reported as a 78 dB
    emitter at BOTH edges of the band - 360 phantom detections, 192 of them
    sealed into the vault because they were in-band and looked like FM voice.

    Two independent defences, both asserted here: the synthesizer applies the
    front-end roll-off a real receiver has near Nyquist, and the detector
    discards candidates that land in the outer 2% of the window (the sweep
    overlaps enough that this costs no coverage).
    """
    scene = iq_synth.compose_scene(
        N, FS, center_hz=463.0e6,
        signals=[{"freq_hz": 463.0e6 + FS / 2, "signal_class": "fm_voice",
                  "snr_db": 80.0, "bandwidth_hz": 12_500.0}],
        rng=np.random.default_rng(3))
    assert find_candidates(scene, FS, 463.0e6) == []


def test_the_edge_guard_does_not_swallow_signals_inside_the_band():
    """The guard must be narrow enough to be free: 5% in is still detected."""
    offset = 0.4 * FS / 2
    scene = iq_synth.compose_scene(
        N, FS, center_hz=463.0e6,
        signals=[{"freq_hz": 463.0e6 + offset, "signal_class": "fm_voice",
                  "snr_db": 30.0, "bandwidth_hz": 12_500.0}],
        rng=np.random.default_rng(4))
    cands = find_candidates(scene, FS, 463.0e6)
    assert len(cands) == 1
    assert abs(cands[0].freq_hz - (463.0e6 + offset)) < 5_000


def test_weak_signals_fall_below_detection_and_strong_ones_do_not():
    """Sensitivity is a threshold, not a cliff at 0 dB - verify both sides."""
    def n_found(snr):
        scene = iq_synth.compose_scene(
            N, FS, 462.5e6,
            signals=[{"freq_hz": 462.51e6, "signal_class": "dmr",
                      "snr_db": snr, "bandwidth_hz": 12_500.0}],
            rng=np.random.default_rng(3))
        return len(find_candidates(scene, FS, 462.5e6))

    assert n_found(30.0) >= 1
    assert n_found(-25.0) == 0


def test_multiple_emitters_in_one_frame_are_all_found():
    scene = iq_synth.compose_scene(
        N, FS, 462.5e6,
        signals=[
            {"freq_hz": 462.30e6, "signal_class": "dmr", "snr_db": 25,
             "bandwidth_hz": 12_500},
            {"freq_hz": 462.55e6, "signal_class": "fm_voice", "snr_db": 22,
             "bandwidth_hz": 12_500},
            {"freq_hz": 462.80e6, "signal_class": "satcom_terminal", "snr_db": 28,
             "bandwidth_hz": 50_000},
        ],
        rng=np.random.default_rng(4))
    cands = find_candidates(scene, FS, 462.5e6)
    found = [c.freq_hz for c in cands]
    for want in (462.30e6, 462.55e6, 462.80e6):
        assert any(abs(f - want) < 15e3 for f in found), f"missed {want/1e6:.2f} MHz"


# ------------------------------------------------------------ channelize
def test_channelize_brings_a_signal_to_baseband():
    n = N
    t = np.arange(n) / FS
    sig = (noise(n) + 8.0 * np.exp(2j * np.pi * 200e3 * t)).astype(np.complex64)
    snip, out_fs = channelize(sig, FS, 0.0, target_hz=200e3, target_bw_hz=12_500)
    psd = welch_psd(snip, nfft=1024, averages=8)
    peak_hz = bin_to_hz(float(np.argmax(psd)), 1024, out_fs, 0.0)
    assert abs(peak_hz) < out_fs * 0.05        # now at DC
    assert out_fs < FS                          # and decimated


def test_channelize_rejects_a_neighbouring_channel():
    """Without anti-alias filtering, decimation folds the neighbour on top."""
    n = N
    t = np.arange(n) / FS
    wanted = 2.0 * np.exp(2j * np.pi * 100e3 * t)
    neighbour = 20.0 * np.exp(2j * np.pi * 400e3 * t)
    snip, out_fs = channelize((noise(n) + wanted + neighbour).astype(np.complex64),
                              FS, 0.0, target_hz=100e3, target_bw_hz=12_500)
    psd = welch_psd(snip, nfft=1024, averages=8)
    peak_hz = abs(bin_to_hz(float(np.argmax(psd)), 1024, out_fs, 0.0))
    assert peak_hz < out_fs * 0.1, "the louder neighbour leaked through"


# --------------------------------------------------------------- features
def test_features_are_power_invariant():
    """Scaling amplitude must not change the answer, or the classifier learns
    distance instead of modulation."""
    x = iq_synth.dmr(65536, 48e3, rng=np.random.default_rng(5))
    a = features(x, 48e3)
    b = features(x * 100.0, 48e3)
    for k in a:
        assert abs(a[k] - b[k]) < 1e-3 * max(1.0, abs(a[k])), f"{k} moved with power"


def test_features_separate_constant_envelope_from_ofdm():
    """The single most useful discriminator: voice modes are constant envelope,
    OFDM is noise-like."""
    rng = np.random.default_rng(6)
    f_dmr = features(iq_synth.dmr(65536, 48e3, rng=rng), 48e3)
    f_ofdm = features(iq_synth.ofdm(65536, 2e6, occupied_hz=1.6e6, rng=rng), 2e6)
    assert f_dmr["amp_var"] < f_ofdm["amp_var"]
    assert f_dmr["papr_db"] < f_ofdm["papr_db"]
    assert f_dmr["spectral_flatness"] < f_ofdm["spectral_flatness"]


# -------------------------------------------------------------- generators
@pytest.mark.parametrize("cls", sorted(iq_synth.GENERATORS))
def test_every_generator_produces_finite_unit_power_iq(cls):
    fs = 2e6 if cls in ("lte", "nr5g", "wifi", "ocusync", "bluetooth") else 100e3
    x = iq_synth.generate(cls, 32768, fs, rng=np.random.default_rng(7))
    assert x.dtype == np.complex64
    assert np.all(np.isfinite(x))
    assert abs(float(np.mean(np.abs(x) ** 2)) - 1.0) < 0.15


def test_narrowband_modes_stay_narrow():
    """A DMR signal that occupies half the band would break every downstream
    bandwidth assumption."""
    scene = iq_synth.compose_scene(
        N, FS, 0.0,
        signals=[{"freq_hz": 0.0, "signal_class": "dmr", "snr_db": 30,
                  "bandwidth_hz": 12_500}],
        rng=np.random.default_rng(8))
    c = find_candidates(scene, FS, 0.0)
    assert c and c[0].bandwidth_hz < 100e3
