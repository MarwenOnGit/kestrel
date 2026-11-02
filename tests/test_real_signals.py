"""KESTREL against signals nobody synthesized (Doc 15).

Fixtures are real amateur-satellite downlinks (tests/data/README.md). Two
questions: does the protocol decoder recover real frames - checked by the
FCS, so a pass is not a matter of opinion - and what does the airborne DSP
chain make of a real waveform it was never designed around.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from kestrel.air.pipeline import EdgePipeline, FeatureClassifier, SweepSegment
from kestrel.air.warrant_gate import WarrantGate
from kestrel.common import crypto
from kestrel.common.warrant import WarrantToken
from kestrel.ground import signer
from kestrel.ground.audit import AuditLog
from kestrel.lab.ax25 import (crc16_x25, decode_afsk1200, decode_fsk9600_g3ruh,
                              g3ruh_descramble, hdlc_frames, nrzi_decode,
                              parse_ax25)
from kestrel.lab.realsignal import (busiest_window, embed_in_scene,
                                    fm_remodulate, load_wav)

DATA = pathlib.Path(__file__).resolve().parent / "data"
CENTER = 436.0e6
OFFSET = 250e3          # where the satellite is placed in the tuned window


# --------------------------------------------------------- encoder (tests only)
def _addr(call, ssid, last):
    return bytes(ord(c) << 1 for c in call.ljust(6)) + bytes([0x60 | ssid << 1 | last])


def _ui(info=b"hello kestrel"):
    return _addr("CQ", 0, 0) + _addr("TEST", 1, 1) + b"\x03\xf0" + info


def _hdlc(raw, flags=24):
    data = raw + crc16_x25(raw).to_bytes(2, "little")
    bits, ones = [], 0
    for byte in data:
        for i in range(8):
            b = byte >> i & 1
            bits.append(b)
            ones = ones + 1 if b else 0
            if ones == 5:
                bits.append(0)
                ones = 0
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    return np.array(flag * flags + bits + flag * 4, dtype=np.uint8)


def _nrzi(bits):
    out, lv = [], 0
    for b in bits:
        lv ^= b == 0
        out.append(lv)
    return np.array(out, dtype=np.uint8)


def _g3ruh_scramble(bits):
    s = np.zeros(len(bits), dtype=np.uint8)
    for n, b in enumerate(bits):
        s[n] = b ^ (s[n - 12] if n >= 12 else 0) ^ (s[n - 17] if n >= 17 else 0)
    return s


# ---------------------------------------------------------- protocol layer
def test_hdlc_nrzi_round_trip():
    assert hdlc_frames(nrzi_decode(_nrzi(_hdlc(_ui())))) == [_ui()]


def test_fcs_rejects_a_single_flipped_bit():
    bits = _hdlc(_ui())
    bits[24 * 8 + 40] ^= 1
    assert hdlc_frames(bits) == []


def test_g3ruh_descrambler_inverts_the_scrambler():
    bits = np.random.default_rng(0).integers(0, 2, 500).astype(np.uint8)
    assert np.array_equal(g3ruh_descramble(_g3ruh_scramble(bits))[17:], bits[17:])


def test_afsk1200_synthetic_round_trip():
    levels = _nrzi(_hdlc(_ui()))
    fs, sps = 48_000, 40
    f = np.where(np.repeat(levels, sps) == 1, 1200.0, 2200.0)
    audio = np.sin(2 * np.pi * np.cumsum(f) / fs)
    frames = decode_afsk1200(audio, fs)
    assert frames == [_ui()]
    assert parse_ax25(frames[0]).summary().startswith("TEST-1>CQ UI")


def test_non_ax25_payload_inside_valid_hdlc_is_not_reported_as_ax25():
    """Seen in the wild: SE01 sends CRC-valid HDLC frames whose addresses are
    plain ASCII, not shifted. A permissive parser printed garbage callsigns."""
    raw = b"ON01SE\x00ON01SE\x00\x03\x00payload"
    with pytest.raises(ValueError, match="non-conforming"):
        parse_ax25(raw)


# ------------------------------------------------------------ real downlinks
def test_decodes_real_afsk1200_frames_from_swiatowid():
    audio, fs = load_wav(str(DATA / "swiatowid-ax25.wav"))
    frames = [parse_ax25(f) for f in decode_afsk1200(audio, fs)]
    assert len(frames) == 2
    assert all(f.source == "SR6SAT-6" and f.kind == "UI" for f in frames)
    assert frames[0].info.startswith(b"=ER;MN;")


def test_decodes_real_g3ruh_9k6_frame_from_ops_sat():
    audio, fs = load_wav(str(DATA / "ops_sat.wav"))
    frames = [parse_ax25(f) for f in decode_fsk9600_g3ruh(audio, fs)]
    assert [(f.source, f.destination) for f in frames] == [("DP0OPS", "DL0ESA")]


# ----------------------------------------------------- the airborne DSP chain
def _real_detections(name, snr_db, seed=0):
    audio, fs = load_wav(str(DATA / name))
    bb = fm_remodulate(busiest_window(audio, fs, 0.15), fs, deviation_hz=3500.0)
    iq, rate = embed_in_scene(bb, fs, 40, OFFSET, snr_db, np.random.default_rng(seed))
    pipe = EdgePipeline(FeatureClassifier())
    out = pipe.process_frame(iq[:262_144], SweepSegment(CENTER, rate), 0, (0.0, 0.0))
    for det, snip in out:
        det.signal_class, det.class_confidence = pipe.classifier.classify(det, snip)
    return [d for d, _ in out]


@pytest.mark.parametrize("name", ["ops_sat.wav", "swiatowid-ax25.wav"])
@pytest.mark.parametrize("snr_db", [30, 12])
def test_detector_finds_real_satellite_signal(name, snr_db):
    """The part that works: one detection, within 500 Hz, at both SNRs."""
    dets = _real_detections(name, snr_db)
    assert len(dets) == 1
    assert abs(dets[0].freq_hz - (CENTER + OFFSET)) < 500
    assert 3e3 < dets[0].bandwidth_hz < 25e3


@pytest.mark.xfail(strict=True, reason=(
    "Doc 15 §3: FeatureClassifier is closed-set. Every real satellite FSK "
    "signal tried comes out as dmr (or fm_voice) at 0.86-0.87, above the "
    "gate's 0.85 threshold. Needs open-set rejection in the trained "
    "classifier (Step 4); two cheap features were tried and rejected because "
    "they only separate the simulator's DMR, not realistic DMR."))
@pytest.mark.parametrize("name", ["ops_sat.wav", "swiatowid-ax25.wav"])
def test_classifier_does_not_claim_a_protocol_it_cannot_identify(name):
    for snr in (30, 12):
        for d in _real_detections(name, snr):
            assert d.signal_class == "unknown" or d.class_confidence < 0.85, (
                f"{name} at {snr} dB: {d.signal_class} {d.class_confidence:.2f}")


def test_exclusion_range_keeps_the_satellite_band_out_of_a_dmr_warrant(tmp_path):
    """The mitigation available today, in the legal layer rather than the
    classifier: a voice warrant (DMR + analog FM) over 430-440 MHz that
    excludes the amateur-satellite segment refuses the misclassified
    downlink."""
    det = _real_detections("ops_sat.wav", 30)[0]
    assert det.signal_class in ("dmr", "fm_voice")    # the misclassification
    auth_sk, auth_pk = crypto.generate_authority_keypair()
    _, seal_pk = crypto.generate_seal_keypair()
    poly = [[-1e3, -1e3], [1e3, -1e3], [1e3, 1e3], [-1e3, 1e3]]

    def gate(**kw):
        token = WarrantToken.load(signer.quick_warrant(
            seal_pk, auth_sk, area_polygon=poly, freq_ranges_hz=[[430e6, 440e6]],
            signal_classes=["dmr", "fm_voice"], **kw), auth_pk)
        return WarrantGate(token, AuditLog(str(tmp_path / f"a{len(kw)}.jsonl")))

    assert gate().decide(det, (0.0, 0.0))[0] is True          # would be recorded
    allowed, reasons = gate(minimization_rules={
        "exclude_freq_ranges_hz": [[435e6, 438e6]]}).decide(det, (0.0, 0.0))
    assert not allowed and "explicitly_excluded" in reasons
