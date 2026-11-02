"""The legal spine (Doc 08 §7). These are the tests that matter most."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import nacl.exceptions
import numpy as np
import pytest

from kestrel.air.sealer import CollectionController, Sealer
from kestrel.air.warrant_gate import WarrantGate
from kestrel.common import crypto
from kestrel.common.identity import generate_airframe_identity
from kestrel.common.messages import Detection
from kestrel.common.warrant import TokenError, WarrantToken
from kestrel.ground import signer
from kestrel.ground.audit import AuditLog

POLY = [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]]
INSIDE = (500.0, 500.0)
OUTSIDE = (5000.0, 5000.0)


@pytest.fixture
def keys():
    auth_sk, auth_pk = crypto.generate_authority_keypair()
    seal_sk, seal_pk = crypto.generate_seal_keypair()
    return dict(auth_sk=auth_sk, auth_pk=auth_pk, seal_sk=seal_sk, seal_pk=seal_pk)


@pytest.fixture
def ident():
    return generate_airframe_identity("K1")


@pytest.fixture
def audit(tmp_path):
    return AuditLog(str(tmp_path / "audit.jsonl"))


def make_token(keys, **overrides):
    raw = signer.quick_warrant(keys["seal_pk"], keys["auth_sk"],
                               area_polygon=POLY, **overrides)
    return WarrantToken.load(raw, keys["auth_pk"])


def det(freq_hz=463.0e6, signal_class="dmr", conf=0.95, **kw):
    return Detection(aircraft_id="K1", t_gpsdo_ns=1, freq_hz=freq_hz,
                     bandwidth_hz=12_500.0, snr_db=20.0,
                     aircraft_pos_xy=INSIDE, signal_class=signal_class,
                     class_confidence=conf, **kw)


# ---------------------------------------------------------------- token
def test_valid_token_loads(keys):
    t = make_token(keys)
    assert t.valid_now()
    assert t.covers(463.0e6, "dmr")


def test_tampered_token_rejected(keys):
    raw = signer.quick_warrant(keys["seal_pk"], keys["auth_sk"], area_polygon=POLY)
    raw["body"]["freq_ranges_hz"] = [[0.0, 8e9]]        # grab the whole spectrum
    with pytest.raises(nacl.exceptions.BadSignatureError):
        WarrantToken.load(raw, keys["auth_pk"])


def test_two_person_rule_enforced_at_issue(keys):
    with pytest.raises(TokenError):
        signer.quick_warrant(keys["seal_pk"], keys["auth_sk"],
                             operator_id="same", co_signer_id="same")


def test_wrong_authority_key_rejected(keys):
    other_sk, _ = crypto.generate_authority_keypair()
    raw = signer.quick_warrant(keys["seal_pk"], other_sk, area_polygon=POLY)
    with pytest.raises(nacl.exceptions.BadSignatureError):
        WarrantToken.load(raw, keys["auth_pk"])


# ----------------------------------------------------------------- gate
def test_authorized_detection_allowed(keys, audit):
    g = WarrantGate(make_token(keys), audit)
    allowed, reasons = g.decide(det(), INSIDE)
    assert allowed, reasons


def test_expired_token_denies(keys, audit):
    now = datetime.now(timezone.utc)
    t = make_token(keys, valid_from=now - timedelta(hours=4),
                   valid_until=now - timedelta(hours=1))
    allowed, reasons = WarrantGate(t, audit).decide(det(), INSIDE)
    assert not allowed and "token_expired" in reasons


def test_revoked_token_denies(keys, audit):
    t = make_token(keys)
    t.revoke()
    allowed, reasons = WarrantGate(t, audit).decide(det(), INSIDE)
    assert not allowed and "token_revoked" in reasons


def test_unknown_class_never_recorded(keys, audit):
    g = WarrantGate(make_token(keys), audit)
    allowed, reasons = g.decide(det(signal_class="unknown", conf=0.99), INSIDE)
    assert not allowed and "class_unknown" in reasons


def test_low_confidence_denies(keys, audit):
    g = WarrantGate(make_token(keys), audit)
    allowed, reasons = g.decide(det(conf=0.60), INSIDE)
    assert not allowed and "low_confidence" in reasons


def test_outside_geofence_denies(keys, audit):
    g = WarrantGate(make_token(keys), audit)
    allowed, reasons = g.decide(det(), OUTSIDE)
    assert not allowed and "outside_geofence" in reasons


def test_unauthorized_frequency_denies(keys, audit):
    g = WarrantGate(make_token(keys), audit)
    allowed, reasons = g.decide(det(freq_hz=1842e6, signal_class="lte"), INSIDE)
    assert not allowed and "freq_not_authorized" in reasons


def test_unauthorized_class_denies(keys, audit):
    g = WarrantGate(make_token(keys), audit)
    allowed, reasons = g.decide(det(freq_hz=463.0e6, signal_class="tetra"), INSIDE)
    assert not allowed and "class_not_authorized" in reasons


def test_explicitly_excluded_frequency_denies(keys, audit):
    t = make_token(keys, minimization_rules={
        "exclude_freq_ranges_hz": [[462.9e6, 463.1e6]]})
    allowed, reasons = WarrantGate(t, audit).decide(det(freq_hz=463.0e6), INSIDE)
    assert not allowed and "explicitly_excluded" in reasons


def test_budget_exhaustion_denies(keys, audit):
    t = make_token(keys, max_recording_seconds=10.0)
    g = WarrantGate(t, audit)
    t.recorded_seconds = 10.0
    allowed, reasons = g.decide(det(), INSIDE)
    assert not allowed and "budget_exhausted" in reasons


def test_every_denial_is_logged(keys, audit):
    g = WarrantGate(make_token(keys), audit)
    g.decide(det(freq_hz=1842e6, signal_class="lte"), INSIDE)
    entries = [e for e in audit.entries() if e["type"] == "GATE_DECISION"]
    assert len(entries) == 1
    assert entries[0]["payload"]["allowed"] is False
    assert entries[0]["payload"]["reasons"]


# --------------------------------------------------------------- sealing
def test_sealed_recording_unreadable_without_private_key(keys, ident, audit, tmp_path):
    t = make_token(keys)
    controller = CollectionController(
        WarrantGate(t, audit),
        Sealer(keys["seal_pk"], str(tmp_path / "vault"), identity=ident), audit)
    iq = (np.random.randn(512) + 1j * np.random.randn(512)).astype(np.complex64)
    d = det()
    sidecar = controller.offer(d, iq, INSIDE, duration_s=1.0)
    assert sidecar is not None

    blob = open(os.path.join(str(tmp_path / "vault"), sidecar.recording_ref), "rb").read()
    assert iq.tobytes() not in blob                       # not stored in the clear

    wrong_sk, _ = crypto.generate_seal_keypair()
    with pytest.raises(Exception):
        crypto.unseal_with(wrong_sk, blob)

    recovered = crypto.unseal_with(keys["seal_sk"], blob)  # ground HSM only
    assert recovered == iq.tobytes()
    assert crypto.sha256_hex(recovered) == sidecar.sha256


def test_denied_detection_writes_nothing(keys, ident, audit, tmp_path):
    vault = str(tmp_path / "vault")
    controller = CollectionController(
        WarrantGate(make_token(keys), audit),
        Sealer(keys["seal_pk"], vault, identity=ident), audit)
    iq = np.zeros(256, dtype=np.complex64)
    assert controller.offer(det(freq_hz=1842e6, signal_class="lte"), iq, INSIDE) is None
    assert os.listdir(vault) == []


def test_vault_full_logs_gap_and_denies(keys, ident, audit, tmp_path):
    controller = CollectionController(
        WarrantGate(make_token(keys), audit),
        Sealer(keys["seal_pk"], str(tmp_path / "vault"), identity=ident,
               capacity_bytes=10), audit)
    iq = np.zeros(4096, dtype=np.complex64)
    assert controller.offer(det(), iq, INSIDE) is None
    assert any(e["type"] == "STORAGE_UNAVAILABLE" for e in audit.entries())


# ----------------------------------------------------------------- audit
def test_audit_chain_verifies(audit):
    for i in range(10):
        audit.append("DETECTION", {"i": i})
    ok, bad = audit.verify()
    assert ok and bad is None


def test_audit_chain_detects_tampering(tmp_path):
    path = str(tmp_path / "a.jsonl")
    log = AuditLog(path)
    for i in range(10):
        log.append("DETECTION", {"i": i})

    lines = open(path).read().splitlines()
    lines[5] = lines[5].replace('"i": 5', '"i": 999')
    open(path, "w").write("\n".join(lines) + "\n")

    ok, bad = AuditLog(path).verify()
    assert not ok and bad == 5


def test_audit_chain_detects_deletion(tmp_path):
    path = str(tmp_path / "a.jsonl")
    log = AuditLog(path)
    for i in range(10):
        log.append("DETECTION", {"i": i})
    lines = open(path).read().splitlines()
    del lines[4]
    open(path, "w").write("\n".join(lines) + "\n")
    ok, _ = AuditLog(path).verify()
    assert not ok
