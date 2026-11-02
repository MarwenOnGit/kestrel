"""Warrant state across a reboot, and the gate's clock (Doc 14 §T4).

Revocation and spent budget used to live only in memory, and neither is part
of the signed token. Pulling the power reset both. Expiry and the geofence
trust the local clock and GPS, so a clock wound backwards stretched a warrant.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from kestrel.air.sealer import CollectionController, Sealer
from kestrel.air.warrant_gate import CLOCK_ROLLBACK_TOLERANCE_S, WarrantGate
from kestrel.common import crypto
from kestrel.common.identity import generate_airframe_identity
from kestrel.common.messages import Detection
from kestrel.common.warrant import WarrantToken, parse_iso8601
from kestrel.ground import signer
from kestrel.ground.audit import AuditLog

POLY = [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]]
INSIDE = (500.0, 500.0)


@pytest.fixture
def world(tmp_path):
    auth_sk, auth_pk = crypto.generate_authority_keypair()
    _, seal_pk = crypto.generate_seal_keypair()
    raw = signer.quick_warrant(seal_pk, auth_sk, area_polygon=POLY,
                               max_recording_seconds=3.0)
    # Times are taken from the token's own window, never from when the module
    # was imported: in a full run that is minutes before the token exists.
    now = parse_iso8601(raw["body"]["valid_from"]) + 60.0
    return dict(raw=raw, auth_pk=auth_pk, seal_pk=seal_pk, now=now,
                path=str(tmp_path / "audit.jsonl"), vault=str(tmp_path / "vault"),
                ident=generate_airframe_identity("K1"))


def _boot(w, resume=True):
    """Power-on: reload the token from its signed bytes, reopen the log."""
    token = WarrantToken.load(w["raw"], w["auth_pk"])
    audit = AuditLog(w["path"])
    gate = WarrantGate.resume(token, audit) if resume else WarrantGate(token, audit)
    ctl = CollectionController(gate, Sealer(w["seal_pk"], w["vault"], identity=w["ident"]), audit)
    return gate, ctl


def _det():
    return Detection(aircraft_id="K1", t_gpsdo_ns=1, freq_hz=463.0e6,
                     bandwidth_hz=12_500.0, snr_db=20.0, aircraft_pos_xy=INSIDE,
                     signal_class="dmr", class_confidence=0.95)


def _iq():
    return np.zeros(256, dtype=np.complex64)


# --------------------------------------------------------------- revocation
def test_revocation_survives_a_reboot(world):
    gate, _ = _boot(world)
    gate.revoke(now=world["now"])
    gate, _ = _boot(world)
    allowed, reasons = gate.decide(_det(), INSIDE, now=world["now"] + 1)
    assert not allowed and "token_revoked" in reasons


def test_without_resume_a_reboot_forgets_revocation(world):
    """The bug, kept as a test so the reason for resume() stays visible."""
    gate, _ = _boot(world)
    gate.revoke(now=world["now"])
    gate, _ = _boot(world, resume=False)
    assert gate.decide(_det(), INSIDE, now=world["now"] + 1)[0] is True


# ------------------------------------------------------------------ budget
def test_spent_budget_survives_a_reboot(world):
    _, ctl = _boot(world)
    assert ctl.offer(_det(), _iq(), INSIDE, duration_s=2.0, now=world["now"])
    gate, ctl = _boot(world)
    assert gate.token.recorded_seconds == pytest.approx(2.0)
    assert ctl.offer(_det(), _iq(), INSIDE, duration_s=2.0, now=world["now"] + 1)   # 4 s > 3 s cap after this
    gate, ctl = _boot(world)
    allowed, reasons = gate.decide(_det(), INSIDE, now=world["now"] + 2)
    assert not allowed and "budget_exhausted" in reasons


def test_other_warrants_spending_is_not_charged_to_this_one(world, tmp_path):
    _, ctl = _boot(world)
    assert ctl.offer(_det(), _iq(), INSIDE, duration_s=2.0, now=world["now"])
    auth_sk, auth_pk = crypto.generate_authority_keypair()
    other = WarrantToken.load(signer.quick_warrant(world["seal_pk"], auth_sk,
                                                   area_polygon=POLY,
                                                   warrant_id="W-OTHER"), auth_pk)
    gate = WarrantGate.resume(other, AuditLog(world["path"]))
    assert gate.token.recorded_seconds == 0.0


# ------------------------------------------------------------------- clock
def test_clock_wound_back_is_refused(world):
    gate, _ = _boot(world)
    assert gate.decide(_det(), INSIDE, now=world["now"])[0]
    allowed, reasons = gate.decide(_det(), INSIDE, now=world["now"] - 3600)
    assert not allowed and "clock_rollback" in reasons


def test_clock_rollback_is_caught_across_a_reboot(world):
    """The useful version of the attack: reboot with the clock set back, so
    an expired warrant reads as valid again."""
    gate, _ = _boot(world)
    gate.decide(_det(), INSIDE, now=world["now"])
    gate, _ = _boot(world)
    allowed, reasons = gate.decide(_det(), INSIDE, now=world["now"] - 600)
    assert not allowed and "clock_rollback" in reasons


def test_small_backward_step_within_tolerance_is_accepted(world):
    gate, _ = _boot(world)
    gate.decide(_det(), INSIDE, now=world["now"])
    assert gate.decide(_det(), INSIDE, now=world["now"] - CLOCK_ROLLBACK_TOLERANCE_S / 2)[0]


def test_rollback_does_not_move_the_watermark_back(world):
    gate, _ = _boot(world)
    gate.decide(_det(), INSIDE, now=world["now"])
    gate.decide(_det(), INSIDE, now=world["now"] - 600)
    assert gate.latest_t == world["now"]
