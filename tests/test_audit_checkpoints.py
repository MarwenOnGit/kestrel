"""Signed audit checkpoints (Doc 14 §T3).

A hash chain alone only catches an attacker who edits one line and stops. These
tests play the attacker who rewrites the tail and recomputes every hash - the
one a bare chain cannot see - and record honestly what each defence buys.
"""
from __future__ import annotations

import json

import pytest

from kestrel.common.crypto import canonical, sha256_hex
from kestrel.common.identity import generate_airframe_identity
from kestrel.ground.audit import AuditLog, verify_audit


@pytest.fixture
def flight(tmp_path):
    ident = generate_airframe_identity("K1")
    downlink: list[dict] = []
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path, identity=ident, checkpoint_every=10,
                   on_checkpoint=downlink.append)
    for i in range(35):
        log.append("DETECTION", {"i": i, "freq_hz": 463e6 + i})
    log.close()
    return dict(path=path, ident=ident, downlink=downlink,
                registry={"K1": ident.verify_key_b64})


def _lines(path):
    return [json.loads(l) for l in open(path).read().splitlines() if l.strip()]


def _write(path, entries):
    open(path, "w").write("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))


def _rechain(entries, start):
    """What an attacker with disk access does: rewrite from `start` and
    recompute every hash after it so the bare chain verifies again."""
    prev = entries[start - 1]["hash"] if start else "0" * 64
    for i, e in enumerate(entries[start:], start):
        e["seq"], e["prev"] = i, prev
        body = {k: v for k, v in e.items() if k != "hash"}
        e["hash"] = sha256_hex(canonical(body))
        prev = e["hash"]
    return entries


# ------------------------------------------------------------------ honest
def test_honest_flight_verifies_against_its_downlink(flight):
    assert verify_audit(flight["path"], flight["registry"], flight["downlink"]) == (True, None, None)
    assert len(flight["downlink"]) == 4          # every 10 entries, plus close


def test_checkpoint_is_written_by_checkpoint_only(flight):
    with pytest.raises(ValueError):
        AuditLog(flight["path"]).append("CHECKPOINT", {"forged": True})


def test_reopened_log_keeps_counting_toward_the_next_checkpoint(tmp_path):
    ident = generate_airframe_identity("K1")
    path = str(tmp_path / "a.jsonl")
    log = AuditLog(path, identity=ident, checkpoint_every=10)
    for i in range(7):
        log.append("X", {"i": i})
    log = AuditLog(path, identity=ident, checkpoint_every=10)      # reboot
    for i in range(3):
        log.append("X", {"i": i})
    assert _lines(path)[-1]["type"] == "CHECKPOINT"


# ----------------------------------------------------------------- attacks
def test_rechained_rewrite_fools_bare_verify_but_not_checkpoints(flight):
    """The motivating attack. Edit entry 5, recompute every hash after it:
    the old verify() passes, the signed checkpoints do not."""
    entries = _lines(flight["path"])
    entries[5]["payload"]["freq_hz"] = 999e6
    _write(flight["path"], _rechain(entries, 5))

    assert AuditLog(flight["path"]).verify() == (True, None)        # fooled
    ok, reason, _ = verify_audit(flight["path"], flight["registry"])
    assert not ok and reason == "checkpoint_mismatch"


def test_deleting_checkpoints_to_hide_a_rewrite_leaves_it_unclosed(flight):
    entries = [e for e in _lines(flight["path"]) if e["type"] != "CHECKPOINT"]
    entries[20]["payload"]["freq_hz"] = 999e6
    _write(flight["path"], _rechain(entries, 0))
    ok, reason, _ = verify_audit(flight["path"], flight["registry"])
    assert not ok and reason == "unclosed"


def test_forged_checkpoint_without_the_key_is_rejected(flight):
    entries = [e for e in _lines(flight["path"]) if e["type"] != "CHECKPOINT"]
    rogue = generate_airframe_identity("K1")
    rechained = _rechain(entries, 0)
    cp = {"airframe_id": "K1", "covers_seq": rechained[-1]["seq"],
          "covers_head": rechained[-1]["hash"]}
    from kestrel.common.identity import CHECKPOINT_CONTEXT
    rechained.append({"type": "CHECKPOINT", "t_ns": 0,
                      "payload": dict(cp, signature=rogue.sign(CHECKPOINT_CONTEXT, cp))})
    _write(flight["path"], _rechain(rechained, len(rechained) - 1))
    ok, reason, _ = verify_audit(flight["path"], flight["registry"])
    assert not ok and reason == "bad_checkpoint_signature"


# --------------------------------------------------------- known limits
def test_truncation_to_an_earlier_checkpoint_needs_the_downlink(flight):
    """Cutting the log right after a genuine checkpoint leaves a log that is
    internally perfect. Offline verification CANNOT see it - stated here so
    nobody believes otherwise. The ground's downlinked copy does."""
    entries = _lines(flight["path"])
    first_cp = next(i for i, e in enumerate(entries) if e["type"] == "CHECKPOINT")
    _write(flight["path"], entries[:first_cp + 1])

    assert verify_audit(flight["path"], flight["registry"])[0] is True       # blind
    ok, reason, _ = verify_audit(flight["path"], flight["registry"], flight["downlink"])
    assert not ok and reason == "contradicts_downlink"


def test_a_stolen_key_can_rewrite_after_the_last_downlinked_checkpoint(flight):
    """The residual risk, recorded rather than hidden. An attacker holding the
    airframe key can rewrite anything the ground has not yet received a
    checkpoint for and re-sign it. Defences: key in a secure element with
    export disabled, and a short checkpoint interval - this window is exactly
    one interval long (Doc 14 §T3)."""
    ident, dl = flight["ident"], flight["downlink"]
    entries = [e for e in _lines(flight["path"]) if e["seq"] <= dl[-2]["covers_seq"] + 1]
    # keep everything up to and including the second-to-last downlinked cp
    _write(flight["path"], entries)
    log = AuditLog(flight["path"], identity=ident, checkpoint_every=10)
    log.append("DETECTION", {"i": "fabricated"})
    log.close()
    ok, _, _ = verify_audit(flight["path"], flight["registry"], dl[:-1])
    assert ok          # undetectable if the final checkpoint never reached the ground
    ok, reason, _ = verify_audit(flight["path"], flight["registry"], dl)
    assert not ok and reason == "contradicts_downlink"   # caught if it did
