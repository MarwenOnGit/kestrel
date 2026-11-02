"""Evidence provenance (Doc 14 §T1).

The sealed box makes a recording unreadable; it does not make it attributable.
These tests attack exactly that gap: everything an adversary can do holding
only what is printed in a warrant token, or holding the vault disk.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from kestrel.air.sealer import CollectionController, Sealer
from kestrel.air.warrant_gate import WarrantGate
from kestrel.common import crypto
from kestrel.common.identity import (RECORDING_CONTEXT, CHECKPOINT_CONTEXT,
                                     generate_airframe_identity)
from kestrel.common.messages import Detection, RecordingSidecar
from kestrel.common.warrant import WarrantToken
from kestrel.ground import signer
from kestrel.ground.audit import AuditLog
from kestrel.ground.evidence import EvidenceError, verify_recording, verify_vault

POLY = [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]]
INSIDE = (500.0, 500.0)


@pytest.fixture
def setup(tmp_path):
    auth_sk, auth_pk = crypto.generate_authority_keypair()
    seal_sk, seal_pk = crypto.generate_seal_keypair()
    token = WarrantToken.load(
        signer.quick_warrant(seal_pk, auth_sk, area_polygon=POLY), auth_pk)
    ident = generate_airframe_identity("K1")
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    vault = str(tmp_path / "vault")
    controller = CollectionController(
        WarrantGate(token, audit), Sealer(seal_pk, vault, identity=ident), audit)
    return dict(seal_sk=seal_sk, seal_pk=seal_pk, token=token, ident=ident,
                vault=vault, controller=controller,
                registry={"K1": ident.verify_key_b64})


def _record(controller, freq_hz=463.0e6):
    iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
    d = Detection(aircraft_id="K1", t_gpsdo_ns=1, freq_hz=freq_hz,
                  bandwidth_hz=12_500.0, snr_db=20.0, aircraft_pos_xy=INSIDE,
                  signal_class="dmr", class_confidence=0.95)
    sc = controller.offer(d, iq, INSIDE, duration_s=1.0)
    assert sc is not None
    return sc, iq


def _reason(fn):
    with pytest.raises(EvidenceError) as e:
        fn()
    return e.value.reason


# ------------------------------------------------------------------ honest
def test_genuine_recording_is_admitted_and_unseals(setup):
    sc, iq = _record(setup["controller"])
    plain = verify_recording(sc.to_dict(), setup["vault"], setup["registry"],
                             seal_sk_b64=setup["seal_sk"])
    assert plain == iq.tobytes()


def test_sidecar_on_disk_verifies_too(setup):
    """The signature is over the JSON form, so the copy written next to the
    ciphertext - tuples turned to lists - must verify exactly like the
    in-memory object."""
    sc, _ = _record(setup["controller"])
    on_disk = json.load(open(os.path.join(setup["vault"], sc.recording_ref + ".json")))
    verify_recording(on_disk, setup["vault"], setup["registry"])


# ----------------------------------------------------------------- attacks
def test_forgery_with_only_the_warrant_is_rejected(setup):
    """The attack the sealed box alone could not stop: the seal public key is
    in the token, so anyone can encrypt fabricated IQ to it. Without the
    airframe key they cannot sign it."""
    real, _ = _record(setup["controller"])
    fake_iq = np.ones(1024, dtype=np.complex64).tobytes()
    ct = crypto.seal_to(setup["token"].seal_pubkey, fake_iq)
    open(os.path.join(setup["vault"], "forged.iqz"), "wb").write(ct)

    forged = dict(real.to_dict(), recording_ref="forged.iqz",
                  sha256=crypto.sha256_hex(fake_iq),
                  ciphertext_sha256=crypto.sha256_hex(ct))
    # signature copied from a genuine recording: covers different fields
    assert _reason(lambda: verify_recording(
        forged, setup["vault"], setup["registry"])) == "bad_signature"

    # signed with the attacker's own key: valid signature, unregistered signer
    rogue = generate_airframe_identity("K1")
    forged["signature"] = rogue.sign(RECORDING_CONTEXT,
                                     {k: v for k, v in forged.items() if k != "signature"})
    assert _reason(lambda: verify_recording(
        forged, setup["vault"], setup["registry"])) == "bad_signature"


def test_unregistered_airframe_is_rejected(setup):
    sc, _ = _record(setup["controller"])
    d = dict(sc.to_dict(), airframe_id="K9")
    assert _reason(lambda: verify_recording(
        d, setup["vault"], setup["registry"])) == "unknown_airframe"


def test_editing_metadata_breaks_the_signature(setup):
    """Moving a recording to an authorized frequency after the fact."""
    sc, _ = _record(setup["controller"])
    d = dict(sc.to_dict(), freq_hz=462.5e6)
    assert _reason(lambda: verify_recording(
        d, setup["vault"], setup["registry"])) == "bad_signature"


def test_swapping_ciphertexts_between_recordings_is_detected(setup):
    """Both files are genuine and both sidecars are genuinely signed - only the
    pairing is wrong. The ciphertext hash inside the signature catches it
    without decrypting anything."""
    a, _ = _record(setup["controller"], 463.0e6)
    b, _ = _record(setup["controller"], 463.5e6)
    pa = os.path.join(setup["vault"], a.recording_ref)
    pb = os.path.join(setup["vault"], b.recording_ref)
    tmp = pa + ".tmp"
    shutil.move(pa, tmp); shutil.move(pb, pa); shutil.move(tmp, pb)
    failures = verify_vault([a.to_dict(), b.to_dict()], setup["vault"], setup["registry"])
    assert failures == {a.recording_ref: "ciphertext_mismatch",
                        b.recording_ref: "ciphertext_mismatch"}


def test_stripping_the_signature_is_rejected(setup):
    sc, _ = _record(setup["controller"])
    d = dict(sc.to_dict(), signature="")
    assert _reason(lambda: verify_recording(
        d, setup["vault"], setup["registry"])) == "unsigned"


def test_checkpoint_signature_cannot_be_replayed_as_a_recording(setup):
    """Same key signs recordings and audit checkpoints. Domain separation
    means a signature made for one can never verify as the other."""
    sc, _ = _record(setup["controller"])
    body = {k: v for k, v in sc.to_dict().items() if k != "signature"}
    d = dict(sc.to_dict(), signature=setup["ident"].sign(CHECKPOINT_CONTEXT, body))
    assert _reason(lambda: verify_recording(
        d, setup["vault"], setup["registry"])) == "bad_signature"


def test_sealer_refuses_to_run_without_an_identity(setup):
    with pytest.raises(TypeError):
        Sealer(setup["seal_pk"], setup["vault"])          # noqa: an unsigned vault


def test_every_recording_in_a_sortie_is_admissible():
    from kestrel.sim.run import run_one
    result, _ = run_one("border_crossing", seed=0, verbose=False)
    assert result.recordings, "scenario should record something"
    assert result.evidence_failures == {}
