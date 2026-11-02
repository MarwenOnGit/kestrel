"""Ground-side evidence verification (Doc 14 §T1).

Runs at the ground station after landing, before anything in the vault is
opened or handed on. A recording is admitted only if every check passes; each
failure has its own reason, because "evidence rejected" is useless to the
person who has to explain why.

Order matters. Provenance is checked BEFORE decryption: an unauthenticated
blob is never fed to the unsealer, so a forged file cannot even reach the
code that would parse it.
"""
from __future__ import annotations

import os
from typing import Any

from kestrel.common import crypto
from kestrel.common.identity import RECORDING_CONTEXT, verify_signed
from kestrel.common.messages import signed_body_of


class EvidenceError(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def verify_recording(sidecar: dict[str, Any], vault_dir: str,
                     airframe_registry: dict[str, str],
                     seal_sk_b64: str | None = None) -> bytes | None:
    """Admit one recording, or raise EvidenceError naming the first failure.

    `airframe_registry` maps airframe_id -> Ed25519 verify key, as provisioned
    before the sortie. It is the trust anchor: a key that was not registered
    on the ground cannot vouch for anything, however valid its signature.

    With `seal_sk_b64` (ground HSM only) the recording is also unsealed and
    its plaintext hash checked; the plaintext IQ is returned.
    """
    airframe = sidecar.get("airframe_id") or ""
    signature = sidecar.get("signature") or ""
    if not signature:
        raise EvidenceError("unsigned", sidecar.get("recording_ref", "?"))
    if airframe not in airframe_registry:
        raise EvidenceError("unknown_airframe", airframe or "<empty>")
    if not verify_signed(airframe_registry[airframe], RECORDING_CONTEXT,
                         signed_body_of(sidecar), signature):
        raise EvidenceError("bad_signature", sidecar.get("recording_ref", "?"))

    path = os.path.join(vault_dir, sidecar["recording_ref"])
    if not os.path.exists(path):
        raise EvidenceError("missing_file", path)
    ciphertext = open(path, "rb").read()
    if crypto.sha256_hex(ciphertext) != sidecar["ciphertext_sha256"]:
        raise EvidenceError("ciphertext_mismatch", sidecar["recording_ref"])

    if seal_sk_b64 is None:
        return None
    plaintext = crypto.unseal_with(seal_sk_b64, ciphertext)
    if crypto.sha256_hex(plaintext) != sidecar["sha256"]:
        raise EvidenceError("plaintext_mismatch", sidecar["recording_ref"])
    return plaintext


def verify_vault(sidecars: list[dict[str, Any]], vault_dir: str,
                 airframe_registry: dict[str, str]) -> dict[str, str]:
    """Check every recording without opening any. Returns {ref: reason} for
    each one that failed; an empty dict means the whole vault is admissible."""
    failures: dict[str, str] = {}
    for sc in sidecars:
        try:
            verify_recording(sc, vault_dir, airframe_registry)
        except EvidenceError as e:
            failures[sc.get("recording_ref", "?")] = e.reason
    return failures
