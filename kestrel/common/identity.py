"""Airframe identity: who produced a piece of evidence (Doc 14 §T1).

A sealed box hides a recording from everyone but the ground HSM. It says
nothing about who made it: the seal public key is printed inside every warrant
token, so anyone holding a token can produce a perfectly valid ciphertext. For
chain of custody that is backwards. A court needs to know that *this* aircraft
recorded *this* signal, not merely that someone encrypted some bytes to the
right key.

So every airframe carries its own Ed25519 signing key, provisioned on the
ground and registered by public key before the sortie. It signs two things:

  * each recording's sidecar together with the hash of the ciphertext on disk;
  * periodic checkpoints of the audit chain head.

Every signature is domain-separated (`context`), so a signature produced for
one purpose can never be replayed as the other - the same key signs both, and
without a context string a checkpoint body that happened to parse as a sidecar
would verify as one.

Where the key lives matters more than the algorithm. In flight it belongs in a
secure element or TPM with signing exposed and export disabled; here it is a
b64 string because this is the simulator. Doc 14 §T1 states what a stolen key
does and does not let an attacker do.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kestrel.common import crypto

RECORDING_CONTEXT = "kestrel/recording/v1"
CHECKPOINT_CONTEXT = "kestrel/audit-checkpoint/v1"


@dataclass(frozen=True)
class AirframeIdentity:
    airframe_id: str
    signing_key_b64: str
    verify_key_b64: str

    def sign(self, context: str, body: dict[str, Any]) -> str:
        return crypto.sign(self.signing_key_b64, {"context": context, "body": body})

    def __repr__(self) -> str:        # never print the signing key
        return f"AirframeIdentity({self.airframe_id!r}, vk={self.verify_key_b64[:8]}...)"


def generate_airframe_identity(airframe_id: str) -> AirframeIdentity:
    sk, vk = crypto.generate_authority_keypair()     # same primitive: Ed25519
    return AirframeIdentity(airframe_id, sk, vk)


def verify_signed(verify_key_b64: str, context: str, body: dict[str, Any],
                  signature_b64: str) -> bool:
    """True if `signature_b64` is this key's signature over (context, body).

    Returns False rather than raising: callers turn a bad signature into a
    specific evidence finding, and an exception would lose which one.
    """
    try:
        return crypto.verify(verify_key_b64, {"context": context, "body": body},
                             signature_b64)
    except Exception:
        return False
