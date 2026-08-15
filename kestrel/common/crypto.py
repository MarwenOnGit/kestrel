"""Cryptographic primitives for the legal spine (Doc 08).

Two distinct key pairs, deliberately separate:

  * AUTHORITY keypair (Ed25519) - signs warrant tokens. Private half lives with
    the issuing authority / HSM. Aircraft hold only the public half.
  * SEAL keypair (Curve25519) - recordings are encrypted TO the public half.
    The private half NEVER leaves the ground HSM, so airborne code cannot
    decrypt its own take.
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

import nacl.encoding
import nacl.exceptions
import nacl.public
import nacl.signing


# --------------------------------------------------------------------------
# Encoding helpers
# --------------------------------------------------------------------------
def b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def canonical(obj: Any) -> bytes:
    """Deterministic serialization. Signatures are over THIS, never over a
    dict's incidental ordering."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Authority signing keys (Ed25519)
# --------------------------------------------------------------------------
def generate_authority_keypair() -> tuple[str, str]:
    """Returns (signing_key_b64, verify_key_b64)."""
    sk = nacl.signing.SigningKey.generate()
    return b64e(bytes(sk)), b64e(bytes(sk.verify_key))


def sign(signing_key_b64: str, payload: Any) -> str:
    sk = nacl.signing.SigningKey(b64d(signing_key_b64))
    return b64e(sk.sign(canonical(payload)).signature)


def verify(verify_key_b64: str, payload: Any, signature_b64: str) -> bool:
    """Raises nacl.exceptions.BadSignatureError on tamper. Returns True on success."""
    vk = nacl.signing.VerifyKey(b64d(verify_key_b64))
    vk.verify(canonical(payload), b64d(signature_b64))
    return True


# --------------------------------------------------------------------------
# Seal keys (Curve25519 sealed boxes)
# --------------------------------------------------------------------------
def generate_seal_keypair() -> tuple[str, str]:
    """Returns (private_key_b64, public_key_b64).

    The private half belongs in the ground HSM only. If it appears anywhere
    under kestrel/air/, tests/test_structural.py fails the build.
    """
    sk = nacl.public.PrivateKey.generate()
    return b64e(bytes(sk)), b64e(bytes(sk.public_key))


def seal_to(public_key_b64: str, plaintext: bytes) -> bytes:
    """Encrypt so that ONLY the holder of the matching private key can read it.

    Anonymous sealed box: the sender itself cannot decrypt the result. This is
    the property that makes collection write-only on the aircraft.
    """
    box = nacl.public.SealedBox(nacl.public.PublicKey(b64d(public_key_b64)))
    return box.encrypt(plaintext)


def unseal_with(private_key_b64: str, ciphertext: bytes) -> bytes:
    """GROUND ONLY. Never import this into kestrel.air."""
    sk = nacl.public.PrivateKey(b64d(private_key_b64))
    return nacl.public.SealedBox(sk).decrypt(ciphertext)
