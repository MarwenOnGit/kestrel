"""Sealed collection - write-only recording (Doc 08 §4).

Recordings are encrypted to the ground station's public key using an anonymous
sealed box. The aircraft therefore CANNOT read back its own recordings: not
after landing, not if the disk is removed, not if the airborne software is
compromised in flight.

This module is intentionally the only writer to the vault. There is no reverse
operation defined anywhere under kestrel/air/ - tests/test_structural.py fails
the build if one appears.
"""
from __future__ import annotations

import os
import time

import numpy as np

from kestrel.common.crypto import seal_to, sha256_hex
from kestrel.common.identity import RECORDING_CONTEXT, AirframeIdentity
from kestrel.common.messages import Detection, RecordingSidecar


class VaultFullError(Exception):
    pass


class Sealer:
    """Encrypt-to-ground, then sign-as-airframe.

    Confidentiality and provenance are separate jobs. The sealed box gives the
    first: only the ground HSM can read the take. It cannot give the second,
    because the seal public key is in every warrant and an anonymous box has
    no sender. The airframe signature over (sidecar, ciphertext hash) is what
    lets the ground prove which aircraft made a recording and that neither the
    file nor its metadata changed afterwards. `identity` is therefore required:
    an unsigned recording is not evidence.
    """

    def __init__(self, seal_pubkey_b64: str, vault_dir: str, *,
                 identity: AirframeIdentity,
                 capacity_bytes: int | None = None, durable: bool = True):
        self.seal_pubkey = seal_pubkey_b64
        self.identity = identity
        self.vault_dir = vault_dir
        self.capacity_bytes = capacity_bytes
        self.durable = durable
        self.closed = False
        self._used_bytes = 0
        os.makedirs(vault_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def used_bytes(self) -> int:
        return self._used_bytes

    def close(self) -> None:
        """Called on warrant expiry or mission abort: no further writes."""
        self.closed = True

    # ------------------------------------------------------------------
    def seal_recording(self, iq: np.ndarray, det: Detection, warrant_id: str,
                       start_ns: int, stop_ns: int,
                       gps_track: list | None = None) -> RecordingSidecar:
        if self.closed:
            raise VaultFullError("vault closed (warrant expired or mission aborted)")

        raw = np.ascontiguousarray(iq, dtype=np.complex64).tobytes()
        digest = sha256_hex(raw)

        if (self.capacity_bytes is not None
                and self.used_bytes() + len(raw) > self.capacity_bytes):
            raise VaultFullError("vault capacity exceeded")

        ciphertext = seal_to(self.seal_pubkey, raw)
        ct_digest = sha256_hex(ciphertext)

        ref = f"{det.detection_id}.iqz"
        path = os.path.join(self.vault_dir, ref)
        with open(path, "wb") as f:
            f.write(ciphertext)
            f.flush()
            if self.durable:
                os.fsync(f.fileno())
        self._used_bytes += len(ciphertext)

        sidecar = RecordingSidecar(
            recording_ref=ref,
            warrant_id=warrant_id,
            detection_ref=det.detection_id,
            start_ns=start_ns,
            stop_ns=stop_ns,
            sha256=digest,
            sealed_to_pubkey=self.seal_pubkey,
            freq_hz=det.freq_hz,
            signal_class=det.signal_class,
            gps_track=gps_track or [],
            airframe_id=self.identity.airframe_id,
            ciphertext_sha256=ct_digest,
        )
        sidecar.signature = self.identity.sign(RECORDING_CONTEXT,
                                               sidecar.signed_body())
        with open(path + ".json", "w") as f:
            f.write(sidecar.to_json())
        return sidecar


class CollectionController:
    """Ties the gate to the sealer so there is exactly ONE path from IQ to disk.

    Nothing else in kestrel/air/ opens a file in the vault. This is enforced
    structurally by tests/test_structural.py.
    """

    def __init__(self, gate, sealer: Sealer, audit):
        self.gate, self.sealer, self.audit = gate, sealer, audit
        self.recordings: list[RecordingSidecar] = []

    def offer(self, det: Detection, iq: np.ndarray,
              aircraft_xy: tuple[float, float],
              duration_s: float = 0.0,
              now: float | None = None) -> RecordingSidecar | None:
        """Offer a detection for capture. Returns a sidecar only if authorized.

        The IQ is discarded, unrecorded, on any denial.
        """
        allowed, reasons = self.gate.decide(det, aircraft_xy, now)
        det.warrant_match = allowed
        det.warrant_reasons = reasons

        if not allowed:
            return None

        det.warrant_id = self.gate.token.warrant_id
        t0 = time.time_ns() if now is None else int(now * 1e9)
        try:
            sidecar = self.sealer.seal_recording(
                iq, det, self.gate.token.warrant_id,
                start_ns=t0, stop_ns=t0 + int(duration_s * 1e9),
                gps_track=[[aircraft_xy[0], aircraft_xy[1], det.aircraft_alt_m, t0]],
            )
        except VaultFullError as e:
            # Never a silent loss: the gap itself is evidence.
            self.audit.append("STORAGE_UNAVAILABLE", {
                "detection_id": det.detection_id, "error": str(e)})
            det.warrant_match = False
            det.warrant_reasons = ["storage_unavailable"]
            return None

        self.gate.charge(det, duration_s)
        det.recording_ref = sidecar.recording_ref
        self.audit.record_sealed(sidecar)
        self.recordings.append(sidecar)
        return sidecar
