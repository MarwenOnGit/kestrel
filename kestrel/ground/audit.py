"""Hash-chained, append-only audit log (Doc 08 §5).

Every entry carries the hash of its predecessor, so deleting, reordering, or
editing any entry anywhere in the history is detectable by re-walking the chain.

Runs in two places: onboard each aircraft, and on the ground. The aircraft's
chain head is recorded at landing so the two can be reconciled.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterator

from kestrel.common.crypto import canonical, sha256_hex

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: str, durable: bool = True):
        """durable=True fsyncs every entry so the chain survives power loss
        mid-sortie. Simulation sets it False: thousands of syncs per simulated
        sortie dominate runtime and prove nothing."""
        self.path = path
        self.durable = durable
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.seq, self.head = self._scan_tail()

    # ------------------------------------------------------------------
    def _scan_tail(self) -> tuple[int, str]:
        seq, head = -1, GENESIS
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    seq, head = e["seq"], e["hash"]
        return seq, head

    # ------------------------------------------------------------------
    def append(self, event_type: str, payload: dict[str, Any],
               t_ns: int | None = None) -> str:
        entry = {
            "seq": self.seq + 1,
            "t_ns": time.time_ns() if t_ns is None else t_ns,
            "type": event_type,
            "payload": payload,
            "prev": self.head,
        }
        entry["hash"] = sha256_hex(canonical(entry))
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            if self.durable:
                os.fsync(f.fileno())      # survive power loss mid-sortie
        self.seq, self.head = entry["seq"], entry["hash"]
        return entry["hash"]

    # ------------------------------------------------------------------
    def entries(self) -> Iterator[dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def verify(self) -> tuple[bool, int | None]:
        """Re-walk the chain. Returns (ok, first_bad_seq)."""
        prev, expect_seq = GENESIS, 0
        for e in self.entries():
            if e.get("seq") != expect_seq:
                return False, e.get("seq")
            if e.get("prev") != prev:
                return False, e["seq"]
            body = {k: v for k, v in e.items() if k != "hash"}
            if sha256_hex(canonical(body)) != e["hash"]:
                return False, e["seq"]
            prev = e["hash"]
            expect_seq += 1
        return True, None

    # ------------------------------------------------------------------
    # Convenience recorders used across the codebase
    # ------------------------------------------------------------------
    def record_decision(self, det, allowed: bool, reasons: list[str]) -> str:
        """Gate decisions are logged BOTH ways. Proving why something was not
        collected matters as much as proving what was."""
        return self.append("GATE_DECISION", {
            "detection_id": det.detection_id,
            "aircraft_id": det.aircraft_id,
            "freq_hz": det.freq_hz,
            "signal_class": det.signal_class,
            "class_confidence": det.class_confidence,
            "allowed": allowed,
            "reasons": reasons,
        })

    def record_detection(self, det) -> str:
        # sanitized() strips every simulator-truth field. Never persist truth:
        # it must be impossible for ground truth to reach an operator or a
        # report and be mistaken for a measurement.
        d = det.sanitized().to_dict()
        for k in ("truth_emitter_id", "truth_class", "truth_xy"):
            d.pop(k, None)
        return self.append("DETECTION", d)

    def record_sealed(self, sidecar) -> str:
        return self.append("RECORDING_SEALED", sidecar.to_dict())

    def find_detections(self) -> list[dict[str, Any]]:
        return [e["payload"] for e in self.entries() if e["type"] == "DETECTION"]

    def find_recordings(self) -> list[dict[str, Any]]:
        return [e["payload"] for e in self.entries()
                if e["type"] == "RECORDING_SEALED"]
