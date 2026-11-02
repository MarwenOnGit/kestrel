"""Hash-chained, append-only audit log (Doc 08 §5).

Every entry carries the hash of its predecessor, so deleting, reordering, or
editing any entry anywhere in the history is detectable by re-walking the chain.

Runs in two places: onboard each aircraft, and on the ground. The aircraft's
chain head is recorded at landing so the two can be reconciled.

A bare hash chain is tamper-EVIDENT only against someone who edits a line and
stops. Anyone with the disk can rewrite from any entry onward and recompute
every hash after it; `verify()` will pass. Two additions make the chain mean
something (Doc 14 §T3):

  * signed CHECKPOINTs - with an `identity`, the log periodically appends an
    entry carrying the airframe's signature over (seq, head). Rewriting history
    behind a checkpoint now needs the airframe key, not just the disk.
  * a closing checkpoint and a downlink - `close()` seals the end of the chain,
    and `on_checkpoint` hands each checkpoint to the datalink as it is made.
    The ground keeps those, so truncating the log back to an earlier
    checkpoint, or rewriting it with a stolen key, contradicts what the ground
    already holds.

`verify_audit()` at the bottom of this module is the ground-side check.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Iterator

from kestrel.common.crypto import canonical, sha256_hex
from kestrel.common.identity import (CHECKPOINT_CONTEXT, AirframeIdentity,
                                     verify_signed)

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: str, durable: bool = True, *,
                 identity: AirframeIdentity | None = None,
                 checkpoint_every: int = 200,
                 on_checkpoint: Callable[[dict[str, Any]], None] | None = None):
        """durable=True fsyncs every entry so the chain survives power loss
        mid-sortie. Simulation sets it False: thousands of syncs per simulated
        sortie dominate runtime and prove nothing.

        With `identity`, a signed CHECKPOINT is appended after every
        `checkpoint_every` ordinary entries, and each one is passed to
        `on_checkpoint` - the datalink, in flight."""
        self.path = path
        self.durable = durable
        self.identity = identity
        self.checkpoint_every = checkpoint_every
        self.on_checkpoint = on_checkpoint
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.seq, self.head, self._since_checkpoint = self._scan_tail()

    # ------------------------------------------------------------------
    def _scan_tail(self) -> tuple[int, str, int]:
        seq, head, since = -1, GENESIS, 0
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    seq, head = e["seq"], e["hash"]
                    since = 0 if e["type"] == "CHECKPOINT" else since + 1
        return seq, head, since

    # ------------------------------------------------------------------
    def append(self, event_type: str, payload: dict[str, Any],
               t_ns: int | None = None) -> str:
        if event_type == "CHECKPOINT":
            raise ValueError("CHECKPOINT entries are written by checkpoint() only")
        h = self._append(event_type, payload, t_ns)
        self._since_checkpoint += 1
        if (self.identity is not None and self.checkpoint_every
                and self._since_checkpoint >= self.checkpoint_every):
            self.checkpoint()
        return h

    def checkpoint(self) -> dict[str, Any]:
        """Sign the current head. Returns the checkpoint as sent to the ground."""
        if self.identity is None:
            raise RuntimeError("checkpoint() needs an airframe identity")
        cp = {"airframe_id": self.identity.airframe_id,
              "covers_seq": self.seq, "covers_head": self.head}
        signed = dict(cp, signature=self.identity.sign(CHECKPOINT_CONTEXT, cp))
        self._append("CHECKPOINT", signed, None)
        self._since_checkpoint = 0
        if self.on_checkpoint is not None:
            self.on_checkpoint(dict(signed))
        return signed

    def close(self) -> dict[str, Any]:
        """End of sortie: a closing checkpoint, so the end of the chain is
        signed too. A log that does not end in one was cut short."""
        return self.checkpoint()

    def _append(self, event_type: str, payload: dict[str, Any],
                t_ns: int | None) -> str:
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
    def record_decision(self, det, allowed: bool, reasons: list[str],
                        t: float | None = None) -> str:
        """Gate decisions are logged BOTH ways. Proving why something was not
        collected matters as much as proving what was. `t` is the time the gate
        decided at - its clock, not the log's - so a restarted gate can tell
        whether its clock has gone backwards."""
        payload = {
            "detection_id": det.detection_id,
            "aircraft_id": det.aircraft_id,
            "freq_hz": det.freq_hz,
            "signal_class": det.signal_class,
            "class_confidence": det.class_confidence,
            "allowed": allowed,
            "reasons": reasons,
        }
        if t is not None:
            payload["t"] = t
        return self.append("GATE_DECISION", payload)

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


# --------------------------------------------------------------------------
# Ground-side verification
# --------------------------------------------------------------------------
def verify_audit(path: str, airframe_registry: dict[str, str],
                 downlinked: list[dict[str, Any]] | None = None
                 ) -> tuple[bool, str | None, int | None]:
    """Full ground check of an airborne log. Returns (ok, reason, seq).

    1. The hash chain is intact.
    2. Every CHECKPOINT is signed by a registered airframe and covers exactly
       the entry before it.
    3. The log ends in a checkpoint (`unclosed` otherwise: it was cut short,
       or the aircraft never landed cleanly - either way a gap to explain).
    4. Every checkpoint the ground received over the datalink appears in the
       log unchanged. This is the only check that catches truncation back to
       an earlier checkpoint, or a rewrite by someone holding the airframe key:
       both produce a log that is internally perfect.
    """
    log = AuditLog(path)
    ok, bad = log.verify()
    if not ok:
        return False, "chain_broken", bad

    seen: dict[int, str] = {}
    last_type, prev_hash = None, GENESIS
    for e in log.entries():
        if e["type"] == "CHECKPOINT":
            p = e["payload"]
            vk = airframe_registry.get(p.get("airframe_id", ""))
            if vk is None:
                return False, "unknown_airframe", e["seq"]
            body = {k: p[k] for k in ("airframe_id", "covers_seq", "covers_head")}
            if not verify_signed(vk, CHECKPOINT_CONTEXT, body, p.get("signature", "")):
                return False, "bad_checkpoint_signature", e["seq"]
            if p["covers_seq"] != e["seq"] - 1 or p["covers_head"] != prev_hash:
                return False, "checkpoint_mismatch", e["seq"]
            seen[p["covers_seq"]] = p["covers_head"]
        last_type, prev_hash = e["type"], e["hash"]

    if last_type != "CHECKPOINT":
        return False, "unclosed", None

    for cp in downlinked or []:
        if seen.get(cp["covers_seq"]) != cp["covers_head"]:
            return False, "contradicts_downlink", cp["covers_seq"]
    return True, None, None
