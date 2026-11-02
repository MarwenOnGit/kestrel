"""The warrant gate (Doc 08 §3).

The ONLY place in the codebase that may authorize content capture.

Deliberately boring: deterministic, no AI, no randomness, no network calls, and
short enough that a non-engineer can read every branch. If a reviewer cannot
follow this file, the system is not defensible.

Default is DENY. Every reason for denial is recorded.
"""
from __future__ import annotations

import time

from kestrel.common.messages import Detection
from kestrel.common.warrant import WarrantToken

# A classification this uncertain is treated as unknown, and unknown is never
# authorized. AI uncertainty must never widen collection.
MIN_CLASS_CONFIDENCE = 0.85

# How far the clock may step backwards between two decisions before the gate
# stops trusting it. GPS time corrections and NTP slews are sub-second; a
# larger backward jump is a reset, a fault, or someone winding the clock back
# to stretch an expiring warrant (Doc 14 §T4).
CLOCK_ROLLBACK_TOLERANCE_S = 1.0


class WarrantGate:
    """State that limits collection - revocation, budgets spent, the latest
    time seen - must survive a reboot. A gate that forgets them is a gate that
    can be reset by pulling the power (Doc 14 §T4). Nothing here keeps a second
    copy of that state: it is all rebuilt from the audit log, which is already
    durable, already hash-chained, and already what the ground will check.
    Use `WarrantGate.resume()` whenever a log already exists.
    """

    def __init__(self, token: WarrantToken, audit, min_confidence: float = MIN_CLASS_CONFIDENCE):
        self.token = token
        self.audit = audit
        self.min_confidence = min_confidence
        self.latest_t: float | None = None

    @classmethod
    def resume(cls, token: WarrantToken, audit, **kw) -> "WarrantGate":
        """A gate whose limits reflect everything already in `audit`:
        recording time charged, revocation, and the latest decision time."""
        gate = cls(token, audit, **kw)
        for e in audit.entries():
            p, kind = e["payload"], e["type"]
            if kind == "RECORDING_SEALED" and p.get("warrant_id") == token.warrant_id:
                seconds = (p["stop_ns"] - p["start_ns"]) / 1e9
                token.charge(gate.emitter_key_for(p["freq_hz"]), seconds)
            elif kind == "REVOCATION" and p.get("warrant_id") == token.warrant_id:
                token.revoke()
            elif kind == "GATE_DECISION" and "t" in p:
                gate.latest_t = (p["t"] if gate.latest_t is None
                                 else max(gate.latest_t, p["t"]))
        return gate

    def revoke(self, now: float | None = None) -> None:
        """Revoke and record it, so the revocation outlives this process."""
        now = time.time() if now is None else now
        self.token.revoke()
        self.audit.append("REVOCATION", {"warrant_id": self.token.warrant_id, "t": now})

    # ------------------------------------------------------------------
    def decide(self, det: Detection, aircraft_xy: tuple[float, float],
               now: float | None = None) -> tuple[bool, list[str]]:
        now = time.time() if now is None else now
        reasons: list[str] = []

        if self.latest_t is not None and now < self.latest_t - CLOCK_ROLLBACK_TOLERANCE_S:
            reasons.append("clock_rollback")
        else:
            self.latest_t = now if self.latest_t is None else max(self.latest_t, now)
        if self.token.revoked:
            reasons.append("token_revoked")
        if not self.token.valid_now(now):
            reasons.append("token_expired")
        if not self.token.contains_position(aircraft_xy):
            reasons.append("outside_geofence")
        if det.signal_class == "unknown":
            reasons.append("class_unknown")
        if det.class_confidence < self.min_confidence:
            reasons.append("low_confidence")
        if not self.token.covers_class(det.signal_class):
            reasons.append("class_not_authorized")
        if not self.token.covers_freq(det.freq_hz):
            reasons.append("freq_not_authorized")
        if self.token.excluded_freq(det.freq_hz):
            reasons.append("explicitly_excluded")
        if self.token.budget_exhausted():
            reasons.append("budget_exhausted")
        if self.token.emitter_budget_exhausted(self._emitter_key(det)):
            reasons.append("emitter_budget_exhausted")

        allowed = not reasons
        self.audit.record_decision(det, allowed, reasons, t=now)
        return allowed, reasons

    # ------------------------------------------------------------------
    @staticmethod
    def emitter_key_for(freq_hz: float) -> str:
        """Per-emitter budgets are keyed by frequency bucket until
        fingerprinting exists (Phase 3). 25 kHz buckets match narrowband
        channel spacing."""
        return f"f{int(freq_hz // 25_000)}"

    @classmethod
    def _emitter_key(cls, det: Detection) -> str:
        return cls.emitter_key_for(det.freq_hz)

    def charge(self, det: Detection, seconds: float) -> None:
        self.token.charge(self._emitter_key(det), seconds)

    def expired(self, now: float | None = None) -> bool:
        return self.token.expired(now)
