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


class WarrantGate:
    def __init__(self, token: WarrantToken, audit, min_confidence: float = MIN_CLASS_CONFIDENCE):
        self.token = token
        self.audit = audit
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    def decide(self, det: Detection, aircraft_xy: tuple[float, float],
               now: float | None = None) -> tuple[bool, list[str]]:
        now = time.time() if now is None else now
        reasons: list[str] = []

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
        self.audit.record_decision(det, allowed, reasons)
        return allowed, reasons

    # ------------------------------------------------------------------
    @staticmethod
    def _emitter_key(det: Detection) -> str:
        """Per-emitter budgets are keyed by frequency bucket until
        fingerprinting exists (Phase 3). 25 kHz buckets match narrowband
        channel spacing."""
        return f"f{int(det.freq_hz // 25_000)}"

    def charge(self, det: Detection, seconds: float) -> None:
        self.token.charge(self._emitter_key(det), seconds)

    def expired(self, now: float | None = None) -> bool:
        return self.token.expired(now)
