"""The warrant token: legal authorization as a cryptographic capability (Doc 08 §2).

The aircraft's ability to capture content is *defined* by this object. Four
properties are enforced here, not in policy:

  1. Signed      - tamper is detected on every load and every decision.
  2. Time-boxed  - authority expires; there is no indefinite collection.
  3. Geofenced   - outside the polygon, no content capture at all.
  4. Two-person  - operator and co-signer must be distinct.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from kestrel.common import crypto


def parse_iso8601(s: str) -> float:
    """ISO-8601 -> POSIX seconds. Accepts trailing 'Z'."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def point_in_polygon(pt: tuple[float, float],
                     poly: list[list[float]]) -> bool:
    """Ray-casting. Polygon is a list of [x, y] vertices (local ENU metres).

    Implemented directly rather than pulling in shapely: this function is part
    of the legal spine and must be readable and auditable by a non-engineer.
    """
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # does the edge straddle the horizontal ray at y?
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_cross > x:
                inside = not inside
    return inside


class TokenError(Exception):
    """Raised for any structural or two-person-rule violation."""


@dataclass
class WarrantToken:
    """A verified warrant. Constructing one PROVES the signature was valid."""

    body: dict[str, Any]
    signature: str

    recorded_seconds: float = 0.0
    per_emitter_seconds: dict[str, float] = field(default_factory=dict)
    revoked: bool = False

    # ---------------------------------------------------------------- load
    @classmethod
    def load(cls, raw: dict[str, Any], authority_pubkey_b64: str) -> "WarrantToken":
        """Verify and construct. Raises on ANY problem - fail closed."""
        if "body" not in raw or "signature" not in raw:
            raise TokenError("malformed token: missing body or signature")

        body, sig = raw["body"], raw["signature"]

        # Signature first: everything below trusts these fields.
        crypto.verify(authority_pubkey_b64, body, sig)   # raises BadSignatureError

        for k in ("warrant_id", "operator_id", "co_signer_id", "freq_ranges_hz",
                  "signal_classes", "area_polygon", "valid_from", "valid_until",
                  "seal_pubkey"):
            if k not in body:
                raise TokenError(f"token missing required field: {k}")

        if body["operator_id"] == body["co_signer_id"]:
            raise TokenError("two-person rule violated: operator == co_signer")

        if parse_iso8601(body["valid_until"]) <= parse_iso8601(body["valid_from"]):
            raise TokenError("token validity window is empty or inverted")

        if len(body["area_polygon"]) < 3:
            raise TokenError("area_polygon must have at least 3 vertices")

        return cls(body=body, signature=sig)

    # ------------------------------------------------------------ queries
    @property
    def warrant_id(self) -> str:
        return self.body["warrant_id"]

    @property
    def seal_pubkey(self) -> str:
        return self.body["seal_pubkey"]

    def valid_now(self, now: float | None = None) -> bool:
        if self.revoked:
            return False
        now = time.time() if now is None else now
        return (parse_iso8601(self.body["valid_from"]) <= now
                <= parse_iso8601(self.body["valid_until"]))

    def expired(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.revoked or now > parse_iso8601(self.body["valid_until"])

    def covers_freq(self, freq_hz: float) -> bool:
        return any(lo <= freq_hz <= hi for lo, hi in self.body["freq_ranges_hz"])

    def excluded_freq(self, freq_hz: float) -> bool:
        rules = self.body.get("minimization_rules", {})
        return any(lo <= freq_hz <= hi
                   for lo, hi in rules.get("exclude_freq_ranges_hz", []))

    def covers_class(self, signal_class: str) -> bool:
        return signal_class in self.body["signal_classes"]

    def contains_position(self, xy: tuple[float, float]) -> bool:
        return point_in_polygon(xy, self.body["area_polygon"])

    def covers(self, freq_hz: float, signal_class: str) -> bool:
        """Convenience used by post-sortie audit review."""
        return (self.covers_freq(freq_hz)
                and self.covers_class(signal_class)
                and not self.excluded_freq(freq_hz))

    # ------------------------------------------------------------ budgets
    def budget_exhausted(self) -> bool:
        cap = self.body.get("max_recording_seconds")
        return cap is not None and self.recorded_seconds >= cap

    def emitter_budget_exhausted(self, emitter_key: str) -> bool:
        cap = self.body.get("max_recording_per_emitter_seconds")
        if cap is None:
            return False
        return self.per_emitter_seconds.get(emitter_key, 0.0) >= cap

    def charge(self, emitter_key: str, seconds: float) -> None:
        self.recorded_seconds += seconds
        self.per_emitter_seconds[emitter_key] = (
            self.per_emitter_seconds.get(emitter_key, 0.0) + seconds)

    def revoke(self) -> None:
        """Emergency revocation. Fail-safe: once set, never cleared in flight."""
        self.revoked = True
