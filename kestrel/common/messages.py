"""Interface contracts between subsystems (Doc 01 §9).

These dataclasses are the ONLY types that cross subsystem boundaries.
Keeping them stable is what lets A/B/C/D/E/F be built and tested alone.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


def new_id() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------------------
# Detection: emitted by edge processing (B), consumed by datalink (D),
# ground fusion (E), and the local autonomy agent (F).
# --------------------------------------------------------------------------
@dataclass
class Detection:
    aircraft_id: str
    t_gpsdo_ns: int
    freq_hz: float
    bandwidth_hz: float
    snr_db: float
    aircraft_pos_xy: tuple[float, float]

    detection_id: str = field(default_factory=new_id)
    signal_class: str = "unknown"
    class_confidence: float = 0.0
    novelty_score: float = 0.0

    bearing_deg: float | None = None
    bearing_sigma_deg: float | None = None
    aircraft_alt_m: float = 120.0

    warrant_match: bool = False
    warrant_id: str | None = None
    warrant_reasons: list[str] = field(default_factory=list)
    recording_ref: str | None = None

    model_version: str = "none"

    # Simulator-only ground truth. MUST be None on real hardware; the
    # ground station asserts this before anything reaches an operator.
    truth_emitter_id: str | None = None
    truth_class: str | None = None
    truth_xy: tuple[float, float] | None = None

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aircraft_pos_xy"] = list(self.aircraft_pos_xy)
        if self.truth_xy is not None:
            d["truth_xy"] = list(self.truth_xy)
        return d

    @classmethod
    def from_json(cls, s: str) -> "Detection":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Detection":
        d = dict(d)
        required = ("aircraft_id", "t_gpsdo_ns", "freq_hz",
                    "bandwidth_hz", "snr_db", "aircraft_pos_xy")
        for k in required:
            if k not in d:
                raise ValueError(f"Detection missing required field: {k}")
        d["aircraft_pos_xy"] = tuple(d["aircraft_pos_xy"])
        if d.get("truth_xy") is not None:
            d["truth_xy"] = tuple(d["truth_xy"])
        known = cls.__dataclass_fields__.keys()
        unknown = set(d) - set(known)
        if unknown:
            raise ValueError(f"Detection has unknown fields: {sorted(unknown)}")
        return cls(**d)

    def sanitized(self) -> "Detection":
        """Strip simulator ground truth. Called before anything leaves the sim."""
        d = self.to_dict()
        d["truth_emitter_id"] = None
        d["truth_class"] = None
        d["truth_xy"] = None
        return Detection.from_dict(d)


# --------------------------------------------------------------------------
# GeoFix: produced by ground fusion (E). Never a bare coordinate — the
# ellipse is part of the contract (Doc 07 §6).
# --------------------------------------------------------------------------
@dataclass
class ErrorEllipse:
    semi_major_m: float
    semi_minor_m: float
    orientation_deg: float
    confidence: float = 0.95

    def area_m2(self) -> float:
        import math
        return math.pi * self.semi_major_m * self.semi_minor_m


@dataclass
class GeoFix:
    emitter_id: str
    xy: tuple[float, float]
    error_ellipse: ErrorEllipse
    method: str
    n_aircraft: int
    n_observations: int
    quality: str = "good"          # good | degraded | void
    gdop: float | None = None
    contributing_detections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["xy"] = list(self.xy)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


# --------------------------------------------------------------------------
# RecordingSidecar: metadata written next to every sealed IQ file.
# --------------------------------------------------------------------------
@dataclass
class RecordingSidecar:
    recording_ref: str
    warrant_id: str
    detection_ref: str
    start_ns: int
    stop_ns: int
    sha256: str
    sealed_to_pubkey: str
    freq_hz: float
    signal_class: str
    gps_track: list[tuple[float, float, float, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
