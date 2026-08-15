"""Mission metrics (Doc 09 §5).

`over_collection_count` must be zero in every scenario, every run. It is a
build-breaking assertion, not a tuning parameter.
"""
from __future__ import annotations

import math

import numpy as np


def evaluate(result, token=None) -> dict:
    token = token or result.token
    emitters = {e.emitter_id: e for e in result.emitters}

    detected = {d.truth_emitter_id for d in result.detections
                if d.truth_emitter_id is not None}
    detection_rate = len(detected) / max(len(emitters), 1)

    # --- fix accuracy against ground truth ------------------------------
    errors, contained, unresolved = [], [], 0
    for t in result.tracks.values():
        if t.truth_xy is None or t.n_observations < 2:
            continue
        t._ensure_solved()
        if not t.resolved:
            # Geometry never made the range observable. Counted, not published.
            unresolved += 1
            continue
        err = float(np.linalg.norm(np.asarray(t.truth_xy) - t.xy))
        errors.append(err)
        contained.append(bool(t.contains(t.truth_xy, 0.95)))

    # --- the legal invariant --------------------------------------------
    over = 0
    for rec in result.recordings:
        if not token.covers(rec.freq_hz, rec.signal_class):
            over += 1
    for rec in result.recordings:
        if rec.warrant_id != token.warrant_id:
            over += 1

    ellipse_areas = [t.ellipse_area_m2() for t in result.tracks.values()
                     if t.n_observations >= 2 and t.resolved]

    return {
        "n_emitters": len(emitters),
        "n_detections": len(result.detections),
        "n_tracks": len(result.tracks),
        "n_unresolved_tracks": unresolved,
        "n_reports": len(result.reports),
        "n_recordings": len(result.recordings),
        "detection_rate": detection_rate,
        "median_fix_error_m": float(np.median(errors)) if errors else math.inf,
        "p95_fix_error_m": float(np.percentile(errors, 95)) if errors else math.inf,
        "median_ellipse_area_m2": float(np.median(ellipse_areas)) if ellipse_areas else math.inf,
        "ellipse_containment": float(np.mean(contained)) if contained else 0.0,
        "over_collection_count": over,
        "safety_violations": result.safety_violations,
        "audit_ok": result.audit_ok,
    }


def assert_invariants(m: dict) -> None:
    """Build-breaking checks (Doc 10 §8)."""
    assert m["over_collection_count"] == 0, \
        f"OVER-COLLECTION: {m['over_collection_count']} recordings outside the warrant"
    assert m["safety_violations"] == 0, \
        f"SAFETY: {m['safety_violations']} geofence violations"
    assert m["audit_ok"], "AUDIT CHAIN FAILED VERIFICATION"


def format_metrics(m: dict) -> str:
    lines = ["", "  metric                     value", "  " + "-" * 40]
    for k, v in m.items():
        if isinstance(v, float):
            v = f"{v:,.2f}" if math.isfinite(v) else "n/a"
        lines.append(f"  {k:<26} {v}")
    return "\n".join(lines)
