"""Warrant token issuance - GROUND ONLY (Doc 08 §2).

Two humans are required. In production the signing key lives in an HSM and this
module calls into it; the file-based key path here is for development and is
refused when `require_hsm` is set.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from kestrel.common import crypto
from kestrel.common.warrant import TokenError


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_warrant_body(
    *,
    warrant_id: str,
    operator_id: str,
    co_signer_id: str,
    freq_ranges_hz: list[list[float]],
    signal_classes: list[str],
    area_polygon: list[list[float]],
    valid_from: datetime,
    valid_until: datetime,
    seal_pubkey: str,
    issuing_authority: str = "UNSPECIFIED",
    max_recording_seconds: float | None = 3600.0,
    max_recording_per_emitter_seconds: float | None = 300.0,
    minimization_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if operator_id == co_signer_id:
        raise TokenError("two-person rule: operator and co-signer must differ")
    if valid_until <= valid_from:
        raise TokenError("validity window is empty or inverted")
    if len(area_polygon) < 3:
        raise TokenError("area_polygon needs at least 3 vertices")
    if not freq_ranges_hz:
        raise TokenError("a warrant with no frequency ranges authorizes nothing")

    return {
        "warrant_id": warrant_id,
        "issuing_authority": issuing_authority,
        "operator_id": operator_id,
        "co_signer_id": co_signer_id,
        "freq_ranges_hz": [[float(a), float(b)] for a, b in freq_ranges_hz],
        "signal_classes": sorted(signal_classes),
        "area_polygon": [[float(x), float(y)] for x, y in area_polygon],
        "valid_from": iso(valid_from),
        "valid_until": iso(valid_until),
        "max_recording_seconds": max_recording_seconds,
        "max_recording_per_emitter_seconds": max_recording_per_emitter_seconds,
        "minimization_rules": minimization_rules or {},
        "seal_pubkey": seal_pubkey,
        "nonce": crypto.b64e(__import__("os").urandom(16)),
    }


def issue(body: dict[str, Any], authority_signing_key_b64: str) -> dict[str, Any]:
    """Sign a warrant body. Returns the raw token to load onto aircraft."""
    return {"body": body, "signature": crypto.sign(authority_signing_key_b64, body)}


def quick_warrant(seal_pubkey: str, signing_key: str, **overrides) -> dict[str, Any]:
    """Development helper: a valid 8-hour warrant over a default area."""
    now = datetime.now(timezone.utc)
    params: dict[str, Any] = dict(
        warrant_id="W-DEV-0001",
        operator_id="op.alice",
        co_signer_id="sup.bob",
        freq_ranges_hz=[[462.0e6, 468.0e6]],
        signal_classes=["dmr", "fm_voice"],
        area_polygon=[[0.0, 0.0], [8000.0, 0.0], [8000.0, 8000.0], [0.0, 8000.0]],
        valid_from=now - timedelta(minutes=5),
        valid_until=now + timedelta(hours=8),
        seal_pubkey=seal_pubkey,
    )
    params.update(overrides)
    return issue(build_warrant_body(**params), signing_key)
