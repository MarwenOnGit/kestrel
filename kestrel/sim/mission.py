"""Full-mission simulation: the vertical slice, end to end (Doc 11 Step 6.5).

    KestrelSim -> Detection -> classifier -> WARRANT GATE -> sealer
                            -> EmitterTrack -> planner -> safety -> waypoint
                            -> audit log -> GeoFix reports

Every stage is the real implementation except the SDR and the classifier
network. Swapping those in is Steps 3, 4 and 8 of the plan; nothing else has to
change, which is the point of building this first.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kestrel.air.agent import MissionAgent
from kestrel.air.classifier import StubClassifier
from kestrel.air.planner import InfoGainPlanner
from kestrel.air.safety import SafetyLayer
from kestrel.air.sealer import CollectionController, Sealer
from kestrel.air.track import TrackStore
from kestrel.air.warrant_gate import WarrantGate
from kestrel.common import crypto
from kestrel.common.warrant import WarrantToken
from kestrel.ground import signer
from kestrel.ground.audit import AuditLog
from kestrel.sim.world import Aircraft, KestrelSim


@dataclass
class MissionResult:
    duration_s: float
    detections: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    recordings: list = field(default_factory=list)
    tracks: dict = field(default_factory=dict)
    audit_path: str = ""
    vault_dir: str = ""
    audit_ok: bool = True
    safety_violations: int = 0
    emitters: list = field(default_factory=list)
    token: Any = None
    workdir: str = ""


def run_mission(sim: KestrelSim, *, duration_s: float = 1800.0, dt: float = 1.0,
                warrant_overrides: dict | None = None,
                planner_factory=None, workdir: str | None = None,
                seed: int = 0, geofence=None, revoke_at_s: float | None = None,
                durable: bool = False) -> MissionResult:
    """Fly one sortie. Returns everything needed to evaluate it."""
    workdir = workdir or tempfile.mkdtemp(prefix="kestrel_sortie_")
    os.makedirs(workdir, exist_ok=True)

    # ---- ground: keys and a signed warrant -----------------------------
    auth_sk, auth_pk = crypto.generate_authority_keypair()
    seal_sk, seal_pk = crypto.generate_seal_keypair()   # seal_sk stays on the ground
    raw_token = signer.quick_warrant(seal_pk, auth_sk, **(warrant_overrides or {}))
    token = WarrantToken.load(raw_token, auth_pk)

    audit = AuditLog(os.path.join(workdir, "audit.jsonl"), durable=durable)
    audit.append("SORTIE_START", {"warrant_id": token.warrant_id,
                                  "n_aircraft": len(sim.aircraft)})

    geofence = geofence or token.body["area_polygon"]

    # ---- per-aircraft airborne stack -----------------------------------
    agents: dict[str, MissionAgent] = {}
    area = _area_from_polygon(geofence)
    for ac in sim.aircraft:
        gate = WarrantGate(token, audit)
        sealer = Sealer(seal_pk, os.path.join(workdir, "vault", ac.aircraft_id),
                        durable=durable)
        controller = CollectionController(gate, sealer, audit)
        safety = SafetyLayer(geofence, home_xy=tuple(ac.xy))
        planner = (planner_factory() if planner_factory else InfoGainPlanner())
        agents[ac.aircraft_id] = MissionAgent(
            aircraft_id=ac.aircraft_id,
            planner=planner,
            safety=safety,
            area=area,
            controller=controller,
            track_store=TrackStore(),
            classifier=StubClassifier(seed=seed + hash(ac.aircraft_id) % 1000),
            audit=audit,
            seed=seed,
        )

    # ---- fly ------------------------------------------------------------
    all_dets, waypoints = [], {}
    steps = int(duration_s / dt)
    now = 0.0
    for _ in range(steps):
        dets = sim.step(dt, waypoints)
        all_dets.extend(dets)
        now += dt

        if revoke_at_s is not None and now >= revoke_at_s:
            if not token.revoked:
                token.revoke()
                audit.append("REVOCATION", {"t": now})

        for ac in sim.alive_aircraft:
            ag = agents[ac.aircraft_id]
            neighbors = [tuple(o.xy) for o in sim.alive_aircraft
                         if o.aircraft_id != ac.aircraft_id]
            waypoints[ac.aircraft_id] = ag.tick(
                dets, tuple(ac.xy), ac.battery_pct, dt,
                neighbors=neighbors, now=_wall_now(token, now))
            ag.safety.check_position(tuple(ac.xy))

    audit.append("SORTIE_END", {"t": now})
    ok, _bad = audit.verify()

    tracks = {}
    for ag in agents.values():
        tracks.update(ag.tracks.tracks)

    return MissionResult(
        duration_s=duration_s,
        detections=all_dets,
        reports=[r for ag in agents.values() for r in ag.reports],
        recordings=[r for ag in agents.values() for r in ag.controller.recordings],
        tracks=tracks,
        audit_path=audit.path,
        vault_dir=os.path.join(workdir, "vault"),
        audit_ok=ok,
        safety_violations=sum(ag.safety.violations for ag in agents.values()),
        emitters=sim.emitters,
        token=token,
        workdir=workdir,
    )


def _wall_now(token, sim_t: float) -> float:
    """Map simulated elapsed time onto the token's validity window, so that
    expiry behaviour can be exercised in simulation."""
    from kestrel.common.warrant import parse_iso8601
    return parse_iso8601(token.body["valid_from"]) + sim_t + 300.0


def _area_from_polygon(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return ((min(xs), min(ys)), (max(xs), max(ys)))


def default_aircraft(n: int, area, alt_m: float = 120.0) -> list[Aircraft]:
    (x0, y0), (x1, y1) = area
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    out = []
    for i in range(n):
        th = 2 * np.pi * i / max(n, 1)
        out.append(Aircraft(f"K{i+1}",
                            xy=(cx + 300 * np.cos(th), cy + 300 * np.sin(th)),
                            alt_m=alt_m))
    return out
