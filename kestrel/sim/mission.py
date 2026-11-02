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
from kestrel.air.pipeline import EdgePipeline, FeatureClassifier, SweepPlan
from kestrel.air.planner import InfoGainPlanner
from kestrel.air.safety import SafetyLayer
from kestrel.air.sdr_source import SimulatedSdrSource
from kestrel.air.sealer import CollectionController, Sealer
from kestrel.common.identity import generate_airframe_identity
from kestrel.air.sensor import RfSensor
from kestrel.air.track import TrackStore
from kestrel.air.warrant_gate import WarrantGate
from kestrel.common import crypto
from kestrel.common.warrant import WarrantToken
from kestrel.ground import signer
from kestrel.ground.audit import AuditLog, verify_audit
from kestrel.ground.evidence import verify_vault
from kestrel.sim.simulated_df import SimulatedDf
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
    sensor_health: dict = field(default_factory=dict)
    airframe_registry: dict = field(default_factory=dict)
    # {recording_ref: reason} for every recording the ground would refuse.
    evidence_failures: dict = field(default_factory=dict)


def run_mission(sim: KestrelSim, *, duration_s: float = 1800.0, dt: float = 1.0,
                warrant_overrides: dict | None = None,
                planner_factory=None, workdir: str | None = None,
                seed: int = 0, geofence=None, revoke_at_s: float | None = None,
                durable: bool = False,
                iq_mode: bool = False, sweep_bands=None,
                sweep_rate_sps: float = 2e6, dwell_s: float = 0.2,
                max_frames_per_step: int | None = None) -> MissionResult:
    """Fly one sortie. Returns everything needed to evaluate it.

    Two sensing paths, one mission loop:

      * default - `KestrelSim.generate_detections()` produces Detections from
        geometry. ~1000x real time, which is what makes autonomy development
        and 20-seed regression runs practical.
      * `iq_mode=True` - each aircraft carries an `RfSensor`: a real SDR
        source, a real sweep with real dwell accounting, and the real DSP chain
        (PSD -> CFAR -> channelize -> features -> classify). Detections are
        *found*, not handed over, and the IQ that reaches the sealer is the
        actual signal rather than a placeholder.

    The second is what will run in the air. It is roughly 100x slower and it
    finds less, both of which are the point: everything the sweep misses is
    something a real receiver would also have missed.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="kestrel_sortie_")
    os.makedirs(workdir, exist_ok=True)

    # ---- ground: keys and a signed warrant -----------------------------
    auth_sk, auth_pk = crypto.generate_authority_keypair()
    seal_sk, seal_pk = crypto.generate_seal_keypair()   # seal_sk stays on the ground
    raw_token = signer.quick_warrant(seal_pk, auth_sk, **(warrant_overrides or {}))
    token = WarrantToken.load(raw_token, auth_pk)

    # The simulator keeps ONE log for the whole swarm, so it gets its own
    # identity. Flight hardware keeps one log per airframe, checkpointed with
    # that airframe's key. `downlinked` is what the ground receives over the
    # datalink as each checkpoint is made.
    airframe_registry: dict[str, str] = {}
    downlinked: list[dict] = []
    log_identity = generate_airframe_identity("swarm-log")
    airframe_registry[log_identity.airframe_id] = log_identity.verify_key_b64
    audit = AuditLog(os.path.join(workdir, "audit.jsonl"), durable=durable,
                     identity=log_identity, on_checkpoint=downlinked.append)
    audit.append("SORTIE_START", {"warrant_id": token.warrant_id,
                                  "n_aircraft": len(sim.aircraft)})

    geofence = geofence or token.body["area_polygon"]

    # ---- per-aircraft airborne stack -----------------------------------
    agents: dict[str, MissionAgent] = {}
    sensors: dict[str, Any] = {}
    dfs: dict[str, Any] = {}
    area = _area_from_polygon(geofence)

    if iq_mode:
        # Sweep the bands the warrant covers. Detection is not limited by the
        # warrant - the gate decides what may be RECORDED, not what may be
        # heard - but a sortie is flown to serve a warrant, and receiver time
        # spent outside its bands is time not spent on the task.
        bands = sweep_bands or [tuple(r) for r in token.body["freq_ranges_hz"]]
        sweep = SweepPlan.from_bands(bands, rate_sps=sweep_rate_sps,
                                     dwell_s=dwell_s,
                                     n_aircraft=len(sim.aircraft))
        audit.append("SWEEP_PLAN", {
            "bands_hz": [list(b) for b in bands],
            "segments": len(sweep.segments),
            "rate_sps": sweep_rate_sps,
            "dwell_s": dwell_s,
            "revisit_time_s": sweep.revisit_time_s()})

    for ac in sim.aircraft:
        gate = WarrantGate(token, audit)
        # Provisioned on the ground; only the verify key is registered.
        identity = generate_airframe_identity(ac.aircraft_id)
        airframe_registry[ac.aircraft_id] = identity.verify_key_b64
        sealer = Sealer(seal_pk, os.path.join(workdir, "vault", ac.aircraft_id),
                        identity=identity, durable=durable)
        controller = CollectionController(gate, sealer, audit)
        safety = SafetyLayer(geofence, home_xy=tuple(ac.xy))
        planner = (planner_factory() if planner_factory else InfoGainPlanner())

        if iq_mode:
            idx = sim.aircraft.index(ac)
            source = SimulatedSdrSource(sim, ac.aircraft_id, seed=seed + idx)
            df = SimulatedDf(sim, source, ac.aircraft_id)
            classifier = FeatureClassifier()
            sensors[ac.aircraft_id] = RfSensor(
                source=source,
                pipeline=EdgePipeline(classifier, aircraft_id=ac.aircraft_id),
                sweep_plan=sweep,
                sector=sweep.sector_for(idx),
                df=df,
                max_frames_per_step=max_frames_per_step)
            dfs[ac.aircraft_id] = df
        else:
            classifier = StubClassifier(seed=seed + hash(ac.aircraft_id) % 1000)

        agents[ac.aircraft_id] = MissionAgent(
            aircraft_id=ac.aircraft_id,
            planner=planner,
            safety=safety,
            area=area,
            controller=controller,
            track_store=TrackStore(),
            classifier=classifier,
            audit=audit,
            seed=seed,
        )

    # ---- fly ------------------------------------------------------------
    all_dets, waypoints = [], {}
    steps = int(duration_s / dt)
    now = 0.0
    for _ in range(steps):
        # In IQ mode the world still advances - aircraft fly, emitters key up
        # and stop - but it hands out no detections: the sensor must find them
        # in samples or not at all.
        world_dets = sim.step(dt, waypoints, generate=not iq_mode)
        all_dets.extend(world_dets)
        now += dt

        if revoke_at_s is not None and now >= revoke_at_s:
            if not token.revoked:
                token.revoke()
                audit.append("REVOCATION", {"t": now})

        for ac in sim.alive_aircraft:
            ag = agents[ac.aircraft_id]
            neighbors = [tuple(o.xy) for o in sim.alive_aircraft
                         if o.aircraft_id != ac.aircraft_id]

            if iq_mode:
                dets, iq_by_det = _sense(sim, sensors[ac.aircraft_id],
                                         dfs[ac.aircraft_id], ac, dt)
                all_dets.extend(dets)
            else:
                dets, iq_by_det = world_dets, None

            waypoints[ac.aircraft_id] = ag.tick(
                dets, tuple(ac.xy), ac.battery_pct, dt,
                neighbors=neighbors, now=_wall_now(token, now),
                iq_by_detection=iq_by_det)
            ag.safety.check_position(tuple(ac.xy))

    for acid, s in sensors.items():
        audit.append("SENSOR_HEALTH", {"aircraft_id": acid, **s.health()})
    audit.append("SORTIE_END", {"t": now})
    audit.close()
    ok, _reason, _seq = verify_audit(audit.path, airframe_registry, downlinked)

    # ---- ground: authenticate every recording before anything is opened --
    evidence_failures: dict[str, str] = {}
    for acid, ag in agents.items():
        evidence_failures.update(verify_vault(
            [r.to_dict() for r in ag.controller.recordings],
            os.path.join(workdir, "vault", acid), airframe_registry))

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
        sensor_health={k: s.health() for k, s in sensors.items()},
        airframe_registry=airframe_registry,
        evidence_failures=evidence_failures,
    )


def _sense(sim, sensor, df, ac, dt: float):
    """One aircraft's slice of receiver time, through the real DSP chain.

    The two simulator-only touches are isolated here, both after the pipeline
    has committed to what it found:
      * the reported position carries GPS error, as the detection-level path
        does - the aircraft does not know exactly where it is;
      * timestamps and truth tags are applied by the simulated DF.
    Neither can change what was detected, only how it is labelled.
    """
    reported = sim.reported_position(ac)
    results = sensor.sense(dt, (float(reported[0]), float(reported[1])),
                           alt_m=ac.alt_m, on_detection=df.annotate)
    dets = [det for det, _ in results]
    iq_by_det = {det.detection_id: snippet for det, snippet in results}
    return dets, iq_by_det


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
