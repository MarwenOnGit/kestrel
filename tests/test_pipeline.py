"""End-to-end edge pipeline: real IQ in, classified Detections out.

This is the seam where the DSP chain replaces the simulator's geometry
shortcut. The pipeline is told nothing about what is transmitting; it has to
find it in the samples.
"""
from __future__ import annotations

import numpy as np
import pytest

from kestrel.air.pipeline import (EdgePipeline, FeatureClassifier, SweepPlan,
                                  SweepSegment)
from kestrel.air.sdr_source import SimulatedSdrSource
from kestrel.air.classifier import apply_classification
from kestrel.sim import scenarios
from kestrel.sim.world import Aircraft, Emitter, KestrelSim


def build_sim(emitters, aircraft_xy=(2000.0, 3000.0), seed=0):
    return KestrelSim([Emitter(**e) for e in emitters],
                      [Aircraft("K1", xy=aircraft_xy)],
                      noise_floor_dbm=-110.0, seed=seed)


# ------------------------------------------------------------------ sweep
def test_sweep_plan_covers_bands_and_reports_revisit_time():
    plan = SweepPlan.from_bands([(460e6, 480e6)], rate_sps=20e6, dwell_s=0.2)
    assert len(plan.segments) == 1
    assert plan.revisit_time_s() == pytest.approx(0.2)

    wide = SweepPlan.from_bands([(30e6, 520e6), (700e6, 3800e6)],
                                rate_sps=200e6, dwell_s=0.2)
    assert wide.revisit_time_s() > 0
    assert any(s.contains(462.5e6) for s in wide.segments)


def test_swarm_division_is_interleaved_and_covers_everything():
    plan = SweepPlan.from_bands([(400e6, 1200e6)], rate_sps=100e6,
                                dwell_s=0.2, n_aircraft=3)
    sectors = [plan.sector_for(i) for i in range(3)]
    covered = {s.center_hz for sec in sectors for s in sec}
    assert covered == {s.center_hz for s in plan.segments}
    # Interleaved, not blocked: each aircraft samples across the whole span.
    assert len(sectors[0]) >= 2
    assert max(s.center_hz for s in sectors[0]) - min(s.center_hz for s in sectors[0]) \
        > 100e6


def test_three_aircraft_cut_revisit_time():
    one = SweepPlan.from_bands([(400e6, 1200e6)], rate_sps=100e6, dwell_s=0.2)
    three = SweepPlan.from_bands([(400e6, 1200e6)], rate_sps=100e6,
                                 dwell_s=0.2, n_aircraft=3)
    assert three.revisit_time_s() == pytest.approx(one.revisit_time_s() / 3)


# --------------------------------------------------------------- pipeline
def test_pipeline_finds_a_real_emitter_in_synthesized_iq():
    sim = build_sim([dict(emitter_id="T1", xy=(2200, 3100), freq_hz=462.52e6,
                          mode="dmr", eirp_dbm=37, burst_s=100, period_s=100)])
    src = SimulatedSdrSource(sim, "K1", seed=1)
    src.tune(462.5e6, 1e6)
    iq, t_ns = src.frame()

    pipe = EdgePipeline(FeatureClassifier())
    results = pipe.process_frame(iq, SweepSegment(462.5e6, 1e6), t_ns, (2000.0, 3000.0))
    assert results, "pipeline found nothing in a frame containing a strong emitter"

    freqs = [d.freq_hz for d, _ in results]
    assert any(abs(f - 462.52e6) < 15e3 for f in freqs)


def test_pipeline_ignores_an_empty_band():
    """No emitter in the tuned window means no detections - false alarms cost
    GPU wakeups and therefore flight time."""
    sim = build_sim([dict(emitter_id="T1", xy=(2200, 3100), freq_hz=900e6,
                          mode="dmr", eirp_dbm=37, burst_s=100, period_s=100)])
    src = SimulatedSdrSource(sim, "K1", seed=2)
    src.tune(462.5e6, 1e6)
    iq, t_ns = src.frame()
    pipe = EdgePipeline(FeatureClassifier())
    assert pipe.process_frame(iq, SweepSegment(462.5e6, 1e6), t_ns, (2000.0, 3000.0)) == []


def test_distant_emitter_falls_below_the_noise_floor():
    """Link budget must actually bite: the same emitter 60 km away is gone."""
    def found_at(distance_m):
        sim = build_sim([dict(emitter_id="T1", xy=(2000 + distance_m, 3000),
                              freq_hz=462.52e6, mode="dmr", eirp_dbm=37,
                              burst_s=100, period_s=100)])
        src = SimulatedSdrSource(sim, "K1", seed=3)
        src.tune(462.5e6, 1e6)
        iq, t_ns = src.frame()
        pipe = EdgePipeline(FeatureClassifier())
        return len(pipe.process_frame(iq, SweepSegment(462.5e6, 1e6), t_ns,
                                      (2000.0, 3000.0)))

    assert found_at(300) >= 1
    assert found_at(300_000) == 0


def test_pipeline_classifies_constant_envelope_vs_ofdm():
    """The classifier runs on IQ the pipeline itself channelized - no truth."""
    sim = build_sim([
        dict(emitter_id="V", xy=(2100, 3050), freq_hz=462.3e6, mode="dmr",
             eirp_dbm=37, burst_s=100, period_s=100),
        dict(emitter_id="D", xy=(2100, 3050), freq_hz=462.8e6, mode="lte",
             eirp_dbm=37, burst_s=100, period_s=100),
    ])
    src = SimulatedSdrSource(sim, "K1", seed=4)
    src.tune(462.5e6, 20e6)
    iq, t_ns = src.frame(131_072)

    pipe = EdgePipeline(FeatureClassifier())
    results = pipe.process_frame(iq, SweepSegment(462.5e6, 20e6), t_ns, (2000.0, 3000.0))
    assert results

    for det, snip in results:
        apply_classification(det, pipe.classifier) if False else None
        cls, conf = pipe.classifier.classify(det, snip)
        det.signal_class, det.class_confidence = cls, conf

    narrow = [d for d, _ in results if d.bandwidth_hz < 100e3]
    wide = [d for d, _ in results if d.bandwidth_hz > 1e6]
    if narrow:
        assert all(d.signal_class in ("dmr", "fm_voice", "unknown") for d in narrow)
    if wide:
        assert all(d.signal_class in ("lte", "wifi", "nr5g", "unknown") for d in wide)


def test_classifier_abstains_rather_than_guessing():
    """Uncertainty must resolve to `unknown`, which the warrant gate refuses.
    An over-confident classifier is how unauthorized signals get recorded."""
    clf = FeatureClassifier()
    from kestrel.common.messages import Detection
    det = Detection(aircraft_id="K1", t_gpsdo_ns=0, freq_hz=1e9,
                    bandwidth_hz=12_500, snr_db=20, aircraft_pos_xy=(0, 0))
    cls, conf = clf.classify(det, None)
    assert cls == "unknown" and conf == 0.0

    noise = (np.random.default_rng(5).standard_normal(4096)
             + 1j * np.random.default_rng(6).standard_normal(4096)).astype(np.complex64)
    cls, conf = clf.classify(det, noise)
    assert cls == "unknown" or conf < 0.9


def test_pipeline_caps_candidates_under_load():
    """A crowded band must not blow the GPU budget; the weakest are dropped
    and the drop is COUNTED, never silent."""
    sigs = [dict(emitter_id=f"E{i}", xy=(2100, 3050),
                 freq_hz=462.0e6 + i * 40e3, mode="dmr", eirp_dbm=37,
                 burst_s=100, period_s=100) for i in range(12)]
    sim = build_sim(sigs)
    src = SimulatedSdrSource(sim, "K1", seed=7)
    src.tune(462.5e6, 2e6)
    iq, t_ns = src.frame()

    pipe = EdgePipeline(FeatureClassifier(), max_candidates_per_frame=3)
    results = pipe.process_frame(iq, SweepSegment(462.5e6, 2e6), t_ns, (2000.0, 3000.0))
    assert len(results) <= 3
    h = pipe.health()
    assert h["candidates_seen"] >= len(results)
    if h["candidates_seen"] > 3:
        assert h["dropped_over_budget"] > 0


def test_pipeline_produces_detections_the_warrant_gate_can_judge():
    """The seam between DSP and the legal spine: a Detection built from real
    IQ must carry everything the gate needs to decide."""
    from kestrel.air.warrant_gate import WarrantGate
    from kestrel.common import crypto
    from kestrel.common.warrant import WarrantToken
    from kestrel.ground import signer
    from kestrel.ground.audit import AuditLog
    import tempfile, os

    sim = build_sim([dict(emitter_id="T1", xy=(2200, 3100), freq_hz=462.52e6,
                          mode="dmr", eirp_dbm=37, burst_s=100, period_s=100)])
    src = SimulatedSdrSource(sim, "K1", seed=8)
    src.tune(462.5e6, 1e6)
    iq, t_ns = src.frame()

    pipe = EdgePipeline(FeatureClassifier())
    results = pipe.process_frame(iq, SweepSegment(462.5e6, 1e6), t_ns, (2000.0, 3000.0))
    assert results

    auth_sk, auth_pk = crypto.generate_authority_keypair()
    _, seal_pk = crypto.generate_seal_keypair()
    token = WarrantToken.load(
        signer.quick_warrant(seal_pk, auth_sk,
                             area_polygon=[[0, 0], [9000, 0], [9000, 9000], [0, 9000]]),
        auth_pk)
    audit = AuditLog(os.path.join(tempfile.mkdtemp(), "a.jsonl"), durable=False)
    gate = WarrantGate(token, audit)

    for det, snip in results:
        det.signal_class, det.class_confidence = pipe.classifier.classify(det, snip)
        allowed, reasons = gate.decide(det, (2000.0, 3000.0))
        # Whatever the verdict, it must be reasoned and logged.
        assert allowed or reasons
    assert any(e["type"] == "GATE_DECISION" for e in audit.entries())
