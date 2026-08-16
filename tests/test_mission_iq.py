"""The mission loop flying on real IQ (Doc 11 Steps 3 + 6.5 joined).

`tests/test_pipeline.py` proves the DSP chain works on a frame. These prove it
works for a *sortie*: that a sweep with finite dwell finds emitters in samples
rather than being handed them, that what reaches the vault is the signal and
not a placeholder, and that every legal invariant still holds when detections
come from measurement instead of from geometry.

Everything here is slow by design - real synthesis and real FFTs - so the runs
are short and the frame budget is capped. `dwell_s_unwatched` records what that
cap cost, so a capped run cannot be mistaken for full coverage.
"""
from __future__ import annotations

import numpy as np
import pytest

from kestrel.sim import scenarios
from kestrel.sim.metrics import evaluate
from kestrel.sim.mission import run_mission

pytestmark = pytest.mark.slow

IQ_KWARGS = dict(iq_mode=True, dt=1.0, max_frames_per_step=2)


@pytest.fixture(scope="module")
def sortie():
    """One IQ-mode sortie, shared: it costs ~20 s of CPU to fly."""
    sc = scenarios.get("border_crossing")
    sim = sc.build(seed=1)
    return run_mission(sim, duration_s=600.0, warrant_overrides=sc.warrant,
                       seed=1, **IQ_KWARGS)


# --------------------------------------------------------------- detection
def test_emitters_are_found_in_samples(sortie):
    """The pipeline is told nothing. It must find the emitters itself."""
    assert sortie.detections, "no detections from a scene with in-band traffic"
    found = {d.truth_emitter_id for d in sortie.detections}
    assert "TGT-1" in found or "TGT-2" in found


def test_no_phantom_emitters(sortie):
    """Every detection lands on a real emitter's frequency.

    An untagged detection is one the simulated DF could not associate with
    anything transmitting - a false alarm, an image, or splatter. A handful is
    tolerable; a stream of them means the detector or the synthesizer is
    manufacturing signals, which is the failure that makes IQ-level simulation
    worse than useless.
    """
    untagged = [d for d in sortie.detections if d.truth_emitter_id is None]
    assert len(untagged) <= 0.05 * len(sortie.detections), (
        f"{len(untagged)}/{len(sortie.detections)} detections match no emitter")


def test_measured_frequency_is_close_to_truth(sortie):
    """Frequency comes from the power centroid of a channel, not from truth."""
    truth = {e.emitter_id: e.freq_hz for e in sortie.emitters}
    for d in sortie.detections:
        if d.truth_emitter_id is None:
            continue
        assert abs(d.freq_hz - truth[d.truth_emitter_id]) < 2000.0


def test_detections_without_association_get_no_bearing(sortie):
    """A detection the DF cannot associate contributes nothing to a track.

    This is what keeps ground truth from quietly repairing the DSP's mistakes:
    a false alarm stays a false alarm instead of acquiring a plausible bearing.
    """
    for d in sortie.detections:
        if d.truth_emitter_id is None:
            assert d.bearing_deg is None


# ------------------------------------------------------------------- legal
def test_what_reaches_the_vault_is_actual_signal(sortie):
    """The recorded IQ is the detected signal, not the zero-filled placeholder.

    Zeros would hash identically every time; distinct digests over distinct
    snippets are the evidence that real samples were sealed. This closes the
    last shortcut listed in docs/13 §4.3.
    """
    assert sortie.recordings, "nothing was recorded from an authorized scene"
    digests = {r.sha256 for r in sortie.recordings}
    assert len(digests) == len(sortie.recordings), "identical recording contents"

    zeros = np.zeros(256, dtype=np.complex64)
    from kestrel.common.crypto import sha256_hex
    assert sha256_hex(zeros.tobytes()) not in digests


def test_recorded_seconds_come_from_the_samples_held(sortie):
    """The warrant's budget is charged for real signal time (Doc 08 §3)."""
    recorded = [d for d in sortie.detections if d.recording_ref is not None]
    assert recorded
    assert all(d.capture_duration_s > 0 for d in recorded)

    charged = sortie.token.recorded_seconds
    expected = sum(d.capture_duration_s for d in recorded)
    assert charged == pytest.approx(expected, rel=1e-6)


def test_no_over_collection_on_the_iq_path(sortie):
    m = evaluate(sortie)
    assert m["over_collection_count"] == 0
    for rec in sortie.recordings:
        assert sortie.token.covers(rec.freq_hz, rec.signal_class)


def test_out_of_band_emitters_are_never_recorded(sortie):
    """SAT-1 at 1626 MHz and TRK-1 at 868 MHz are outside the warrant's bands,
    so the sweep never tunes there and nothing about them is collected."""
    for rec in sortie.recordings:
        assert 462.0e6 <= rec.freq_hz <= 468.0e6
    assert not [d for d in sortie.detections
                if d.truth_emitter_id in ("SAT-1", "TRK-1")]


def test_audit_verifies_and_carries_no_ground_truth(sortie):
    from kestrel.ground.audit import AuditLog
    assert sortie.audit_ok
    log = AuditLog(sortie.audit_path)
    for entry in log.entries():
        blob = str(entry)
        assert "truth_" not in blob
        assert "TGT-1" not in blob and "DECOY-farm" not in blob


def test_safety_holds(sortie):
    assert sortie.safety_violations == 0


# ------------------------------------------------------------------ sweep
def test_dwell_accounting_is_conserved(sortie):
    """Receiver time is the resource the aircraft cannot cheat on: every
    second of the sortie is either spent on a segment or explicitly recorded
    as spectrum nobody watched."""
    h = sortie.sensor_health["K1"]
    assert h["dwell_s_used"] + h["dwell_s_unwatched"] == pytest.approx(
        sortie.duration_s, rel=1e-6)
    assert h["dwell_s_unwatched"] > 0, "capped run should report the shortfall"
    assert h["sweep_frames"] > 0
    assert h["bearings"] <= h["detections"]


def test_revisit_time_bounds_what_can_be_heard():
    """A sweep slower than an emitter's duty cycle misses it - not through a
    bug, but because the receiver was elsewhere. The metric that predicts this
    must be reported, so plan it and check it (Doc 04 §7)."""
    from kestrel.air.pipeline import SweepPlan
    plan = SweepPlan.from_bands([(462e6, 468e6)], rate_sps=2e6, dwell_s=0.2)
    # 6 MHz of band on a 2 MHz radio needs 4 overlapping segments, not 3:
    # abutting them would leave every seam frequency inside a guard band.
    assert len(plan.segments) == 4
    assert plan.revisit_time_s() == pytest.approx(len(plan.segments) * 0.2)

    # Two aircraft split the sector, so each band comes round twice as often.
    plan2 = SweepPlan.from_bands([(462e6, 468e6)], rate_sps=2e6, dwell_s=0.2,
                                 n_aircraft=2)
    assert plan2.revisit_time_s() == pytest.approx(plan.revisit_time_s() / 2)


def test_sweep_only_covers_the_warrants_bands(sortie):
    """The gate decides what may be RECORDED; the sweep decides what is even
    heard. Pointing the receiver outside the warrant's bands is not illegal,
    it is simply time not spent on the task."""
    from kestrel.ground.audit import AuditLog
    log = AuditLog(sortie.audit_path)
    plans = [e for e in log.entries() if e["type"] == "SWEEP_PLAN"]
    assert plans, "the sweep plan must be on the record"
    bands = plans[0]["payload"]["bands_hz"]
    assert bands == [[462.0e6, 468.0e6]]


# ------------------------------------------------- the two paths agree
def test_iq_path_uses_the_same_df_error_model_as_the_fast_path():
    """One BearingModel, two simulators. If they diverged, a geolocation
    result would depend on which simulator produced it."""
    sc = scenarios.get("border_crossing")
    sim = sc.build(seed=3)
    from kestrel.sim.simulated_df import SimulatedDf
    from kestrel.sim.world import BearingModel
    assert isinstance(sim.bearing, BearingModel)

    src = type("S", (), {"truth_for": lambda self, f, tol_hz=30e3: None})()
    df = SimulatedDf(sim, src, "K1")
    assert df.sim.bearing is sim.bearing
    assert df.estimate(_stub_detection()) is None
    assert df.unassociated == 1


def _stub_detection():
    from kestrel.common.messages import Detection
    return Detection(aircraft_id="K1", t_gpsdo_ns=0, freq_hz=462.5e6,
                     bandwidth_hz=12_500.0, snr_db=20.0,
                     aircraft_pos_xy=(0.0, 0.0))
