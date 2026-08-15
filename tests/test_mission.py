"""End-to-end mission tests (Doc 11 Step 6.5) - the Phase 1 acceptance tests."""
from __future__ import annotations

import numpy as np
import pytest

from kestrel.air.planner import (InfoGainPlanner, LawnmowerPlanner,
                                 fisher_information_aoa)
from kestrel.sim import scenarios
from kestrel.sim.metrics import assert_invariants, evaluate
from kestrel.sim.mission import run_mission
from kestrel.sim.run import run_one


# ------------------------------------------------------------- invariants
@pytest.mark.parametrize("name", sorted(scenarios.SCENARIOS))
def test_scenario_invariants_hold(name):
    _, m = run_one(name, seed=0, verbose=False)
    assert_invariants(m)


def test_no_over_collection_in_decoy_heavy():
    """Most traffic is lawful. Only the two authorized emitters may be captured."""
    result, m = run_one("decoy_heavy", seed=1, verbose=False)
    assert m["over_collection_count"] == 0
    for rec in result.recordings:
        assert result.token.covers(rec.freq_hz, rec.signal_class)
        assert 462.0e6 <= rec.freq_hz <= 465.0e6


def test_decoy_outside_warrant_band_is_detected_but_never_recorded():
    result, _ = run_one("decoy_heavy", seed=2, verbose=False)
    lte = [d for d in result.detections if d.truth_emitter_id == "D-lte"]
    assert lte, "the LTE decoy should be detected (metadata is always allowed)"
    assert all(not d.warrant_match for d in lte)
    assert all(d.recording_ref is None for d in lte)


def test_revocation_stops_recording_midflight():
    sc = scenarios.get("warrant_expiry_midflight")
    sim = sc.build(seed=0)
    result = run_mission(sim, duration_s=sc.duration_s, warrant_overrides=sc.warrant,
                         revoke_at_s=300.0, seed=0)
    assert result.token.revoked
    matched_after = [d for d in result.detections if d.warrant_match]
    # recordings exist from before revocation, and none after
    assert any("token_revoked" in d.warrant_reasons for d in result.detections)
    assert len(result.recordings) == len(matched_after)


def test_audit_chain_verifies_after_full_sortie():
    result, _ = run_one("border_crossing", seed=0, verbose=False)
    assert result.audit_ok


def test_every_recording_maps_to_an_authorized_detection():
    """The whole legal subsystem in one assertion."""
    result, _ = run_one("border_crossing", seed=0, verbose=False)
    from kestrel.ground.audit import AuditLog
    log = AuditLog(result.audit_path)
    detections = {d["detection_id"]: d for d in log.find_detections()}
    for rec in log.find_recordings():
        d = detections.get(rec["detection_ref"])
        assert d is not None, "recording with no corresponding detection"
        assert d["warrant_match"] is True
        assert result.token.covers(d["freq_hz"], d["signal_class"])


def test_safety_layer_keeps_aircraft_inside_geofence():
    result, m = run_one("border_crossing", seed=3, verbose=False)
    assert m["safety_violations"] == 0


def test_simulator_truth_never_reaches_the_audit_log():
    result, _ = run_one("border_crossing", seed=0, verbose=False)
    from kestrel.ground.audit import AuditLog
    for e in AuditLog(result.audit_path).entries():
        assert "truth_emitter_id" not in e["payload"]
        assert "truth_xy" not in e["payload"]


# ------------------------------------------------------------- capability
def test_detection_and_tracking_work():
    result, m = run_one("border_crossing", seed=0, verbose=False)
    assert m["n_detections"] > 100
    assert m["detection_rate"] >= 0.6
    assert m["n_tracks"] >= 2


def test_error_ellipses_are_honest():
    """The single most important accuracy test in the system (Doc 07 §6).

    A 95% ellipse must contain the truth about 95% of the time. If it does not,
    the reported uncertainty is a lie and every downstream decision - which
    fixes to act on, what goes in a report, what is defensible in court - is
    built on it.

    Applies to PUBLISHED fixes: tracks the system declares resolved. Tracks
    withheld as unresolved make no accuracy claim, so there is nothing to be
    honest or dishonest about.
    """
    contained = []
    for seed in range(20):
        result, _ = run_one("single_emitter", seed=seed, verbose=False)
        for t in result.tracks.values():
            if t.truth_xy is None or t.n_observations < 5:
                continue
            t._ensure_solved()
            if not t.resolved:
                continue
            contained.append(t.contains(t.truth_xy, 0.95))
    assert len(contained) >= 15, "not enough published fixes to judge containment"
    rate = float(np.mean(contained))
    assert rate >= 0.85, (
        f"ellipse containment only {rate:.0%} against a nominal 95%; the "
        "covariance is overconfident and the system is reporting fixes it "
        "cannot support")


def test_cochannel_channel_is_tracked_at_all():
    """Co-channel behaviour, asserted so regressions are visible.

    border_crossing puts three DMR emitters on 462.5 MHz: two targets ~250 m
    apart and a lawful farm radio 5 km away. The system reliably produces
    mature tracks on this channel and usually localizes the target pair.

    It does NOT reliably separate the three emitters - see the xfail below and
    docs/13-implementation-status.md. Note the operational consequence: the
    lawful farm radio sits inside the warrant's band and class, so its traffic
    IS collected. That is the warrant working as written, not a gate failure;
    separating lawful from unlawful users of an authorized channel is a
    minimization problem for analysts, and an argument for narrowing warrants.
    """
    localized = 0
    for seed in range(5):
        result, _ = run_one("border_crossing", seed=seed, verbose=False)
        mature = [t for t in result.tracks.values()
                  if abs(t.freq_hz - 462.5e6) < 12_500 and t.n_observations >= 40]
        assert mature, f"seed {seed}: no mature track on the target channel"
        for t in mature:
            t._ensure_solved()
        if any(t.resolved and np.linalg.norm(t.xy - np.array([2225.0, 3455.0])) < 800
               for t in mature):
            localized += 1
    assert localized >= 2, (
        f"target pair localized in only {localized}/5 seeds; co-channel "
        "tracking has regressed below its already-limited baseline")


@pytest.mark.xfail(reason=(
    "KNOWN LIMITATION - multi-emitter data association. In dense co-channel "
    "scenes, short early tracks converge to wrong positions and are published "
    "as resolved, so containment falls well below nominal. Needs proper "
    "multi-hypothesis association (JPDA/MHT) plus TDOA from the swarm. "
    "Tracked as an open item; see docs/13-implementation-status.md."),
    strict=True)
def test_multi_emitter_containment_matches_nominal():
    contained = []
    for seed in range(10):
        result, _ = run_one("border_crossing", seed=seed, verbose=False)
        for t in result.tracks.values():
            if t.truth_xy is None or t.n_observations < 5:
                continue
            t._ensure_solved()
            if t.resolved:
                contained.append(t.contains(t.truth_xy, 0.95))
    assert float(np.mean(contained)) >= 0.85


def test_info_gain_beats_lawnmower():
    """Signature capability #1, measured (Doc 06 §4).

    Measured on FIX ERROR, not ellipse area. Once the simulator includes
    systematic bearing bias (array calibration residual + airframe reflection,
    which is what real DF error is actually made of), the reported covariance
    is floored by that bias for every planner. Better geometry then shows up
    as a more accurate estimate rather than a smaller ellipse - and accuracy
    is the quantity that matters operationally.
    """
    sc = scenarios.get("single_emitter")
    ig, lm = [], []
    for seed in range(5):
        _, m1 = run_one(sc.name, seed=seed, verbose=False,
                        planner_factory=lambda: InfoGainPlanner())
        _, m2 = run_one(sc.name, seed=seed, verbose=False,
                        planner_factory=lambda: LawnmowerPlanner(sc.area))
        ig.append(m1["median_fix_error_m"])
        lm.append(m2["median_fix_error_m"])
    assert np.median(ig) < np.median(lm), (
        f"info-gain error {np.median(ig):,.0f} m vs lawnmower {np.median(lm):,.0f} m")


def test_single_aircraft_aoa_accuracy_is_physically_plausible():
    """Guards against the simulator flattering us (Doc 12 §7).

    Doc 07 §7 predicts 200-800 m for single-aircraft AOA. If the sim starts
    reporting metre-level accuracy from bearing-only DF, a systematic-error
    term has been lost and every downstream conclusion is worthless.
    """
    errs = [run_one("single_emitter", seed=s, verbose=False)[1]["median_fix_error_m"]
            for s in range(5)]
    median = float(np.median(errs))
    # Lower bound guards the simulator: with correlated GPS bias, systematic
    # bearing error and multipath all modelled, no amount of averaging should
    # reach sub-metre. If it does, an error source has been dropped.
    # The upper bound is generous because the info-gain planner deliberately
    # flies a wide orbit, which legitimately beats the Doc 07 §7 figures for
    # conventional single-aircraft AOA - that is the capability working.
    assert 1.0 < median < 2000.0, (
        f"median AOA fix error {median:,.1f} m is outside the plausible range; "
        "the simulator's error model is probably wrong")


# ---------------------------------------------------------------- planner
def test_fisher_information_rewards_geometric_diversity():
    """Two bearings from the same direction carry far less information than two
    from perpendicular directions. This is the premise of the whole planner."""
    emitter = (0.0, 0.0)
    sigma = np.radians(3.0)
    collinear = fisher_information_aoa(emitter, [(-1000, 0), (-1100, 0)], sigma)
    spread = fisher_information_aoa(emitter, [(-1000, 0), (0, -1000)], sigma)
    assert np.linalg.det(spread) > np.linalg.det(collinear) * 100


def test_swarm_scenario_runs_with_three_aircraft():
    result, m = run_one("swarm_three", seed=0, verbose=False)
    ids = {d.aircraft_id for d in result.detections}
    assert ids == {"K1", "K2", "K3"}
    assert_invariants(m)
