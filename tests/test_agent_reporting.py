"""When the agent publishes a fix, and when it stays quiet (Doc 06 §2).

Reporting used to be one threshold: ellipse below 20,000 m^2. That threshold
sits right at the systematic error floor (range x calibration residual), so on
the IQ path it published nothing for a whole sortie while holding 200 m fixes,
and on the fast path it published only tracks whose ellipses were small because
they were *wrong*. These tests pin the replacement behaviour.
"""
from __future__ import annotations

import numpy as np

from kestrel.air.agent import MissionAgent
from kestrel.common.messages import ErrorEllipse


class FakeTrack:
    """The minimum an agent needs to decide whether to report."""

    def __init__(self, area_m2: float, xy=(1000.0, 1000.0), track_id="E-0001"):
        self.track_id = track_id
        self.xy = np.asarray(xy, dtype=float)
        self.resolved = True
        self.n_observations = 12
        self.detections = ["d1", "d2"]
        self._area = area_m2

    def ellipse_area_m2(self) -> float:
        return self._area

    def error_ellipse(self) -> ErrorEllipse:
        # A circle of the right area, so semi_major is meaningful.
        r = float(np.sqrt(self._area / np.pi))
        return ErrorEllipse(semi_major_m=r, semi_minor_m=r, orientation_deg=0.0)


def _agent(**kw) -> MissionAgent:
    class _NullAudit:
        def append(self, *a, **k):
            return "h"

    return MissionAgent(aircraft_id="K1", planner=None, safety=None,
                        area=((0, 0), (1, 1)), controller=None,
                        track_store=None, classifier=None,
                        audit=_NullAudit(), **kw)


# ------------------------------------------------------------------ stall
def test_a_shrinking_ellipse_is_not_stalled():
    """While refinement is still buying accuracy, keep refining."""
    ag = _agent(stall_window_s=100.0, stall_improvement=0.10)
    for i, area in enumerate([1e6, 9e5, 8e5, 7e5, 6e5, 5e5]):
        ag.t = i * 25.0
        stalled = ag._refinement_stalled(FakeTrack(area))
    assert not stalled


def test_an_ellipse_that_stops_shrinking_triggers_a_report():
    """Below the systematic floor, loitering costs sortie time and buys
    nothing. The agent must notice and move on."""
    ag = _agent(stall_window_s=100.0, stall_improvement=0.10)
    stalled = False
    for i in range(10):
        ag.t = i * 25.0
        stalled = ag._refinement_stalled(FakeTrack(500_000.0))
    assert stalled


def test_stall_needs_a_full_window_of_history():
    """Two samples a second apart say nothing about a 120 s trend."""
    ag = _agent(stall_window_s=120.0)
    ag.t = 0.0
    assert not ag._refinement_stalled(FakeTrack(500_000.0))
    ag.t = 1.0
    assert not ag._refinement_stalled(FakeTrack(500_000.0))


# ------------------------------------------------------------ publication
def test_the_same_fix_is_not_republished():
    ag = _agent()
    track = FakeTrack(5e6)
    assert ag._publish(track) is not None
    assert ag._publish(track) is None
    assert len(ag.reports) == 1


def test_a_materially_better_fix_is_republished():
    ag = _agent()
    ag._publish(FakeTrack(5e6))
    assert ag._publish(FakeTrack(1e6)) is not None, "a 5x smaller ellipse is news"


def test_a_fix_that_moved_further_than_its_own_ellipse_is_republished():
    ag = _agent()
    ag._publish(FakeTrack(1e4, xy=(1000.0, 1000.0)))
    moved = FakeTrack(1e4, xy=(1000.0 + 500.0, 1000.0))   # r ~ 56 m
    assert ag._publish(moved) is not None


# --------------------------------------------------------------- quality
def test_quality_reflects_the_ellipse_not_just_maturity():
    """A well-observed unknown is still an unknown: 65 observations of a 5 km
    ellipse must not read as a good fix."""
    ag = _agent(target_ellipse_area_m2=20_000.0)
    huge = FakeTrack(8e7)
    huge.n_observations = 65
    assert ag._publish(huge).quality == "degraded"

    tight = FakeTrack(6_000.0, track_id="E-0002")
    assert ag._publish(tight).quality == "good"
