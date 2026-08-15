"""Information-Gain Flight (Doc 06 §4) - signature capability #1.

Conventional airborne DF flies a search grid and hopes the geometry works out.
Geolocation accuracy, though, depends enormously on WHERE you observe from: two
bearings taken from nearly the same direction produce a long thin ellipse that
is nearly useless, while moving a few hundred metres sideways can shrink it by
an order of magnitude.

So treat flight as experiment design. At each tick, score candidate next
positions by the predicted reduction in the uncertainty of the current best
track (Fisher information, D-optimality) and fly toward the best one.

The whole thing is 2x2 matrix algebra: microseconds of CPU, no GPU, no training,
and fully explainable to a reviewer.
"""
from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------
def fisher_information_aoa(emitter_xy, obs_positions, sigma_rad: float) -> np.ndarray:
    """Fisher Information Matrix for a set of bearing observations.

    Larger determinant = more information = smaller achievable error ellipse.
    """
    emitter_xy = np.asarray(emitter_xy, dtype=float)
    J = np.zeros((2, 2))
    for p in obs_positions:
        p = np.asarray(p, dtype=float)
        dx, dy = emitter_xy[0] - p[0], emitter_xy[1] - p[1]
        r2 = dx * dx + dy * dy
        if r2 < 1.0:
            continue
        g = np.array([-dy / r2, dx / r2])       # d(bearing)/d(emitter position)
        J += np.outer(g, g) / (sigma_rad ** 2)
    return J


def d_optimality(J: np.ndarray) -> float:
    """log-det of the information matrix, guarded against singularity."""
    det = float(np.linalg.det(J))
    return math.log(det) if det > 1e-12 else math.log(1e-12)


def expected_info_gain(track_xy, my_pos, candidate_pos, others_pos,
                       sigma_rad: float) -> float:
    """How much would moving to candidate_pos improve this fix?"""
    before = fisher_information_aoa(track_xy, list(others_pos) + [my_pos], sigma_rad)
    after = fisher_information_aoa(track_xy, list(others_pos) + [candidate_pos], sigma_rad)
    return d_optimality(after) - d_optimality(before)


# --------------------------------------------------------------------------
class InfoGainPlanner:
    """Ring search around the current position. Deterministic and cheap."""

    name = "infogain"

    def __init__(self, radius_m: float = 400.0, n_candidates: int = 16,
                 cost_weight: float = 0.002, default_sigma_deg: float = 3.0,
                 max_history: int = 40):
        self.radius_m = radius_m
        self.n_candidates = n_candidates
        self.cost_weight = cost_weight
        self.default_sigma_rad = math.radians(default_sigma_deg)
        # Cap the history so a long-lived track cannot make every candidate
        # look equally worthless (and to bound per-tick cost).
        self.max_history = max_history

    def refine_waypoint(self, track, my_pos, others_pos=()) -> tuple[float, float]:
        """Choose the next position that most improves THIS track.

        The baseline is every observation already taken (track.obs_positions),
        not merely the current position: information has diminishing returns,
        and a planner blind to what it already measured just oscillates between
        two directions forever.
        """
        my_pos = np.asarray(my_pos, dtype=float)

        history = list(getattr(track, "obs_positions", []))[-self.max_history:]
        baseline = list(history) + list(others_pos)
        J_before = fisher_information_aoa(track.xy, baseline, self.default_sigma_rad)
        base_score = d_optimality(J_before)

        best, best_score = tuple(my_pos), -math.inf
        for th in np.linspace(0, 2 * math.pi, self.n_candidates, endpoint=False):
            cand = (float(my_pos[0] + self.radius_m * math.cos(th)),
                    float(my_pos[1] + self.radius_m * math.sin(th)))
            J_after = fisher_information_aoa(track.xy, baseline + [cand],
                                             self.default_sigma_rad)
            gain = d_optimality(J_after) - base_score
            travel = self.cost_weight * math.hypot(cand[0] - my_pos[0],
                                                   cand[1] - my_pos[1])
            score = gain - travel          # information per unit battery
            if score > best_score:
                best, best_score = cand, score
        return best

    # Survey behaviour when there is nothing to refine yet.
    def survey_waypoint(self, my_pos, area, rng, sector=None) -> tuple[float, float]:
        return _random_survey(my_pos, area, rng)


class LawnmowerPlanner:
    """Baseline to beat: a fixed boustrophedon grid, ignoring what it hears."""

    name = "lawnmower"

    def __init__(self, area, lane_spacing_m: float = 800.0):
        (self.x0, self.y0), (self.x1, self.y1) = area
        self.lane_spacing = lane_spacing_m
        self._waypoints = self._build()
        self._i = 0

    def _build(self):
        wps, y, going_right = [], self.y0 + self.lane_spacing / 2, True
        while y < self.y1:
            xs = [self.x0 + 200, self.x1 - 200]
            if not going_right:
                xs.reverse()
            wps += [(xs[0], y), (xs[1], y)]
            y += self.lane_spacing
            going_right = not going_right
        return wps

    def refine_waypoint(self, track, my_pos, others_pos=()):
        return self.survey_waypoint(my_pos, None, None)

    def survey_waypoint(self, my_pos, area, rng, sector=None):
        wp = self._waypoints[self._i % len(self._waypoints)]
        if math.hypot(wp[0] - my_pos[0], wp[1] - my_pos[1]) < 100.0:
            self._i += 1
            wp = self._waypoints[self._i % len(self._waypoints)]
        return wp


class RandomPlanner:
    """Control condition. If a planner cannot beat this, it is not working."""

    name = "random"

    def __init__(self, area):
        self.area = area

    def refine_waypoint(self, track, my_pos, others_pos=()):
        return self.survey_waypoint(my_pos, self.area, np.random.default_rng())

    def survey_waypoint(self, my_pos, area, rng, sector=None):
        rng = rng or np.random.default_rng()
        (x0, y0), (x1, y1) = area or self.area
        return (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))


def _random_survey(my_pos, area, rng):
    (x0, y0), (x1, y1) = area
    rng = rng or np.random.default_rng()
    return (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
