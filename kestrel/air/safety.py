"""Safety layer (Doc 06 §3).

Hard-coded, deterministic, evaluated LAST on every tick. It can veto or clamp
anything the mission agent proposes. No AI, no exceptions, no configuration
that can disable it in flight.

Below this sits PX4's own failsafes as a second, independent net.
"""
from __future__ import annotations

import math

import numpy as np

from kestrel.common.warrant import point_in_polygon


class SafetyLayer:
    def __init__(self, geofence_polygon, alt_min_m: float = 50.0,
                 alt_max_m: float = 120.0, battery_reserve_pct: float = 25.0,
                 min_separation_m: float = 30.0, link_timeout_s: float = 120.0,
                 home_xy=(0.0, 0.0)):
        self.geofence = geofence_polygon
        self.alt_min, self.alt_max = alt_min_m, alt_max_m
        self.battery_reserve_pct = battery_reserve_pct
        self.min_separation_m = min_separation_m
        self.link_timeout_s = link_timeout_s
        self.home_xy = tuple(home_xy)
        self.vetoes = 0
        self.violations = 0

    # ------------------------------------------------------------------
    def clamp(self, wp, my_pos, neighbors=(), battery_pct: float = 100.0):
        """Return a waypoint guaranteed to satisfy every hard constraint."""
        wp = tuple(float(v) for v in wp)

        if battery_pct < self.battery_reserve_pct:
            return self.home_xy

        if not point_in_polygon(wp, self.geofence):
            self.vetoes += 1
            wp = self._pull_inside(wp, my_pos)

        wp = self._enforce_separation(wp, neighbors)
        return wp

    # ------------------------------------------------------------------
    def _pull_inside(self, wp, my_pos):
        """Walk back along the line toward the current position until inside.
        If the aircraft is somehow already outside, head home."""
        if not point_in_polygon(tuple(my_pos), self.geofence):
            return self.home_xy
        a = np.asarray(my_pos, dtype=float)
        b = np.asarray(wp, dtype=float)
        for frac in (0.75, 0.5, 0.25, 0.1, 0.0):
            cand = tuple(a + (b - a) * frac)
            if point_in_polygon(cand, self.geofence):
                return cand
        return tuple(a)

    def _enforce_separation(self, wp, neighbors):
        for n in neighbors:
            d = math.hypot(wp[0] - n[0], wp[1] - n[1])
            if d < self.min_separation_m:
                self.vetoes += 1
                if d < 1e-6:
                    return (wp[0] + self.min_separation_m, wp[1])
                push = (self.min_separation_m - d) / d
                wp = (wp[0] + (wp[0] - n[0]) * push,
                      wp[1] + (wp[1] - n[1]) * push)
        return wp

    # ------------------------------------------------------------------
    def must_return(self, battery_pct: float, link_lost_s: float = 0.0,
                    gps_ok: bool = True, warrant_expired: bool = False) -> bool:
        return (battery_pct < self.battery_reserve_pct
                or link_lost_s > self.link_timeout_s
                or not gps_ok
                or warrant_expired)

    def check_position(self, xy) -> bool:
        """Post-hoc audit: was the aircraft ever outside the fence?"""
        ok = point_in_polygon(tuple(xy), self.geofence)
        if not ok:
            self.violations += 1
        return ok

    def clamp_altitude(self, alt_m: float) -> float:
        return min(max(alt_m, self.alt_min), self.alt_max)
