"""Autonomous mission agent (Doc 06 §2).

A five-state behaviour tree that orchestrates specialists. Deliberately NOT a
learned policy at this level: a reviewer must be able to reason about every
transition, and mistakes here are the expensive kind.

The agent emits waypoints only. It never commands motors, and it never decides
what may be recorded - that is the warrant gate's job alone.
"""
from __future__ import annotations

import numpy as np

SURVEY, CONTACT, REFINE, REPORT, RETURN = "SURVEY", "CONTACT", "REFINE", "REPORT", "RETURN"


class MissionAgent:
    def __init__(self, aircraft_id, planner, safety, area, controller,
                 track_store, classifier, audit,
                 target_ellipse_area_m2: float = 20_000.0,
                 contact_timeout_s: float = 60.0,
                 min_state_dwell_s: float = 3.0,
                 seed: int = 0):
        self.aircraft_id = aircraft_id
        self.planner = planner
        self.safety = safety
        self.area = area
        self.controller = controller       # gate + sealer, the ONLY write path
        self.tracks = track_store
        self.classifier = classifier
        self.audit = audit

        self.state = SURVEY
        self.state_since_s = 0.0
        self.min_dwell = min_state_dwell_s
        self.target_area = target_ellipse_area_m2
        self.contact_timeout_s = contact_timeout_s

        self.rng = np.random.default_rng(seed)
        self.current_track = None
        self.waypoint = None
        self.reports: list = []
        self.t = 0.0

    # ------------------------------------------------------------------
    def _set_state(self, new_state: str) -> None:
        """Hysteresis: a minimum dwell prevents the oscillation that multi-agent
        systems fall into (Doc 12 §6.5)."""
        if new_state != self.state and self.state_since_s >= self.min_dwell:
            self.audit.append("AGENT_STATE", {
                "aircraft_id": self.aircraft_id, "from": self.state,
                "to": new_state, "t": self.t})
            self.state = new_state
            self.state_since_s = 0.0

    # ------------------------------------------------------------------
    def tick(self, detections, my_pos, battery_pct, dt: float,
             neighbors=(), now: float | None = None):
        self.t += dt
        self.state_since_s += dt

        # ---- perceive -------------------------------------------------
        for det in detections:
            if det.aircraft_id != self.aircraft_id:
                continue
            self._process(det, my_pos, now)

        # ---- decide ---------------------------------------------------
        if self.safety.must_return(battery_pct,
                                   warrant_expired=self.controller.gate.expired(now)):
            self.state = RETURN

        if self.state == SURVEY:
            wp = self.planner.survey_waypoint(my_pos, self.area, self.rng)
            best = self.tracks.best()
            if best is not None:
                self.current_track = best
                self._set_state(CONTACT)

        elif self.state == CONTACT:
            t = self.current_track
            if t is None:
                self._set_state(SURVEY)
                wp = self.planner.survey_waypoint(my_pos, self.area, self.rng)
            else:
                wp = self.planner.refine_waypoint(t, my_pos, neighbors)
                if t.n_observations >= 3:
                    self._set_state(REFINE)

        elif self.state == REFINE:
            t = self.current_track
            if t is None:
                self._set_state(SURVEY)
                wp = self.planner.survey_waypoint(my_pos, self.area, self.rng)
            else:
                wp = self.planner.refine_waypoint(t, my_pos, neighbors)
                if t.resolved and t.ellipse_area_m2() < self.target_area:
                    self._set_state(REPORT)

        elif self.state == REPORT:
            t = self.current_track
            if t is not None:
                self._publish(t)
            self.current_track = None
            self._set_state(SURVEY)
            wp = self.planner.survey_waypoint(my_pos, self.area, self.rng)

        else:                                   # RETURN
            wp = self.safety.home_xy

        # ---- act (safety ALWAYS last) --------------------------------
        self.waypoint = self.safety.clamp(wp, my_pos, neighbors, battery_pct)
        return self.waypoint

    # ------------------------------------------------------------------
    def _process(self, det, my_pos, now):
        """Classify, offer to the gate, then update the belief state.

        Order matters: classification happens BEFORE the gate (so uncertainty
        can deny), and tracking happens regardless of the gate decision (we may
        always count and locate; we may not always record).
        """
        from kestrel.air.classifier import apply_classification
        apply_classification(det, self.classifier)

        iq = np.zeros(256, dtype=np.complex64)   # placeholder until Step 3/8
        self.controller.offer(det, iq, my_pos, duration_s=1.0, now=now)

        self.audit.record_detection(det)
        self.tracks.update(det)

    def _publish(self, track):
        from kestrel.common.messages import GeoFix
        fix = GeoFix(
            emitter_id=track.track_id,
            xy=(float(track.xy[0]), float(track.xy[1])),
            error_ellipse=track.error_ellipse(),
            method="aoa",
            n_aircraft=1,
            n_observations=track.n_observations,
            quality=("good" if (track.resolved and track.n_observations >= 5)
                     else "degraded"),
            contributing_detections=list(track.detections),
        )
        self.reports.append(fix)
        self.audit.append("GEOFIX", fix.to_dict())
        return fix
