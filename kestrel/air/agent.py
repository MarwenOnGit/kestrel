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
                 stall_window_s: float = 120.0,
                 stall_improvement: float = 0.10,
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
        self.stall_window_s = stall_window_s
        self.stall_improvement = stall_improvement
        self._area_history: list[tuple[float, float]] = []
        self._published: dict[str, tuple[float, tuple[float, float]]] = {}

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
             neighbors=(), now: float | None = None,
             iq_by_detection: dict | None = None):
        """`iq_by_detection` maps detection_id -> the narrowband snippet that
        produced it. Present when the sensor is the real DSP chain; absent when
        detections come from the detection-level simulator, in which case there
        is no IQ to record and the placeholder below is used."""
        self.t += dt
        self.state_since_s += dt

        # ---- perceive -------------------------------------------------
        iq_by_detection = iq_by_detection or {}
        for det in detections:
            if det.aircraft_id != self.aircraft_id:
                continue
            self._process(det, my_pos, now,
                          iq=iq_by_detection.get(det.detection_id))

        # ---- decide ---------------------------------------------------
        if self.safety.must_return(battery_pct,
                                   warrant_expired=self.controller.gate.expired(now)):
            self.state = RETURN

        if self.state == SURVEY:
            wp = self.planner.survey_waypoint(my_pos, self.area, self.rng)
            best = self.tracks.best()
            if best is not None:
                self.current_track = best
                self._area_history.clear()
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
                if t.resolved and (t.ellipse_area_m2() < self.target_area
                                   or self._refinement_stalled(t)):
                    self._set_state(REPORT)

        elif self.state == REPORT:
            t = self.current_track
            if t is not None:
                self._publish(t)
            self.current_track = None
            self._area_history.clear()
            self._set_state(SURVEY)
            wp = self.planner.survey_waypoint(my_pos, self.area, self.rng)

        else:                                   # RETURN
            wp = self.safety.home_xy

        # ---- act (safety ALWAYS last) --------------------------------
        self.waypoint = self.safety.clamp(wp, my_pos, neighbors, battery_pct)
        return self.waypoint

    # ------------------------------------------------------------------
    def _refinement_stalled(self, track) -> bool:
        """Has the ellipse stopped shrinking?

        A fixed target area assumes accuracy is limited by how long you loiter.
        It is not: below `range x calibration_sigma` the error is systematic,
        and no amount of extra geometry touches it (docs/13 §3.5). Against that
        floor a fixed threshold either publishes nothing at all - which is what
        the IQ path did, while holding 200 m fixes it never reported - or gets
        lowered until over-confident tracks slip through.

        So the trigger is diminishing returns: once `stall_window_s` of
        refinement has bought less than `stall_improvement` of the ellipse,
        loitering is spending sortie time for nothing. Report and move on.
        """
        area = track.ellipse_area_m2()
        self._area_history.append((self.t, area))
        cutoff = self.t - self.stall_window_s
        while len(self._area_history) > 2 and self._area_history[0][0] < cutoff:
            self._area_history.pop(0)

        if self._area_history[0][0] > cutoff:
            return False                      # not enough history yet
        oldest = self._area_history[0][1]
        if oldest <= 0:
            return False
        return (oldest - area) / oldest < self.stall_improvement

    # ------------------------------------------------------------------
    def _process(self, det, my_pos, now, iq=None):
        """Classify, offer to the gate, then update the belief state.

        Order matters: classification happens BEFORE the gate (so uncertainty
        can deny), and tracking happens regardless of the gate decision (we may
        always count and locate; we may not always record).
        """
        from kestrel.air.classifier import apply_classification
        apply_classification(det, self.classifier, iq)

        if iq is None:
            # Detection-level simulator: there is no waveform to seal. The
            # sealing path is still exercised end to end, with empty contents.
            iq = np.zeros(256, dtype=np.complex64)
            duration_s = 1.0
        else:
            # Charge the gate for the signal time actually held, measured from
            # the snippet itself (Doc 08 §3): a warrant budget spent on
            # guessed durations is not a budget.
            duration_s = det.capture_duration_s or 1.0
        self.controller.offer(det, iq, my_pos, duration_s=duration_s, now=now)

        self.audit.record_detection(det)
        self.tracks.update(det)

    def _worth_publishing(self, track) -> bool:
        """Is this fix news?

        Because REPORT returns to SURVEY and SURVEY re-selects the *most
        uncertain* track, an emitter the sortie cannot pin down is exactly the
        one the agent keeps coming back to - so without this it republishes the
        same coordinates every couple of minutes. An operator reading ten
        identical fixes has no way to tell repetition from confirmation.
        """
        prev = self._published.get(track.track_id)
        if prev is None:
            return True
        prev_area, prev_xy = prev
        area = track.ellipse_area_m2()
        moved = float(np.linalg.norm(np.asarray(prev_xy) - track.xy))
        return area < 0.5 * prev_area or moved > track.error_ellipse().semi_major_m

    def _publish(self, track):
        from kestrel.common.messages import GeoFix
        if not self._worth_publishing(track):
            return None

        ellipse = track.error_ellipse()
        fix = GeoFix(
            emitter_id=track.track_id,
            xy=(float(track.xy[0]), float(track.xy[1])),
            error_ellipse=ellipse,
            method="aoa",
            n_aircraft=1,
            n_observations=track.n_observations,
            # Quality must reflect the ELLIPSE, not just maturity. A track with
            # 65 observations and a 5 km ellipse is a well-observed unknown -
            # calling it "good" because it is old invites an operator to act on
            # a position the system does not actually have.
            quality=("good" if (track.resolved and track.n_observations >= 5
                                and ellipse.area_m2() < self.target_area)
                     else "degraded"),
            contributing_detections=list(track.detections),
        )
        self._published[track.track_id] = (ellipse.area_m2(),
                                           tuple(float(v) for v in track.xy))
        self.reports.append(fix)
        self.audit.append("GEOFIX", fix.to_dict())
        return fix
