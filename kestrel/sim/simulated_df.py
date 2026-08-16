"""Simulated direction finder (Doc 07 §2) - the last shortcut in the IQ path.

The real bearing comes from the phase difference between antenna elements. Our
IQ synthesizer produces one channel, so there is no phase front to measure and
no honest way to derive a bearing from the samples themselves. Rather than
pretend, this module says plainly what it is: geometry plus the measured error
model from `kestrel.sim.world.BearingModel`, the same one the detection-level
simulator uses.

Two properties keep it from flattering the system:

  * It uses the SNR the *detector measured*, not the link budget - so a weak
    or partially-masked signal yields a correspondingly poor bearing.
  * A detection it cannot associate with any emitter (a false alarm, an image,
    a splatter product) gets **no bearing at all**, and a bearingless detection
    contributes nothing to a track. The DSP's mistakes therefore stay
    mistakes; they are not quietly repaired by ground truth.

Replacing this is Step 8: multi-channel synthesis, then real interferometry.
Nothing above it changes when that happens - `RfSensor` takes the DF as a
constructor argument.
"""
from __future__ import annotations

import math

import numpy as np

C = 299_792_458.0


class SimulatedDf:
    def __init__(self, sim, source, aircraft_id: str, assoc_tol_hz: float = 30e3):
        self.sim = sim
        self.source = source
        self.aircraft_id = aircraft_id
        self.assoc_tol_hz = assoc_tol_hz
        self.unassociated = 0

    # ------------------------------------------------------------------
    def _aircraft(self):
        for a in self.sim.aircraft:
            if a.aircraft_id == self.aircraft_id:
                return a
        raise KeyError(self.aircraft_id)

    def estimate(self, det, iq: np.ndarray | None = None):
        """Bearing and sigma for a detection, or None if it is not real."""
        truth = self.source.truth_for(det.freq_hz, tol_hz=self.assoc_tol_hz)
        if truth is None:
            self.unassociated += 1
            return None
        a = self._aircraft()
        return self.sim.bearing.measure(self.aircraft_id, a.xy,
                                        truth["emitter"].xy, det.snr_db)

    # ------------------------------------------------------------------
    def annotate(self, det, iq: np.ndarray | None = None) -> None:
        """Stamp arrival time and tag ground truth, for metrics ONLY.

        This MUST run while the receiver is still tuned to the frame that
        produced the detection: `truth_for` only sees emitters inside the
        current window, so annotating after the sweep has moved on silently
        loses every truth tag. `RfSensor` calls this per detection, in the
        frame loop, for exactly that reason.

        `AuditLog.record_detection` strips the truth fields via
        `Detection.sanitized()`, and `tests/test_mission.py` asserts nothing
        truth-bearing survives into a report. On real hardware they stay None.
        """
        truth = self.source.truth_for(det.freq_hz, tol_hz=self.assoc_tol_hz)
        a = self._aircraft()

        # The frame timestamp is the moment the RECEIVER sampled; what matters
        # for TDOA is the moment the wave arrived, on a clock that is not
        # perfect. Wrong by 100 ns is wrong by 30 m.
        slant = float(truth["slant_m"]) if truth is not None else 0.0
        det.t_gpsdo_ns = int(det.t_gpsdo_ns + slant / C * 1e9 + a.clock_error_ns)

        if truth is None:
            return
        e = truth["emitter"]
        det.truth_emitter_id = e.emitter_id
        det.truth_class = e.mode
        det.truth_xy = (float(e.xy[0]), float(e.xy[1]))
