"""IQ sources (Doc 04 §2).

One interface, two implementations. Everything downstream of `frames()` is
identical whether the samples come from a USRP or from the scene composer,
which is what lets the whole pipeline be developed and regression-tested with
no radio attached.
"""
from __future__ import annotations

import math

import numpy as np

from kestrel.sim import iq_synth
from kestrel.sim.world import C, path_loss_db


class SdrSourceBase:
    def frames(self, n_samples: int):
        raise NotImplementedError

    def tune(self, center_hz: float, rate_sps: float) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------
class SimulatedSdrSource(SdrSourceBase):
    """Composes a wideband IQ frame from the world model.

    Applies real link-budget physics - path loss, altitude slant range, the
    detection floor - then hands over raw samples. The pipeline is told nothing
    about what is present and must find it, which is the whole point.
    """

    def __init__(self, sim, aircraft_id: str, noise_floor_dbm: float | None = None,
                 seed: int = 0):
        self.sim = sim
        self.aircraft_id = aircraft_id
        self.noise_floor_dbm = (noise_floor_dbm if noise_floor_dbm is not None
                                else sim.noise_floor_dbm)
        self.rng = np.random.default_rng(seed)
        self.center_hz = 462.5e6
        self.rate_sps = 1e6

    def tune(self, center_hz: float, rate_sps: float) -> None:
        self.center_hz, self.rate_sps = center_hz, rate_sps

    # ------------------------------------------------------------------
    def _aircraft(self):
        for a in self.sim.aircraft:
            if a.aircraft_id == self.aircraft_id:
                return a
        raise KeyError(self.aircraft_id)

    def visible_signals(self) -> list[dict]:
        """Which emitters are audible right now, and at what SNR."""
        a = self._aircraft()
        out = []
        for e in self.sim.emitters:
            if not e.is_transmitting(self.sim.t):
                continue
            if abs(e.freq_hz - self.center_hz) > self.rate_sps / 2:
                continue
            d = float(np.linalg.norm(e.xy - a.xy))
            slant = math.sqrt(d * d + a.alt_m * a.alt_m)
            snr = (e.eirp_dbm - path_loss_db(slant, e.freq_hz, h_rx_m=a.alt_m)
                   - self.noise_floor_dbm)
            if snr < -10.0:
                continue                      # far below anything detectable
            out.append({
                "freq_hz": e.freq_hz,
                "signal_class": e.mode,
                "snr_db": snr,
                "bandwidth_hz": self.sim._bandwidth_for(e.mode),
                "emitter": e,
                "slant_m": slant,
            })
        return out

    def frame(self, n_samples: int = 262_144) -> tuple[np.ndarray, int]:
        signals = self.visible_signals()
        iq = iq_synth.compose_scene(n_samples, self.rate_sps, self.center_hz,
                                    signals, rng=self.rng)
        t_ns = int(self.sim.t * 1e9)
        return iq, t_ns

    def truth_for(self, freq_hz: float, tol_hz: float = 50e3):
        """Ground truth lookup for metrics ONLY. Never used by the pipeline."""
        best = None
        for s in self.visible_signals():
            d = abs(s["freq_hz"] - freq_hz)
            if d <= tol_hz and (best is None or d < best[0]):
                best = (d, s)
        return best[1] if best else None


# --------------------------------------------------------------------------
class UhdSdrSource(SdrSourceBase):
    """Real USRP via UHD. Implemented at Step 8 (hardware bring-up).

    Kept as a stub with the intended shape so the interface stays honest:
    overflow handling and GPSDO timestamps are not optional extras, they are
    the two things that make airborne IQ usable at all.
    """

    def __init__(self, args: str = "type=x4xx", clock_source: str = "gpsdo"):
        raise NotImplementedError(
            "Step 8 of docs/11-phase1-implementation-plan.md. Requires UHD and "
            "a USRP. Must: set clock/time source to GPSDO, verify ref_locked "
            "and gps_locked, stamp every frame with GPSDO time, and treat any "
            "overflow as a hole in time that invalidates TDOA.")
