"""The airborne RF sensor: a radio, a sweep, and the DSP chain (Doc 04 §7).

This is the piece that turns "there is a receiver on the aircraft" into a
stream of Detections. It owns the one resource the aircraft genuinely cannot
cheat on - *time on frequency*. A 20 MHz receiver cannot watch 400 MHz of
spectrum at once, so every second is spent choosing what not to listen to, and
an emitter that keys up while the sweep is elsewhere is simply not heard.

Everything here is flight code. The two injected pieces are what change on real
hardware:

  * `source`  - `SimulatedSdrSource` now, `UhdSdrSource` at Step 8.
  * `df`      - the direction finder. Bearings come from an interferometer on
                multi-channel IQ; `iq_synth` is single-channel, so in
                simulation this is `kestrel.sim.simulated_df.SimulatedDf`.
                Injected rather than assumed, so the swap is a constructor
                argument and not a rewrite.
"""
from __future__ import annotations

import numpy as np

from kestrel.common.messages import Detection


class RfSensor:
    def __init__(self, source, pipeline, sweep_plan, sector=None, df=None,
                 max_frame_samples: int = 262_144,
                 max_frames_per_step: int | None = None):
        self.source = source
        self.pipeline = pipeline
        self.sweep = sweep_plan
        self.sector = sector or sweep_plan.segments
        self.df = df
        self.max_frame_samples = max_frame_samples
        # None = sweep continuously, which is what the radio actually does. A
        # cap trades coverage for wall-clock time in long simulations; the
        # spectrum it skips is counted in `dwell_s_unwatched` rather than
        # forgotten, so a capped run cannot be mistaken for a full one.
        self.max_frames_per_step = max_frames_per_step

        # Sweep bookkeeping. `dwell_deficit_s` is the honest accounting: if the
        # frame budget stops us short of what dt allowed, the time is recorded
        # as spectrum that went unwatched rather than quietly forgotten.
        self._i = 0
        self.frames = 0
        self.dwell_s_used = 0.0
        self.dwell_deficit_s = 0.0
        self.signal_s_examined = 0.0
        self.detections_out = 0
        self.bearings_out = 0

    # ------------------------------------------------------------------
    @property
    def revisit_time_s(self) -> float:
        """How long between successive looks at any one segment. An emitter
        that transmits for 4 s every 10 minutes is invisible if this exceeds
        its burst length by enough - this is the number that decides what the
        sortie can and cannot find."""
        return sum(s.dwell_s for s in self.sector)

    def next_segment(self):
        seg = self.sector[self._i % len(self.sector)]
        self._i += 1
        return seg

    # ------------------------------------------------------------------
    def sense(self, dt: float, aircraft_pos_xy, alt_m: float = 120.0,
              on_detection=None) -> list[tuple[Detection, np.ndarray]]:
        """Spend `dt` seconds of receiver time and return what was heard.

        Each returned Detection carries the narrowband IQ snippet of exactly
        that signal - never the wideband stream, which would amount to
        recording everyone on the air. What happens to the snippet is the
        warrant gate's decision, not this module's.

        `on_detection(det, snippet)` is called per detection while the receiver
        is still tuned to the frame that produced it. In simulation that is
        where ground-truth annotation happens; in flight it is unused. It runs
        after the pipeline has committed, so it cannot influence what is found.
        """
        budget = float(dt)
        out: list[tuple[Detection, np.ndarray]] = []
        frames_this_step = 0

        cap = self.max_frames_per_step
        while budget > 0 and (cap is None or frames_this_step < cap):
            seg = self.next_segment()
            if seg.dwell_s > budget:
                break
            budget -= seg.dwell_s
            self.dwell_s_used += seg.dwell_s

            self.source.tune(seg.center_hz, seg.rate_sps)
            # A full dwell at 2 Msps is 400k complex samples. Capping the frame
            # keeps memory and synthesis cost bounded, at the price of looking
            # at only part of the dwell - so the fraction actually examined is
            # reported (`frame_coverage`) instead of being assumed to be 1.0.
            n = min(int(seg.rate_sps * seg.dwell_s), self.max_frame_samples)
            n = max(n, 4096)
            iq, t_ns = self.source.frame(n)
            self.frames += 1
            frames_this_step += 1
            self.signal_s_examined += n / seg.rate_sps

            for det, snip in self.pipeline.process_frame(
                    iq, seg, t_ns, aircraft_pos_xy, aircraft_alt_m=alt_m):
                if self.df is not None:
                    b = self.df.estimate(det, snip)
                    if b is not None:
                        det.bearing_deg, det.bearing_sigma_deg = b
                        self.bearings_out += 1
                if on_detection is not None:
                    on_detection(det, snip)
                out.append((det, snip))

        # Whatever dt remained that we could not spend is spectrum nobody
        # watched. Counting it keeps the simulation from claiming coverage the
        # radio never had.
        self.dwell_deficit_s += max(budget, 0.0)
        self.detections_out += len(out)
        return out

    # ------------------------------------------------------------------
    def health(self) -> dict:
        h = dict(self.pipeline.health())
        h.update({
            "sweep_frames": self.frames,
            "dwell_s_used": self.dwell_s_used,
            "dwell_s_unwatched": self.dwell_deficit_s,
            "frame_coverage": (self.signal_s_examined
                               / max(self.dwell_s_used, 1e-9)),
            "revisit_time_s": self.revisit_time_s,
            "detections": self.detections_out,
            "bearings": self.bearings_out,
        })
        return h
