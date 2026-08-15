"""Scenario definitions (Doc 09 §4).

Scenarios are data so they are reproducible and shareable. Every scenario
includes DECOYS - legitimate emitters that must NOT be collected. A scenario
where everything is a target teaches nothing and tests nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kestrel.sim.world import Aircraft, Emitter, KestrelSim


@dataclass
class Scenario:
    name: str
    area: tuple
    duration_s: float
    emitters: list[dict]
    n_aircraft: int = 1
    noise_floor_dbm: float = -110.0
    warrant: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def build(self, seed: int = 0) -> KestrelSim:
        from kestrel.sim.mission import default_aircraft
        emitters = [Emitter(**e) for e in self.emitters]
        aircraft: list[Aircraft] = default_aircraft(self.n_aircraft, self.area)
        return KestrelSim(emitters, aircraft,
                          noise_floor_dbm=self.noise_floor_dbm, seed=seed)


AREA_8K = ((0.0, 0.0), (8000.0, 8000.0))
POLY_8K = [[0.0, 0.0], [8000.0, 0.0], [8000.0, 8000.0], [0.0, 8000.0]]


SCENARIOS: dict[str, Scenario] = {

    "border_crossing": Scenario(
        name="border_crossing",
        description="Two handhelds coordinating a night crossing, plus a satcom "
                    "terminal and a mobile tracker. One legitimate farm radio "
                    "shares the target channel and must not be collected.",
        area=AREA_8K, duration_s=1800.0, n_aircraft=1, noise_floor_dbm=-112.0,
        emitters=[
            dict(emitter_id="TGT-1", xy=(2100, 3400), freq_hz=462.5e6, mode="dmr",
                 eirp_dbm=37, burst_s=6, period_s=120),
            dict(emitter_id="TGT-2", xy=(2350, 3510), freq_hz=462.5e6, mode="dmr",
                 eirp_dbm=37, burst_s=4, period_s=120, phase_s=30),
            dict(emitter_id="SAT-1", xy=(5900, 1200), freq_hz=1626e6,
                 mode="satcom_terminal", eirp_dbm=45, burst_s=60, period_s=600),
            dict(emitter_id="TRK-1", xy=(4000, 4000), freq_hz=868e6, mode="lora",
                 eirp_dbm=14, burst_s=0.5, period_s=60, mobile=True, speed_mps=3),
            dict(emitter_id="DECOY-farm", xy=(1000, 7000), freq_hz=462.5e6,
                 mode="dmr", eirp_dbm=37, burst_s=8, period_s=180,
                 note="legitimate farm radio - lawful, must not be recorded"),
        ],
        warrant=dict(freq_ranges_hz=[[462.0e6, 468.0e6]],
                     signal_classes=["dmr", "fm_voice"],
                     area_polygon=POLY_8K),
    ),

    "decoy_heavy": Scenario(
        name="decoy_heavy",
        description="Mostly lawful traffic. Tests that the gate stays shut: only "
                    "the two authorized-band emitters may ever be recorded.",
        area=AREA_8K, duration_s=1200.0, n_aircraft=1,
        emitters=[
            dict(emitter_id="TGT-1", xy=(3000, 3000), freq_hz=463.0e6, mode="dmr",
                 eirp_dbm=37, burst_s=5, period_s=90),
            dict(emitter_id="TGT-2", xy=(5200, 2400), freq_hz=464.2e6,
                 mode="fm_voice", eirp_dbm=37, burst_s=5, period_s=110),
            dict(emitter_id="D-lte", xy=(1500, 1500), freq_hz=1842e6, mode="lte",
                 eirp_dbm=23, burst_s=30, period_s=40),
            dict(emitter_id="D-wifi", xy=(6000, 6000), freq_hz=2437e6, mode="wifi",
                 eirp_dbm=20, burst_s=20, period_s=25),
            dict(emitter_id="D-tetra", xy=(4200, 6600), freq_hz=390e6, mode="tetra",
                 eirp_dbm=35, burst_s=10, period_s=60),
            dict(emitter_id="D-lora", xy=(700, 5200), freq_hz=868e6, mode="lora",
                 eirp_dbm=14, burst_s=0.4, period_s=30),
            dict(emitter_id="D-bt", xy=(3900, 900), freq_hz=2440e6, mode="bluetooth",
                 eirp_dbm=4, burst_s=5, period_s=15),
        ],
        warrant=dict(freq_ranges_hz=[[462.0e6, 465.0e6]],
                     signal_classes=["dmr", "fm_voice"],
                     area_polygon=POLY_8K),
    ),

    "warrant_expiry_midflight": Scenario(
        name="warrant_expiry_midflight",
        description="Authority is revoked mid-sortie. Recording must stop "
                    "immediately; detection and reporting continue.",
        area=AREA_8K, duration_s=900.0, n_aircraft=1,
        emitters=[
            dict(emitter_id="TGT-1", xy=(2600, 2600), freq_hz=463.0e6, mode="dmr",
                 eirp_dbm=37, burst_s=8, period_s=45),
        ],
        warrant=dict(freq_ranges_hz=[[462.0e6, 465.0e6]],
                     signal_classes=["dmr"], area_polygon=POLY_8K),
    ),

    "single_emitter": Scenario(
        name="single_emitter",
        description="One steady emitter. Used to compare planners head to head.",
        area=AREA_8K, duration_s=900.0, n_aircraft=1,
        emitters=[
            dict(emitter_id="TGT-1", xy=(2500, 5500), freq_hz=463.5e6, mode="dmr",
                 eirp_dbm=37, burst_s=20, period_s=25),
        ],
        warrant=dict(freq_ranges_hz=[[462.0e6, 465.0e6]],
                     signal_classes=["dmr"], area_polygon=POLY_8K),
    ),

    "swarm_three": Scenario(
        name="swarm_three",
        description="Three aircraft, several emitters. Phase 2 groundwork.",
        area=AREA_8K, duration_s=1200.0, n_aircraft=3,
        emitters=[
            dict(emitter_id="TGT-1", xy=(2000, 6000), freq_hz=463.0e6, mode="dmr",
                 eirp_dbm=37, burst_s=6, period_s=60),
            dict(emitter_id="TGT-2", xy=(6200, 2200), freq_hz=464.0e6,
                 mode="fm_voice", eirp_dbm=37, burst_s=6, period_s=75),
            dict(emitter_id="D-lte", xy=(4000, 4000), freq_hz=1842e6, mode="lte",
                 eirp_dbm=23, burst_s=30, period_s=40),
        ],
        warrant=dict(freq_ranges_hz=[[462.0e6, 465.0e6]],
                     signal_classes=["dmr", "fm_voice"], area_polygon=POLY_8K),
    ),
}


def get(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'. Available: {sorted(SCENARIOS)}")
    return SCENARIOS[name]
