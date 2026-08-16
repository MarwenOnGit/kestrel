"""CLI runner (Doc 09 §8).

    python3 -m kestrel.sim.run --scenario border_crossing
    python3 -m kestrel.sim.run --all-scenarios --seeds 5 --assert-metrics
    python3 -m kestrel.sim.run --compare --seeds 10
    python3 -m kestrel.sim.run --scenario border_crossing --iq
"""
from __future__ import annotations

import argparse
import statistics
import sys

from kestrel.sim import scenarios
from kestrel.sim.metrics import assert_invariants, evaluate, format_metrics
from kestrel.sim.mission import run_mission


def run_one(name: str, seed: int = 0, planner_factory=None, verbose: bool = True,
            iq: bool = False, **kwargs):
    sc = scenarios.get(name)
    sim = sc.build(seed=seed)
    revoke = 400.0 if name == "warrant_expiry_midflight" else None
    result = run_mission(sim, duration_s=sc.duration_s,
                         warrant_overrides=sc.warrant,
                         planner_factory=planner_factory,
                         seed=seed, revoke_at_s=revoke, iq_mode=iq, **kwargs)
    m = evaluate(result)
    if verbose:
        print(f"\n=== {sc.name}  (seed {seed}{', IQ' if iq else ''}) ===")
        print(f"  {sc.description}")
        print(format_metrics(m))
        for acid, h in result.sensor_health.items():
            watched = h["dwell_s_used"] / max(result.duration_s, 1e-9)
            print(f"  sensor {acid}: {h['sweep_frames']} frames, "
                  f"revisit {h['revisit_time_s']:.2f} s, "
                  f"{watched:.0%} of the sortie on frequency, "
                  f"{h['detections']} detections "
                  f"({h['candidates_per_frame']:.3f}/frame)")
    return result, m


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kestrel.sim.run")
    p.add_argument("--scenario", default="border_crossing")
    p.add_argument("--all-scenarios", action="store_true")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--assert-metrics", action="store_true")
    p.add_argument("--compare", action="store_true",
                   help="compare planners on the single_emitter scenario")
    p.add_argument("--iq", action="store_true",
                   help="fly on real IQ: sweep, synthesize, and run the DSP "
                        "chain instead of generating detections from geometry "
                        "(~100x slower, and finds less - as a radio would)")
    p.add_argument("--max-frames-per-step", type=int, default=None,
                   help="cap frames per simulated second; the unwatched "
                        "spectrum is reported rather than ignored")
    args = p.parse_args(argv)

    if args.compare:
        return _compare(args.seeds)

    names = sorted(scenarios.SCENARIOS) if args.all_scenarios else [args.scenario]
    failures = 0
    extra = ({"max_frames_per_step": args.max_frames_per_step}
             if args.iq else {})
    for name in names:
        for seed in range(args.seeds):
            _, m = run_one(name, seed=seed, verbose=(args.seeds <= 3),
                           iq=args.iq, **extra)
            if args.assert_metrics:
                try:
                    assert_invariants(m)
                except AssertionError as e:
                    print(f"  !! {name} seed {seed}: {e}")
                    failures += 1
    if args.assert_metrics:
        print(f"\n{'FAILED' if failures else 'OK'}: {failures} invariant failure(s)")
    return 1 if failures else 0


def _compare(n_seeds: int) -> int:
    """Signature capability #1, measured: information-gain vs. baselines."""
    from kestrel.air.planner import InfoGainPlanner, LawnmowerPlanner, RandomPlanner
    sc = scenarios.get("single_emitter")
    factories = {
        "infogain": lambda: InfoGainPlanner(),
        "lawnmower": lambda: LawnmowerPlanner(sc.area),
        "random": lambda: RandomPlanner(sc.area),
    }
    print(f"\nPlanner comparison on '{sc.name}', {n_seeds} seeds")
    print(f"  {'planner':<12} {'median err (m)':>15} {'median ellipse (m2)':>19} {'reports':>9}")
    print("  " + "-" * 58)
    results = {}
    for label, factory in factories.items():
        errs, areas, reports = [], [], []
        for seed in range(n_seeds):
            _, m = run_one(sc.name, seed=seed, planner_factory=factory, verbose=False)
            errs.append(m["median_fix_error_m"])
            areas.append(m["median_ellipse_area_m2"])
            reports.append(m["n_reports"])
        results[label] = (statistics.median(errs), statistics.mean(areas),
                          statistics.mean(reports))
        print(f"  {label:<12} {results[label][0]:>15,.1f} "
              f"{results[label][1]:>19,.0f} {results[label][2]:>9.1f}")

    ig, lm = results["infogain"][1], results["lawnmower"][1]
    if ig > 0:
        print(f"\n  info-gain ellipse is {lm / ig:.1f}x smaller than lawnmower")
    return 0


if __name__ == "__main__":
    sys.exit(main())
