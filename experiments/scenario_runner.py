"""Run one (scenario, strategy, seed) experiment end-to-end.

CLI:
    python -m experiments.scenario_runner --scenario morning_peak \
        --strategy fixed --seed 0 [--template cross_basic] [--gui]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.scenarios import resolve_scenario

RESULTS_ROOT = Path("data/results/experiments")


def exp_id(scenario_name: str, strategy: str, seed: int) -> str:
    return f"{scenario_name}__{strategy}__s{seed}"


def run_experiment(
    scenario: str,
    strategy: str,
    seed: int = 0,
    template: str = "cross_basic",
    base_state: str | None = None,
    gui: bool = False,
    force: bool = False,
):
    """Generate routes for the scenario and run one episode. Returns RunResult-like info."""
    from simulation.route_generator import generate_routes
    from simulation.sumo_runner import run_episode

    spec = resolve_scenario(scenario, base_state=base_state)
    out_dir = RESULTS_ROOT / exp_id(spec["name"], strategy, seed)

    summary = out_dir / "metrics_summary.json"
    if summary.exists() and not force:
        metrics = json.loads(summary.read_text(encoding="utf-8"))
        return {"exp_id": out_dir.name, "metrics": metrics, "out_dir": out_dir, "cached": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "scenario_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")

    route_file = generate_routes(
        state_path=spec["base_state"],
        template=template,
        out_dir=out_dir,
        scenario=spec,
        seed=seed,
    )
    result = run_episode(
        template=template,
        route_file=route_file,
        strategy=strategy,
        seed=seed,
        out_dir=out_dir,
        gui=gui,
        duration_sec=int(spec["duration_sec"]),
        lane_closures=spec["lane_closures"],
    )
    return {"exp_id": out_dir.name, "metrics": result.metrics, "out_dir": out_dir, "cached": False}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one FlowMind experiment")
    ap.add_argument("--scenario", required=True, help="registry name or spec JSON path")
    ap.add_argument("--strategy", required=True, choices=["fixed", "actuated", "dqn", "ppo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template", default="cross_basic")
    ap.add_argument("--base-state", default=None)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if cached")
    args = ap.parse_args()

    info = run_experiment(
        args.scenario, args.strategy, args.seed,
        template=args.template, base_state=args.base_state,
        gui=args.gui, force=args.force,
    )
    tag = " (cached)" if info["cached"] else ""
    print(f"[scenario_runner] {info['exp_id']}{tag}")
    for k, v in info["metrics"].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
