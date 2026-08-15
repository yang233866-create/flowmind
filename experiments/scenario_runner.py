"""Run one (scenario, strategy, seed) experiment end-to-end.

CLI:
    python -m experiments.scenario_runner --scenario morning_peak \
        --strategy fixed --seed 0 [--template cross_basic] [--gui]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.scenarios import resolve_scenario

RESULTS_ROOT = Path("data/results/experiments")
RUN_META = "run_meta.json"


def exp_id(scenario_name: str, strategy: str, seed: int) -> str:
    return f"{scenario_name}__{strategy}__s{seed}"


def _sha1(path: str | Path) -> str | None:
    p = Path(path)
    return hashlib.sha1(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def run_fingerprint(spec: dict, strategy: str, seed: int, template: str) -> dict:
    """Identity of everything that changes the numbers, not just the directory name.

    The directory name is (scenario, strategy, seed) only, so a cache keyed on it
    happily mixes runs built from different TrafficStates or different model
    checkpoints into one comparison table. Hash the inputs instead.
    """
    from optimization.rl_agents import model_path

    payload = {
        "base_state": str(spec["base_state"]),
        "base_state_sha1": _sha1(spec["base_state"]),
        "flow_multipliers": spec.get("flow_multipliers"),
        "lane_closures": spec.get("lane_closures"),
        "duration_sec": spec.get("duration_sec"),
        "template": template,
        "strategy": strategy,
        "seed": seed,
        "model_sha1": (_sha1(model_path(strategy, template))
                       if strategy in ("dqn", "ppo") else None),
    }
    payload["key"] = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return payload


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
    fp = run_fingerprint(spec, strategy, seed, template)

    summary = out_dir / "metrics_summary.json"
    meta_path = out_dir / RUN_META
    if summary.exists() and not force:
        cached_fp = {}
        if meta_path.exists():
            cached_fp = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached_fp.get("key") == fp["key"]:
            metrics = json.loads(summary.read_text(encoding="utf-8"))
            return {"exp_id": out_dir.name, "metrics": metrics, "out_dir": out_dir,
                    "cached": True, "fingerprint": fp}
        print(f"    [cache] {out_dir.name} was built from different inputs "
              f"({cached_fp.get('key', 'no run_meta.json')} != {fp['key']}); re-running",
              flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "scenario_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    # written only after the run succeeds, so a crashed run stays uncached
    meta_path.unlink(missing_ok=True)

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
    fp["route_sha1"] = _sha1(route_file)
    meta_path.write_text(json.dumps(fp, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"exp_id": out_dir.name, "metrics": result.metrics, "out_dir": out_dir,
            "cached": False, "fingerprint": fp}


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
