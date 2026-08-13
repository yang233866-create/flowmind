"""RL environment construction and evaluation for FlowMind.

Training and evaluation both go through sumo-rl's SumoEnvironment so the
agent always sees the observation space it was trained on. Metrics for
evaluation episodes come from the same tripinfo/MetricsCollector pipeline
as the fixed/actuated baselines (see docs/CONTRACTS.md).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from simulation.sumo_home import ensure_sumo_home

ensure_sumo_home()  # must run before importing sumo_rl

TEMPLATES_ROOT = Path("simulation/templates")
MODELS_ROOT = Path("models")

# sumo-rl agent timing (contract: delta_time=5, yellow_time=3, min_green=5)
ENV_KWARGS = dict(delta_time=5, yellow_time=3, min_green=5, max_green=60)


def template_paths(template: str) -> tuple[Path, dict]:
    tdir = TEMPLATES_ROOT / template
    meta = json.loads((tdir / "meta.json").read_text(encoding="utf-8"))
    return tdir / "net.net.xml", meta


def model_path(strategy: str, template: str) -> Path:
    return MODELS_ROOT / f"{strategy}_{template}.zip"


def make_env(
    template: str,
    route_file: str | Path,
    duration_sec: int = 1800,
    seed: int = 0,
    out_csv_name: str | None = None,
    tripinfo_out: str | Path | None = None,
    use_libsumo: bool = False,
):
    """Create a single-agent sumo-rl environment on a FlowMind template."""
    if use_libsumo:
        os.environ["LIBSUMO_AS_TRACI"] = "1"
    else:
        os.environ.pop("LIBSUMO_AS_TRACI", None)

    # sumo_rl reads LIBSUMO at import time -> import after env var is set
    import importlib

    import sumo_rl.environment.env as sumo_rl_env

    if sumo_rl_env.LIBSUMO != use_libsumo:
        importlib.reload(sumo_rl_env)

    net_file, _ = template_paths(template)
    additional = "--no-step-log --duration-log.disable"
    if tripinfo_out is not None:
        additional += f" --tripinfo-output {tripinfo_out}"

    return sumo_rl_env.SumoEnvironment(
        net_file=str(net_file),
        route_file=str(route_file),
        single_agent=True,
        num_seconds=duration_sec,
        sumo_seed=seed,
        out_csv_name=out_csv_name,
        time_to_teleport=300,
        additional_sumo_cmd=additional,
        sumo_warnings=False,
        **ENV_KWARGS,
    )


def load_model(strategy: str, template: str):
    """Load a trained SB3 model; raises with a clear message if missing."""
    from stable_baselines3 import DQN, PPO

    path = model_path(strategy, template)
    if not path.exists():
        raise RuntimeError(
            f"No trained {strategy.upper()} checkpoint for template '{template}' "
            f"(expected {path}). Train first: python -m optimization.train_{strategy} "
            f"--template {template} --route <routes.rou.xml>"
        )
    cls = {"dqn": DQN, "ppo": PPO}[strategy]
    return cls.load(str(path), device="cpu")  # inference is tiny; avoid GPU init cost


def run_rl_episode(
    template: str,
    route_file: str | Path,
    strategy: str,
    seed: int,
    out_dir: Path,
    duration_sec: int = 1800,
    lane_closures: list[dict] | None = None,
    gui: bool = False,
):
    """Evaluate a trained agent for one episode; returns (metrics, timeseries_df).

    Timeseries/metrics are collected exactly like the baselines: per-second
    halting counts on approach in-edges + tripinfo aggregation.
    """
    import pandas as pd

    from simulation.metrics import MetricsCollector

    _, meta = template_paths(template)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tripinfo = out_dir / "tripinfo.xml"

    model = load_model(strategy, template)
    env = make_env(
        template, route_file, duration_sec=duration_sec, seed=seed,
        tripinfo_out=tripinfo, use_libsumo=False,  # traci for evaluation stability
    )
    if gui:
        env.use_gui = True

    obs, _ = env.reset()
    sumo = env.sumo  # traci-compatible handle
    if lane_closures:
        for closure in lane_closures:
            in_edge = meta["approaches"][closure["approach"]]["in_edge"]
            n_total = meta["approaches"][closure["approach"]]["n_lanes"]
            for li in range(min(int(closure["n_lanes"]), n_total)):
                sumo.lane.setAllowed(f"{in_edge}_{li}", ["authority"])

    collector = MetricsCollector(sumo, meta)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        t = int(env.sim_step)
        # env.step advances delta_time seconds; sample once per agent step
        collector.step(t)
    env.close()

    ts = collector.to_dataframe()
    metrics = collector.parse_tripinfo(tripinfo)  # includes teleport count
    ts.to_csv(out_dir / "timeseries.csv", index=False)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics, ts
