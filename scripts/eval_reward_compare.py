"""Small experiment: compare diff-waiting-time vs speed-weighted-queue on 'normal'.

Runs one evaluation episode of each DQN checkpoint (both trained short) and
prints the four metrics that matter for the "travel time vs waiting" tradeoff.

Usage:
    .venv\\Scripts\\python.exe scripts/eval_reward_compare.py
"""
from __future__ import annotations

from simulation.sumo_home import ensure_sumo_home

ensure_sumo_home()

from pathlib import Path

from experiments.scenarios import resolve_scenario
from optimization.rl_agents import register_reward_fn, run_rl_episode
from simulation.route_generator import generate_routes

register_reward_fn()

TEMPLATE = "cross_basic"
SCENARIO = "normal"
SEED = 0
EPISODE_SEC = 1800

KEYS = ("avg_waiting_s", "avg_travel_time_s", "throughput_veh", "avg_speed_mps", "avg_queue_veh")


def evaluate(reward_fn: str) -> dict:
    spec = resolve_scenario(SCENARIO)
    out_dir = Path("data/results/_reward_compare") / reward_fn.replace("-", "_")
    route_file = generate_routes(
        state_path=spec["base_state"], template=TEMPLATE, out_dir=out_dir,
        scenario=spec, seed=SEED,
    )
    metrics, _ = run_rl_episode(
        template=TEMPLATE, route_file=route_file, strategy="dqn", seed=SEED,
        out_dir=out_dir, duration_sec=EPISODE_SEC, reward_fn=reward_fn,
    )
    return {k: metrics[k] for k in KEYS}


if __name__ == "__main__":
    print("=== diff-waiting-time (baseline) ===")
    base = evaluate("diff-waiting-time")
    for k, v in base.items():
        print(f"  {k}: {v:.2f}")
    print("=== speed-weighted-queue (experiment) ===")
    exp = evaluate("speed-weighted-queue")
    for k, v in exp.items():
        print(f"  {k}: {v:.2f}")
    print("=== delta (exp - base) ===")
    for k in KEYS:
        d = exp[k] - base[k]
        print(f"  {k}: {d:+.2f}")
