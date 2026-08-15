"""FlowMind end-to-end pipeline: vision -> twin -> baselines -> RL -> arena -> figures.

Usage:
    python -m scripts.run_pipeline [--quick] [--skip-vision] [--skip-train]

--quick: short RL training (20k steps), 1 seed, short episodes (900 s) -- smoke run.
Full mode: 100k steps, 3 seeds, 1800 s episodes.

Each stage is skipped automatically when its outputs already exist
(delete the outputs or pass --force to re-run).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable

DEMO_VIDEO = Path("data/videos/demo.mp4")
DEMO_ROI = Path("data/videos/demo_roi.json")
DEMO_STATE = Path("data/traffic_states/demo_001.json")
SYNTH_STATE = Path("data/traffic_states/synthetic_demo.json")
TRAIN_ROUTES = Path("data/simulations/train_cross_basic/routes.rou.xml")


def run(cmd: list[str], name: str) -> None:
    print(f"\n=== [{name}] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"[{name}] failed with exit code {r.returncode}")
    print(f"=== [{name}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def _routes_stale(route_file: Path, base_state: Path, episode_sec: int) -> bool:
    """True when the cached training routes came from a different state/horizon."""
    meta = route_file.parent / "gen_meta.json"
    if not route_file.exists() or not meta.exists():
        return True
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    same_state = Path(info.get("state", "")) == base_state
    covers = float(info.get("duration_sec") or 0) >= episode_sec
    if not same_state or not covers:
        print(f"[routes] regenerating {route_file}: state={info.get('state')} "
              f"duration={info.get('duration_sec')} (want {base_state}, >={episode_sec})")
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    timesteps = "20000" if args.quick else "100000"
    seeds = "1" if args.quick else "3"
    episode_sec = 900 if args.quick else 1800

    # 1. Vision: video -> TrafficState (+ annotated video + vision figures)
    if not args.skip_vision and DEMO_VIDEO.exists() and DEMO_ROI.exists() and (
        args.force or not DEMO_STATE.exists()
    ):
        run([PY, "-m", "vision.analyze",
             "--video", str(DEMO_VIDEO), "--roi-config", str(DEMO_ROI),
             "--state-out", str(DEMO_STATE),
             "--annotated-video", "data/videos/demo_annotated.mp4",
             "--figures-dir", "figures"], "vision")
    else:
        print("[vision] skipped (missing inputs, cached state, or --skip-vision)")

    base_state = DEMO_STATE if DEMO_STATE.exists() else SYNTH_STATE
    if not base_state.exists():
        raise SystemExit("No TrafficState available; run vision first or add "
                         f"{SYNTH_STATE}")
    print(f"[pipeline] base TrafficState: {base_state}")

    # 2. Training routes for cross_basic (normal scenario demand).
    # The routes must cover the whole training episode and come from the same
    # TrafficState the arena evaluates on, otherwise the agent trains on an empty
    # intersection for part of every episode / on a different demand entirely.
    if args.force or _routes_stale(TRAIN_ROUTES, base_state, episode_sec):
        run([PY, "-m", "simulation.route_generator",
             "--state", str(base_state), "--template", "cross_basic",
             "--out", str(TRAIN_ROUTES.parent), "--seed", "0",
             "--duration", str(episode_sec)], "routes")
    else:
        print(f"[routes] reusing {TRAIN_ROUTES}")

    # 3. RL training (DQN then PPO)
    strategies = ["fixed", "actuated"]
    if not args.skip_train:
        for algo in ("dqn", "ppo"):
            ckpt = Path(f"models/{algo}_cross_basic.zip")
            if args.force or not ckpt.exists():
                run([PY, "-m", f"optimization.train_{algo}",
                     "--template", "cross_basic", "--route", str(TRAIN_ROUTES),
                     "--timesteps", timesteps, "--episode-sec", str(episode_sec)],
                    f"train_{algo}")
            strategies.append(algo)
    else:
        for algo in ("dqn", "ppo"):
            if Path(f"models/{algo}_cross_basic.zip").exists():
                strategies.append(algo)

    # 4. Strategy Arena: all scenarios x strategies x seeds (+ arena figures)
    run([PY, "-m", "experiments.strategy_compare",
         "--scenarios", "all", "--strategies", ",".join(strategies),
         "--seeds", seeds, "--base-state", str(base_state)], "arena")

    # 5. Remaining paper figures
    run([PY, "-m", "scripts.make_architecture_figure"], "fig_architecture")
    run([PY, "-m", "scripts.plot_networks"], "fig_networks")
    if not args.skip_train:
        run([PY, "-m", "scripts.plot_training_curves"], "fig_training")

    print("\n[pipeline] complete. figures/ and data/results/ are up to date.")


if __name__ == "__main__":
    main()
