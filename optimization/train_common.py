"""Shared training loop for DQN/PPO on FlowMind templates."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from optimization.rl_agents import ENV_KWARGS, make_env, model_path

TB_ROOT = Path("data/results/tb")


def build_arg_parser(algo: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"Train {algo.upper()} on a FlowMind template")
    ap.add_argument("--template", default="cross_basic")
    ap.add_argument("--route", required=True, help="routes.rou.xml for training demand")
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--episode-sec", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--libsumo", action="store_true", default=True,
                    help="use libsumo backend (faster); default on")
    ap.add_argument("--no-libsumo", dest="libsumo", action="store_false")
    ap.add_argument("--device", default="auto")
    return ap


def train(algo: str, args: argparse.Namespace) -> Path:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor

    save_path = model_path(algo, args.template)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tb_dir = TB_ROOT / f"{algo}_{args.template}"
    tb_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        args.template, args.route,
        duration_sec=args.episode_sec, seed=args.seed,
        use_libsumo=args.libsumo,
    )
    env = Monitor(env)

    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(tb_dir),
        device=args.device,
    )
    if algo == "dqn":
        model = DQN(
            **common,
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            train_freq=1,
            target_update_interval=500,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            batch_size=64,
            gamma=0.99,
        )
    elif algo == "ppo":
        model = PPO(
            **common,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
    else:
        raise ValueError(algo)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    elapsed = time.time() - t0

    model.save(str(save_path))
    env.close()

    info = {
        "algo": algo, "template": args.template, "route": str(args.route),
        "timesteps": args.timesteps, "episode_sec": args.episode_sec,
        "seed": args.seed, "libsumo": args.libsumo,
        "env_kwargs": ENV_KWARGS, "train_seconds": round(elapsed, 1),
        "checkpoint": str(save_path),
    }
    meta_path = save_path.with_suffix(".train.json")
    meta_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[train_{algo}] saved {save_path} ({elapsed / 60:.1f} min, "
          f"{args.timesteps} steps)")
    return save_path
