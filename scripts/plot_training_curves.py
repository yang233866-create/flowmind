"""Plot RL training curves from TensorBoard event files (SCI style).

Only the most recent SB3 run per algorithm is plotted: `data/results/tb/dqn_*`
accumulates one subdirectory per training run (DQN_1, DQN_2, ...), and merging
them would splice an older run's scalars into the curve at the same step numbers.

Usage: python -m scripts.plot_training_curves [--tb-root data/results/tb]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from scripts.sci_style import STRATEGY_LABELS, apply_style, save_fig

REWARD_TAG = "rollout/ep_rew_mean"
LOSS_TAG = {"dqn": "train/loss", "ppo": "train/value_loss"}
RUN_DIR_RE = re.compile(r"_(\d+)$")


def latest_run_dir(tb_root: Path, algo: str) -> Path | None:
    """Newest SB3 run directory (e.g. .../dqn_cross_basic/DQN_3)."""
    candidates = []
    for algo_dir in sorted(tb_root.glob(f"{algo}_*")):
        for run in algo_dir.iterdir() if algo_dir.is_dir() else []:
            if not run.is_dir() or not any(run.glob("events.out.tfevents.*")):
                continue
            m = RUN_DIR_RE.search(run.name)
            candidates.append(((int(m.group(1)) if m else 0), run.stat().st_mtime, run))
    if not candidates:
        return None
    return max(candidates)[2]


def load_scalars(run_dir: Path, tag: str) -> pd.DataFrame | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    events = acc.Scalars(tag)
    df = pd.DataFrame({"step": [e.step for e in events],
                       "value": [e.value for e in events]})
    return df.sort_values("step").drop_duplicates("step")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb-root", default="data/results/tb")
    ap.add_argument("--out", default="figures/fig_rl_training_curves.png")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    apply_style()
    tb_root = Path(args.tb_root)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    for ax, algo in zip(axes, ("dqn", "ppo")):
        run_dir = latest_run_dir(tb_root, algo)
        ax.set_title(f"{STRATEGY_LABELS[algo]} training progress")
        ax.set_xlabel("Training steps (thousands)")
        if run_dir is None:
            ax.text(0.5, 0.5, f"No {algo.upper()} runs", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        print(f"[{algo}] {run_dir}")

        reward = load_scalars(run_dir, REWARD_TAG)
        ax.set_ylabel("Episode reward (mean)")
        if reward is not None and not reward.empty:
            ax.plot(reward["step"] / 1000,
                    reward["value"].rolling(5, min_periods=1).mean(),
                    color="#1f4e79", linewidth=1.6, label="ep_rew_mean")

        loss = load_scalars(run_dir, LOSS_TAG[algo])
        if loss is not None and not loss.empty:
            ax2 = ax.twinx()
            ax2.plot(loss["step"] / 1000,
                     loss["value"].rolling(20, min_periods=1).mean(),
                     color="#c0504d", linewidth=1.0, alpha=0.8,
                     label=LOSS_TAG[algo].split("/")[-1])
            ax2.set_ylabel(LOSS_TAG[algo].split("/")[-1].replace("_", " "))
            ax2.grid(False)
            handles = ax.get_lines() + ax2.get_lines()
            ax.legend(handles, [h.get_label() for h in handles], fontsize=8)
        else:
            ax.legend(fontsize=8)

    fig.tight_layout()
    print(f"saved {save_fig(fig, args.out)}")


if __name__ == "__main__":
    main()
