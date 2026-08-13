"""Plot RL training curves from TensorBoard event files (SCI style).

Usage: python -m scripts.plot_training_curves [--tb-root data/results/tb]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.sci_style import STRATEGY_COLORS, STRATEGY_LABELS, apply_style, save_fig

TAG = "rollout/ep_rew_mean"


def load_scalars(run_dir: Path, tag: str = TAG) -> pd.DataFrame | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    frames = []
    for event_file in sorted(run_dir.rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(event_file.parent), size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            continue
        events = acc.Scalars(tag)
        frames.append(pd.DataFrame(
            {"step": [e.step for e in events], "value": [e.value for e in events]}
        ))
    if not frames:
        return None
    df = pd.concat(frames).sort_values("step").drop_duplicates("step")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb-root", default="data/results/tb")
    ap.add_argument("--out", default="figures/fig_rl_training_curves.png")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    apply_style()
    tb_root = Path(args.tb_root)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plotted = 0
    for algo in ("dqn", "ppo"):
        run_dirs = sorted(tb_root.glob(f"{algo}_*"))
        for run_dir in run_dirs:
            df = load_scalars(run_dir)
            if df is None or df.empty:
                continue
            smooth = df["value"].rolling(10, min_periods=1).mean()
            template = run_dir.name.split("_", 1)[1]
            label = STRATEGY_LABELS[algo]
            if len(run_dirs) > 1:
                label += f" ({template})"
            ax.plot(df["step"] / 1000, df["value"], color=STRATEGY_COLORS[algo],
                    alpha=0.18, linewidth=0.8)
            ax.plot(df["step"] / 1000, smooth, color=STRATEGY_COLORS[algo],
                    linewidth=1.5, label=label)
            plotted += 1
    if plotted == 0:
        print("No training scalars found; skip.")
        return
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Episode reward (mean)")
    ax.set_title("RL training curves (reward: diff-waiting-time)")
    ax.legend()
    out = save_fig(fig, args.out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
