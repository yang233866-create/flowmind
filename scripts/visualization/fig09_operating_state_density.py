from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from .design_system import (
    GRID, INK, MUTED, RISK, STRATEGY_LABELS,
    add_figure_title, export_figure, panel_label,
)

FIGURE_TITLE = "总排队—总等待运行状态密度"
STRATEGIES = ["fixed", "actuated", "dqn", "ppo"]


def _raw_states(data):
    lookup = data.arena.set_index("exp_id")[["scenario", "strategy", "seed"]]
    rows = []
    for exp_id, frame in data.timeseries.items():
        meta = lookup.loc[exp_id]
        block = frame[["queue_total", "waiting_total", "running_veh"]].copy()
        block["scenario"] = meta["scenario"]
        block["strategy"] = meta["strategy"]
        block["seed"] = int(meta["seed"])
        block["exp_id"] = exp_id
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def build(data, output_dir, source_dir):
    states = _raw_states(data)
    pseudocount = 1.0
    states["log_waiting"] = np.log10(states["waiting_total"].to_numpy(dtype=float) + pseudocount)
    summary = states.groupby("strategy", observed=True).agg(
        seconds=("queue_total", "size"), queue_median=("queue_total", "median"),
        queue_p95=("queue_total", lambda x: np.percentile(x, 95)),
        waiting_median=("waiting_total", "median"),
        waiting_p95=("waiting_total", lambda x: np.percentile(x, 95)),
        running_mean=("running_veh", "mean"),
    )
    risk = states.assign(risk=(states["queue_total"] > 40) & (states["waiting_total"] > 1000)).groupby("strategy", observed=True)["risk"].mean().mul(100)
    summary["high_risk_state_pct"] = risk
    summary.to_csv(source_dir / "09_operating_state_summary.csv")

    x_edges = np.linspace(0, 110, 29)
    y_edges = np.linspace(0, np.log10(5001), 29)
    density_rows, global_max = [], 1
    for strategy in STRATEGIES:
        block = states.query("strategy == @strategy")
        counts, _, _ = np.histogram2d(block["queue_total"], block["log_waiting"], bins=[x_edges, y_edges])
        global_max = max(global_max, int(counts.max()))
        for i, j in zip(*np.nonzero(counts)):
            density_rows.append({
                "strategy": strategy, "queue_low": x_edges[i], "queue_high": x_edges[i + 1],
                "waiting_low_s": 10 ** y_edges[j] - 1, "waiting_high_s": 10 ** y_edges[j + 1] - 1,
                "seconds": int(counts[i, j]),
            })
    pd.DataFrame(density_rows).to_csv(source_dir / "09_operating_state_density_bins.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.1), sharex=True, sharey=True, facecolor="#FFFFFF")
    fig.subplots_adjust(left=0.075, right=0.91, bottom=0.105, top=0.80, hspace=0.24, wspace=0.15)
    add_figure_title(fig, FIGURE_TITLE)
    fig.text(0.91, 0.885, "风险区：总排队 > 40 辆，总等待 > 1000 秒",
             ha="right", va="center", fontsize=9, color=RISK)

    cmap = LinearSegmentedColormap.from_list("state_density", ["#FFFFFF", "#A8DCD7", "#318C89", "#163A55"])
    norm = LogNorm(vmin=1, vmax=global_max)
    last = None
    for ax, strategy, letter in zip(axes.flat, STRATEGIES, ["A", "B", "C", "D"]):
        block = states.query("strategy == @strategy")
        last = ax.hexbin(block["queue_total"], block["log_waiting"], gridsize=(38, 32),
                         extent=(0, 110, 0, np.log10(5001)), mincnt=1, cmap=cmap, norm=norm, linewidths=0)
        ax.axvline(40, color=RISK, lw=0.9, ls=(0, (4, 3)))
        ax.axhline(np.log10(1001), color=RISK, lw=0.9, ls=(0, (4, 3)))
        ax.fill_between([40, 110], np.log10(1001), np.log10(5001), color=RISK, alpha=0.035)
        panel_label(ax, letter, STRATEGY_LABELS[strategy])
        ax.text(0.97, 0.07, f"风险状态占比：{summary.loc[strategy, 'high_risk_state_pct']:.1f}%",
                transform=ax.transAxes, ha="right", fontsize=9, color=INK)
        ax.grid(color=GRID, lw=0.6, alpha=0.28)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(length=0, labelsize=9, colors=MUTED)
        ax.set_xlim(0, 110)
        ax.set_ylim(0, np.log10(5001))
    y_values = np.array([1, 11, 101, 1001, 5001], dtype=float)
    axes[0, 0].set_yticks(np.log10(y_values), ["0", "10", "100", "1,000", "5,000"])
    for ax in axes[1, :]:
        ax.set_xlabel("总排队（辆）")
    for ax in axes[:, 0]:
        ax.set_ylabel("总等待（秒）")
    cbar = fig.colorbar(last, ax=axes, fraction=0.027, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9, length=0)
    cbar.set_label("状态停留时间（秒）", fontsize=10)
    cbar.ax.text(0.5, 1.025, "对数色阶", transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=9, color=MUTED)
    return export_figure(fig, output_dir, "09_operating_state_density")
