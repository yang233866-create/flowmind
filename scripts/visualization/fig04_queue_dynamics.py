from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .design_system import (
    INK, RISK, STRATEGY_COLORS, STRATEGY_LABELS, add_figure_title,
    export_figure, panel_label, strategy_handles, style_axis,
)
from .statistics import per_run_queue_summary, queue_exceedance

FIGURE_TITLE = "不同控制策略的队列动态特征"
DIRECTIONS = ["queue_north", "queue_south", "queue_east", "queue_west"]
DIRECTION_LABELS = ["北向", "南向", "东向", "西向"]


def _time_direction_matrix(frame, bins=60):
    usable = frame[DIRECTIONS].copy()
    index_groups = np.array_split(np.arange(len(usable)), bins)
    return np.vstack([usable.iloc[index].mean(axis=0).to_numpy(dtype=float) for index in index_groups]).T


def build(data, output_dir, source_dir):
    exceedance = queue_exceedance(data.timeseries, data.arena)
    queue_runs = per_run_queue_summary(data.timeseries, data.arena)
    exceedance.to_csv(source_dir / "04_queue_exceedance.csv", index=False)
    queue_runs.to_csv(source_dir / "04_per_run_queue_summary.csv", index=False)

    matrices = {}
    for strategy in ("fixed", "ppo"):
        matrix = _time_direction_matrix(data.timeseries[f"morning_peak__{strategy}__s0"])
        matrices[strategy] = matrix
        pd.DataFrame(matrix, index=DIRECTION_LABELS).to_csv(source_dir / f"04_{strategy}_morning_peak_direction_matrix.csv")
    vmax = max(10.0, np.ceil(max(float(v.max()) for v in matrices.values()) / 5) * 5)

    fig = plt.figure(figsize=(7.2, 4.4), facecolor="#FFFFFF")
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.94, bottom=0.10, top=0.76,
                          width_ratios=[0.92, 1.08], hspace=0.36, wspace=0.27)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[1, 0])
    heat_gs = gs[:, 1].subgridspec(2, 1, hspace=0.36)
    heat_axes = [fig.add_subplot(heat_gs[i, 0]) for i in range(2)]
    add_figure_title(fig, FIGURE_TITLE)

    panel_label(ax_curve, "A", "队列超限概率曲线")
    for strategy in ["fixed", "actuated", "dqn", "ppo"]:
        block = exceedance.query("strategy == @strategy")
        ax_curve.plot(block["threshold"], block["exceedance_pct"], lw=1.9,
                      color=STRATEGY_COLORS[strategy])
    ax_curve.axvline(40, color=RISK, lw=1.0, ls=(0, (4, 3)))
    ax_curve.text(41.5, 92, "阈值：40 辆", color=RISK, fontsize=9)
    style_axis(ax_curve, "y")
    ax_curve.set_xlim(0, 90)
    ax_curve.set_ylim(0, 100)
    ax_curve.set_xlabel("队列阈值（辆）")
    ax_curve.set_ylabel("超过阈值的时间占比（%）")

    panel_label(ax_dist, "B", "单次实验 P95 队列分布")
    strategies = ["fixed", "actuated", "dqn", "ppo"]
    values = [queue_runs.query("strategy == @s")["queue_p95"].to_numpy() for s in strategies]
    violins = ax_dist.violinplot(values, positions=np.arange(4), widths=0.72, showmeans=False, showextrema=False)
    for body, strategy in zip(violins["bodies"], strategies):
        body.set_facecolor(STRATEGY_COLORS[strategy])
        body.set_edgecolor(STRATEGY_COLORS[strategy])
        body.set_linewidth(0.7)
        body.set_alpha(0.16)
    for i, (strategy, vals) in enumerate(zip(strategies, values)):
        jitter = np.linspace(-0.07, 0.07, len(vals))
        ax_dist.scatter(np.full(len(vals), i) + jitter, vals, s=22,
                        color=STRATEGY_COLORS[strategy], edgecolor="white",
                        linewidth=0.45, alpha=0.75, zorder=3)
        median = float(np.median(vals))
        ax_dist.plot([i - 0.20, i + 0.20], [median, median], color="#4B5563", lw=1.1, zorder=4)
    style_axis(ax_dist, "y")
    ax_dist.set_xticks(range(4), [STRATEGY_LABELS[s].split()[0] for s in strategies])
    ax_dist.set_ylabel("单次实验 P95 队列（辆）")

    cmap = LinearSegmentedColormap.from_list("queue", ["#F5FAFA", "#83CFC7", "#E7B266", "#C95B65"])
    image = None
    for axis, strategy, letter, title in zip(
        heat_axes, ("fixed", "ppo"), ("C", "D"), ("Fixed 方向队列时序", "PPO 方向队列时序")
    ):
        image = axis.imshow(matrices[strategy], aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        panel_label(axis, letter, title)
        axis.set_yticks(range(4), DIRECTION_LABELS)
        axis.set_xticks([0, 15, 30, 45, 59], ["0", "7.5", "15", "22.5", "30"])
        axis.set_xlabel("仿真时间（分钟）")
        axis.tick_params(length=0, labelsize=9)
        for spine in axis.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(image, ax=heat_axes, orientation="vertical", fraction=0.028, pad=0.025)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9, length=0)
    cbar.set_label("方向队列（辆）", fontsize=10)
    fig.legend(handles=strategy_handles(), loc="upper center", bbox_to_anchor=(0.48, 0.89),
               ncol=4, fontsize=9)
    return export_figure(fig, output_dir, "04_queue_dynamics")
