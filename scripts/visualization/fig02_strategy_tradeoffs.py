from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

import numpy as np
import matplotlib.pyplot as plt

from .design_system import (
    INK, STRATEGY_COLORS, STRATEGY_LABELS, add_figure_title,
    export_figure, panel_label, strategy_handles, style_axis,
)
from .statistics import paired_effects

FIGURE_TITLE = "不同控制策略的多指标性能比较"
SCENARIO_MARKERS = {"normal": "o", "morning_peak": "^", "evening_peak": "s", "event_surge": "D", "lane_closure": "P"}


def build(data, output_dir, source_dir):
    arena = data.arena
    effects = paired_effects(arena)
    effects.to_csv(source_dir / "02_paired_effects.csv", index=False)
    summary = arena.groupby("strategy", observed=True).agg(
        waiting_mean=("avg_waiting_s", "mean"), waiting_sem=("avg_waiting_s", "sem"),
        throughput_mean=("throughput_veh", "mean"), throughput_sem=("throughput_veh", "sem"),
        max_queue_mean=("max_queue_veh", "mean"),
    )
    summary.to_csv(source_dir / "02_strategy_summary.csv")

    fig = plt.figure(figsize=(7.2, 4.3), facecolor="#FFFFFF")
    outer = fig.add_gridspec(1, 2, left=0.07, right=0.975, bottom=0.12, top=0.76,
                             width_ratios=[1.18, 0.82], wspace=0.28)
    ax = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(3, 1, hspace=0.58)
    forest_axes = [fig.add_subplot(right[i, 0]) for i in range(3)]
    add_figure_title(fig, FIGURE_TITLE)

    panel_label(ax, "A", "平均等待与平均吞吐分布")
    offsets = {"fixed": (9, -14), "actuated": (9, 10), "dqn": (9, 10), "ppo": (9, -15)}
    for strategy, block in arena.groupby("strategy", observed=True):
        color = STRATEGY_COLORS[strategy]
        for scenario, points in block.groupby("scenario", observed=True):
            ax.scatter(points["avg_waiting_s"], points["throughput_veh"], s=27,
                       marker=SCENARIO_MARKERS[scenario], color=color, edgecolor="white",
                       linewidth=0.45, alpha=0.20, zorder=2)
        row = summary.loc[strategy]
        ax.errorbar(row["waiting_mean"], row["throughput_mean"],
                    xerr=1.96 * row["waiting_sem"], yerr=1.96 * row["throughput_sem"],
                    fmt="none", ecolor=color, elinewidth=1.3, capsize=3, zorder=4)
        ax.scatter(row["waiting_mean"], row["throughput_mean"], s=92, color=color,
                   edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate(STRATEGY_LABELS[strategy], (row["waiting_mean"], row["throughput_mean"]),
                    xytext=offsets[strategy], textcoords="offset points", fontsize=9,
                    weight="bold", color=INK)
    style_axis(ax, "y")
    ax.set_xlabel("平均等待（秒）")
    ax.set_ylabel("平均吞吐（辆）")
    ax.set_xlim(15, 64)
    ax.set_ylim(1090, 1820)

    panel_label(forest_axes[0], "B", "相对 Fixed 的配对差值")
    specs = [
        ("avg_waiting_s", "平均等待差值（秒）", (-5, 45)),
        ("max_queue_veh", "最大排队差值（辆）", (-5, 52)),
        ("throughput_veh", "吞吐差值（辆）", (-340, 310)),
    ]
    strategies = ["actuated", "dqn", "ppo"]
    for axis, (metric, label, xlim) in zip(forest_axes, specs):
        block = effects.query("metric == @metric").set_index("strategy").loc[strategies]
        for yi, strategy in enumerate(strategies):
            row = block.loc[strategy]
            axis.errorbar(row["benefit_mean"], yi,
                          xerr=[[row["benefit_mean"] - row["ci_low"]], [row["ci_high"] - row["benefit_mean"]]],
                          fmt="o", ms=6, color=STRATEGY_COLORS[strategy],
                          ecolor=STRATEGY_COLORS[strategy], elinewidth=1.7, capsize=3, zorder=3)
            span = xlim[1] - xlim[0]
            axis.text(min(row["ci_high"] + span * 0.025, xlim[1] - span * 0.08), yi,
                      f"{row['benefit_mean']:+.1f}", va="center", fontsize=9, color=INK)
        axis.axvline(0, color="#667085", lw=0.8, alpha=0.65)
        axis.set_xlim(*xlim)
        axis.set_yticks(np.arange(3), [STRATEGY_LABELS[s] for s in strategies])
        axis.invert_yaxis()
        axis.set_xlabel(label)
        style_axis(axis, "x")

    fig.legend(handles=strategy_handles(), loc="upper center", bbox_to_anchor=(0.52, 0.89),
               ncol=4, fontsize=9, columnspacing=1.5)
    return export_figure(fig, output_dir, "02_strategy_tradeoffs")
