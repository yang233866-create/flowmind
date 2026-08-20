from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .design_system import (
    INK, MUTED, SCENARIO_LABELS, STRATEGY_COLORS, STRATEGY_LABELS,
    add_figure_title, export_figure, panel_label, strategy_handles, style_axis,
)
from .statistics import normalized_regret_profiles

FIGURE_TITLE = "不同场景下各策略的标准化后悔值"
METRICS = ["avg_waiting_s", "avg_travel_time_s", "throughput_veh", "avg_speed_mps", "avg_queue_veh", "max_queue_veh"]
METRIC_LABELS = ["等待", "行程", "吞吐", "速度", "平均\n排队", "最大\n排队"]
SCENARIOS = ["normal", "morning_peak", "evening_peak", "event_surge", "lane_closure"]
STRATEGIES = ["fixed", "actuated", "dqn", "ppo"]


def build(data, output_dir, source_dir):
    regret = normalized_regret_profiles(data.arena)
    regret.to_csv(source_dir / "07_normalized_regret_profiles.csv", index=False)
    fig = plt.figure(figsize=(7.2, 5.05), facecolor="#FFFFFF")
    gs = fig.add_gridspec(3, 5, left=0.065, right=0.975, bottom=0.09, top=0.69,
                          height_ratios=[1.0, 0.11, 0.43], hspace=0.34, wspace=0.15)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    ax_key = fig.add_subplot(gs[1, :])
    ax_key.axis("off")
    ax_summary = fig.add_subplot(gs[2, :])
    add_figure_title(fig, FIGURE_TITLE)

    maximum = float(regret["normalized_regret"].max() * 100)
    ymax = max(100, np.ceil(maximum / 25) * 25)
    yticks = np.arange(0, ymax + 1, 25)
    x = np.arange(6)
    for index, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        block = regret.query("scenario == @scenario")
        for strategy in STRATEGIES:
            values = block.query("strategy == @strategy").set_index("metric").reindex(METRICS)["normalized_regret"].to_numpy(dtype=float) * 100
            ax.plot(x, values, color=STRATEGY_COLORS[strategy], lw=1.7, marker="o", ms=3.8, alpha=0.92)
        ax.axvline(3.5, color="#D9E1E8", lw=0.8, zorder=0)
        ax.set_ylim(-4, ymax)
        ax.set_yticks(yticks)
        ax.set_xlim(-0.3, 5.3)
        ax.set_xticks(x, [])
        if index == 0:
            ax.set_ylabel("标准化后悔值（%）")
            fig.text(0.065, 0.765, "A", fontsize=11.5, weight="bold", color="#0EA5A4")
            fig.text(0.09, 0.765, "五场景标准化后悔值", fontsize=11.5, weight="bold", color=INK)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=10.5, weight="bold", pad=5, color=INK)
        style_axis(ax, "y")

    frame = regret.assign(objective=np.where(regret["metric"].isin(["avg_queue_veh", "max_queue_veh"]), "控堵目标", "效率目标"))
    summary = frame.groupby(["strategy", "objective"], observed=True)["normalized_regret"].mean().mul(100).unstack()
    summary.to_csv(source_dir / "07_objective_mean_regret.csv")
    ax_key.text(0.5, 0.5, "1 等待  ·  2 行程  ·  3 吞吐  ·  4 速度  ·  5 平均排队  ·  6 最大排队",
                ha="center", va="center", fontsize=9, color=MUTED, transform=ax_key.transAxes)
    panel_label(ax_summary, "B", "跨场景平均标准化后悔值")
    for i, strategy in enumerate(STRATEGIES):
        efficiency = float(summary.loc[strategy, "效率目标"])
        congestion = float(summary.loc[strategy, "控堵目标"])
        ax_summary.plot([efficiency, congestion], [i, i], color=STRATEGY_COLORS[strategy], lw=2.0, alpha=0.35)
        ax_summary.scatter(efficiency, i, s=54, color=STRATEGY_COLORS[strategy], edgecolor="white", linewidth=0.8, zorder=3)
        ax_summary.scatter(congestion, i, s=54, facecolor="white", edgecolor=STRATEGY_COLORS[strategy], linewidth=1.6, zorder=3)
        ax_summary.text(efficiency + 1.1, i - 0.13, f"{efficiency:.1f}%", fontsize=9, color=INK)
        ax_summary.text(congestion + 1.1, i + 0.18, f"{congestion:.1f}%", fontsize=9, color=INK)
    ax_summary.set_yticks(np.arange(4), [STRATEGY_LABELS[s] for s in STRATEGIES])
    ax_summary.invert_yaxis()
    ax_summary.set_xlabel("跨场景平均标准化后悔值（%）")
    style_axis(ax_summary, "x")
    ax_summary.legend(handles=[
        Line2D([0], [0], marker="o", color="#475569", markerfacecolor="#475569", lw=0, label="效率目标"),
        Line2D([0], [0], marker="o", color="#475569", markerfacecolor="white", lw=0, label="控堵目标"),
    ], loc="lower right", ncol=2, fontsize=9)
    fig.legend(handles=strategy_handles(), loc="upper center", bbox_to_anchor=(0.57, 0.89), ncol=4, fontsize=9)
    return export_figure(fig, output_dir, "07_regret_landscape")
