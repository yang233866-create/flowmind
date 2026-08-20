from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

import numpy as np
import matplotlib.pyplot as plt

from .design_system import (
    SCENARIO_LABELS, STRATEGY_COLORS, add_figure_title, export_figure,
    panel_label, strategy_handles, style_axis,
)
from .statistics import paired_run_transitions

FIGURE_TITLE = "不同策略相对 Fixed 的配对实验结果"
SCENARIOS = ["normal", "morning_peak", "evening_peak", "event_surge", "lane_closure"]
STRATEGIES = ["actuated", "dqn", "ppo"]
METRICS = [
    ("avg_waiting_s", "平均等待", "平均等待（秒）", "A"),
    ("max_queue_veh", "最大排队", "最大排队（辆）", "B"),
    ("throughput_veh", "吞吐", "吞吐（辆）", "C"),
]


def build(data, output_dir, source_dir):
    transitions = paired_run_transitions(data.arena)
    transitions.to_csv(source_dir / "08_paired_run_transitions.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.2), facecolor="#FFFFFF")
    fig.subplots_adjust(left=0.09, right=0.975, bottom=0.12, top=0.76, wspace=0.20)
    add_figure_title(fig, FIGURE_TITLE)
    base_y = np.arange(len(SCENARIOS))[::-1]
    offsets = {"actuated": 0.22, "dqn": 0.0, "ppo": -0.22}
    seed_offsets = {0: -0.045, 1: 0.0, 2: 0.045}
    for ax, (metric, title, xlabel, letter) in zip(axes, METRICS):
        block = transitions.query("metric == @metric")
        for scenario_index, scenario in enumerate(SCENARIOS):
            center = base_y[scenario_index]
            if scenario_index % 2 == 0:
                ax.axhspan(center - 0.42, center + 0.42, color="#F7F9FB", zorder=0)
            for strategy in STRATEGIES:
                pairs = block.query("scenario == @scenario and strategy == @strategy").sort_values("seed")
                for row in pairs.itertuples(index=False):
                    y = center + offsets[strategy] + seed_offsets[row.seed]
                    ax.plot([row.baseline_value, row.method_value], [y, y],
                            color=STRATEGY_COLORS[strategy], alpha=0.45, lw=0.95, zorder=2)
                    ax.scatter(row.baseline_value, y, s=15, color=STRATEGY_COLORS["fixed"],
                               edgecolor="white", linewidth=0.35, zorder=3)
                    ax.scatter(row.method_value, y, s=25, color=STRATEGY_COLORS[strategy],
                               edgecolor="white", linewidth=0.45, zorder=4)
        panel_label(ax, letter, title)
        ax.set_yticks(base_y, [SCENARIO_LABELS[s] for s in SCENARIOS])
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(xlabel)
        style_axis(ax, "x")
    fig.legend(handles=strategy_handles(), loc="upper center", bbox_to_anchor=(0.53, 0.89),
               ncol=4, fontsize=9)
    return export_figure(fig, output_dir, "08_paired_transitions")
