from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

import numpy as np
import matplotlib.pyplot as plt

from .design_system import (
    INK, MUTED, SCENARIO_LABELS, STRATEGY_COLORS,
    add_figure_title, export_figure, strategy_handles, style_axis,
)
from .statistics import aligned_queue_trajectories

FIGURE_TITLE = "五类场景下的队列时序变化"
SCENARIOS = ["normal", "morning_peak", "evening_peak", "event_surge", "lane_closure"]
STRATEGIES = ["fixed", "actuated", "dqn", "ppo"]


def _condition(spec):
    multipliers = spec.get("flow_multipliers", {})
    if spec.get("lane_closures", []):
        return "东向车道封闭"
    changed = [(key, value) for key, value in multipliers.items() if abs(float(value) - 1.0) > 1e-9]
    if not changed:
        return "基准需求"
    labels = {"north": "N", "south": "S", "east": "E", "west": "W"}
    return " ".join(f"{labels[key]}×{value:g}" for key, value in changed)


def build(data, output_dir, source_dir):
    trajectories = aligned_queue_trajectories(data.timeseries, data.arena, window=30)
    trajectories.to_csv(source_dir / "10_aligned_queue_trajectories_30s.csv", index=False)

    fig, axes = plt.subplots(5, 1, figsize=(7.2, 4.6), sharex=True, sharey=True, facecolor="#FFFFFF")
    fig.subplots_adjust(left=0.15, right=0.975, bottom=0.09, top=0.82, hspace=0.15)
    add_figure_title(fig, FIGURE_TITLE)
    observed_max = float(trajectories["queue_max"].max())
    ymax = max(100.0, np.ceil(observed_max / 10) * 10)
    yticks = np.arange(0, ymax + 1, 20)

    for index, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        block = trajectories.query("scenario == @scenario")
        for strategy in STRATEGIES:
            line = block.query("strategy == @strategy").sort_values("time_s")
            x = line["time_s"].to_numpy(dtype=float) / 60
            mean = line["queue_mean"].to_numpy(dtype=float)
            low = line["queue_min"].to_numpy(dtype=float)
            high = line["queue_max"].to_numpy(dtype=float)
            ax.fill_between(x, low, high, color=STRATEGY_COLORS[strategy], alpha=0.10, linewidth=0)
            ax.plot(x, mean, color=STRATEGY_COLORS[strategy], lw=1.6)
        style_axis(ax, "y")
        ax.set_ylim(0, ymax)
        ax.set_yticks(yticks)
        ax.set_xlim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.text(-0.035, 0.62, SCENARIO_LABELS[scenario], transform=ax.transAxes,
                ha="right", va="center", fontsize=9.8, color=INK, weight="bold")
        ax.text(-0.035, 0.36, _condition(data.scenarios[scenario]), transform=ax.transAxes,
                ha="right", va="center", fontsize=9, color=MUTED)
        if index < 4:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("仿真时间（分钟）")
    fig.text(0.15, 0.835, '队列（辆）', ha='left', va='bottom', fontsize=10, color=INK, gid='queue-unit-label')
    fig.legend(handles=strategy_handles(), loc="upper center", bbox_to_anchor=(0.57, 0.90),
               ncol=4, fontsize=9)
    return export_figure(fig, output_dir, "10_scenario_timeline_atlas")
