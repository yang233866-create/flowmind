from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from .design_system import (
    HEATMAP_VALUE_SIZE, INK, MUTED, SCENARIO_LABELS, STRATEGY_LABELS,
    add_figure_title, export_figure, panel_label,
)
from .statistics import LOWER_IS_BETTER, scenario_benefits

FIGURE_TITLE = "五类场景下相对 Fixed 的性能变化率"
HEATMAP_BENEFIT_LIMIT = 70
SCENARIO_ORDER = ["normal", "morning_peak", "evening_peak", "event_surge", "lane_closure"]
STRATEGY_ORDER = ["actuated", "dqn", "ppo"]


def _condition(spec):
    mult = spec.get("flow_multipliers", {})
    if spec.get("lane_closures"):
        return "东向车道封闭"
    changed = [(k, v) for k, v in mult.items() if abs(v - 1.0) > 1e-9]
    if not changed:
        return "基准需求"
    direction = {"north": "N", "south": "S", "east": "E", "west": "W"}
    return " ".join(f"{direction[k]}×{v:g}" for k, v in changed)


def _seed_win_counts(arena, metric):
    base = arena.query("strategy == 'fixed'").set_index(["scenario", "seed"])
    rows = []
    for strategy in STRATEGY_ORDER:
        method = arena.query("strategy == @strategy").set_index(["scenario", "seed"])
        for scenario in SCENARIO_ORDER:
            m, b = method.loc[scenario, metric], base.loc[scenario, metric]
            benefit = b - m if metric in LOWER_IS_BETTER else m - b
            rows.append({"scenario": scenario, "strategy": strategy, "wins": int((benefit > 0).sum())})
    return pd.DataFrame(rows)


def build(data, output_dir, source_dir):
    benefits = scenario_benefits(data.arena)
    benefits.to_csv(source_dir / "03_scenario_benefits.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.4), facecolor="#FFFFFF")
    fig.subplots_adjust(left=0.145, right=0.965, bottom=0.18, top=0.76, wspace=0.13)
    add_figure_title(fig, FIGURE_TITLE)
    cmap = LinearSegmentedColormap.from_list("benefit", ["#C96A71", "#FFFFFF", "#4A9D8F"])
    norm = TwoSlopeNorm(vmin=-HEATMAP_BENEFIT_LIMIT, vcenter=0, vmax=HEATMAP_BENEFIT_LIMIT)
    specs = [("avg_waiting_s", "平均等待改善率", "A"),
             ("max_queue_veh", "最大排队改善率", "B"),
             ("throughput_veh", "吞吐改善率", "C")]
    image = None
    for ax, (metric, title, letter) in zip(axes, specs):
        block = benefits.query("metric == @metric").pivot(index="scenario", columns="strategy", values="benefit_pct").reindex(index=SCENARIO_ORDER, columns=STRATEGY_ORDER)
        wins = _seed_win_counts(data.arena, metric).pivot(index="scenario", columns="strategy", values="wins").reindex(index=SCENARIO_ORDER, columns=STRATEGY_ORDER)
        image = ax.imshow(block.to_numpy(), cmap=cmap, norm=norm, aspect="equal")
        panel_label(ax, letter, title)
        ax.set_xticks(range(3), [STRATEGY_LABELS[s].split()[0] for s in STRATEGY_ORDER])
        ax.set_yticks(range(5), [])
        ax.tick_params(length=0, labelsize=9.2)
        if ax is axes[0]:
            for i, scenario in enumerate(SCENARIO_ORDER):
                ax.text(-0.08, i - 0.11, SCENARIO_LABELS[scenario], transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=9.5, weight="bold", color=INK)
                ax.text(-0.08, i + 0.16, _condition(data.scenarios[scenario]), transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=8.5, color=MUTED)
        for i in range(5):
            for j in range(3):
                value = float(block.iloc[i, j])
                color = "white" if abs(value) >= 43 else INK
                ax.text(j, i, f"{value:+.0f}%", ha="center", va="center",
                        fontsize=HEATMAP_VALUE_SIZE, weight="bold", color=color)
                marker = "●" if wins.iloc[i, j] == 3 else ("◐" if wins.iloc[i, j] == 2 else "○")
                ax.text(j + 0.32, i - 0.31, marker, ha="center", va="center",
                        fontsize=7.5, color=color)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.045, pad=0.13, aspect=38)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9, length=0)
    cbar.set_label("相对 Fixed 改善率（%）", fontsize=10)
    fig.legend(handles=[Line2D([0], [0], marker="o", linestyle="none", color=INK,
                               markerfacecolor=INK, markersize=5, label="3/3 种子同方向")],
               loc="upper right", bbox_to_anchor=(0.965, 0.88), fontsize=9)
    return export_figure(fig, output_dir, "03_scenario_robustness")
