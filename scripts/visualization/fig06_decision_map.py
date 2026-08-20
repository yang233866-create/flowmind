from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from .design_system import (
    GRID, INK, MUTED, SCENARIO_LABELS, STRATEGY_COLORS, STRATEGY_LABELS,
    add_figure_title, export_figure, panel_label, style_axis,
)
from .statistics import metric_winners

FIGURE_TITLE = "不同场景与指标下的最优策略分布"
METRIC_ORDER = ["avg_waiting_s", "avg_travel_time_s", "throughput_veh", "avg_speed_mps", "avg_queue_veh", "max_queue_veh"]
METRIC_LABELS = {"avg_waiting_s": "等待", "avg_travel_time_s": "行程", "throughput_veh": "吞吐",
                 "avg_speed_mps": "速度", "avg_queue_veh": "平均排队", "max_queue_veh": "最大排队"}
OBJECTIVE = {"avg_waiting_s": "效率目标", "avg_travel_time_s": "效率目标", "throughput_veh": "效率目标",
             "avg_speed_mps": "效率目标", "avg_queue_veh": "控堵目标", "max_queue_veh": "控堵目标"}
SCENARIO_ORDER = ["normal", "morning_peak", "evening_peak", "event_surge", "lane_closure"]
STRATEGY_ORDER = ["fixed", "actuated", "dqn", "ppo"]
OBJECTIVE_ORDER = ["效率目标", "控堵目标"]
OBJECTIVE_TINTS = {"效率目标": "#FFF3DB", "控堵目标": "#E8F3FA"}
OBJECTIVE_EDGES = {"效率目标": "#D6A53B", "控堵目标": "#5B8DB8"}


def build(data, output_dir, source_dir):
    winners = metric_winners(data.arena)
    winners.to_csv(source_dir / "06_metric_winners_and_regret.csv", index=False)
    winner_rows = winners.query("winner").copy()
    winner_rows["objective"] = winner_rows["metric"].map(OBJECTIVE)
    counts = winner_rows.groupby(["objective", "strategy"], observed=True).size().rename("winner_count").reset_index()
    counts.to_csv(source_dir / "06_objective_strategy_counts.csv", index=False)

    count_table = (
        counts.pivot(index="objective", columns="strategy", values="winner_count")
        .reindex(index=OBJECTIVE_ORDER, columns=STRATEGY_ORDER, fill_value=0)
        .fillna(0)
        .astype(int)
    )

    fig = plt.figure(figsize=(7.2, 4.35), facecolor="#FFFFFF")
    gs = fig.add_gridspec(1, 2, left=0.12, right=0.975, bottom=0.20, top=0.79,
                          width_ratios=[0.95, 1.18], wspace=0.32)
    ax_counts = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[0, 1])
    add_figure_title(fig, FIGURE_TITLE)

    # A — exact winner counts, shown as aligned horizontal stacked bars.
    panel_label(ax_counts, "A", "目标类别的赢家计数")
    y_positions = np.array([1.0, 0.0])
    left = np.zeros(len(OBJECTIVE_ORDER), dtype=float)
    for strategy in STRATEGY_ORDER:
        values = count_table[strategy].to_numpy(dtype=float)
        bars = ax_counts.barh(
            y_positions,
            values,
            left=left,
            height=0.34,
            color=STRATEGY_COLORS[strategy],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            if value <= 0:
                continue
            label = f"{STRATEGY_LABELS[strategy].split()[0]} {int(value)}" if value >= 4 else f"{int(value)}"
            ax_counts.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                label,
                ha="center",
                va="center",
                fontsize=8.8 if value >= 4 else 8.5,
                color="white",
                weight="bold",
                zorder=4,
            )
        left += values

    totals = count_table.sum(axis=1).astype(int).tolist()
    y_labels = [f"{objective}   n={total}" for objective, total in zip(OBJECTIVE_ORDER, totals)]
    ax_counts.set_yticks(y_positions, y_labels)
    for tick, objective in zip(ax_counts.get_yticklabels(), OBJECTIVE_ORDER):
        tick.set_color(INK)
        tick.set_fontweight("bold")
        tick.set_bbox(dict(
            boxstyle="round,pad=0.28,rounding_size=0.12",
            facecolor=OBJECTIVE_TINTS[objective],
            edgecolor=OBJECTIVE_EDGES[objective],
            linewidth=0.8,
        ))
    ax_counts.tick_params(axis="y", pad=9)
    ax_counts.set_xlim(0, 21)
    ax_counts.set_ylim(-0.68, 1.62)
    ax_counts.set_xticks([0, 5, 10, 15, 20])
    ax_counts.set_xlabel("赢家单元格数")
    ax_counts.text(0.0, -0.53, "Fixed：0 个赢家", fontsize=9, color=MUTED, va="center")
    style_axis(ax_counts, "x")
    ax_counts.spines["left"].set_visible(False)

    # B — the same 5 × 6 winner matrix, encoded once by the shared strategy colors.
    panel_label(ax_matrix, "B", "五场景 × 六指标最优策略")
    lookup = winner_rows.set_index(["scenario", "metric"])["strategy"]
    matrix = np.zeros((len(SCENARIO_ORDER), len(METRIC_ORDER)), dtype=int)
    for i, scenario in enumerate(SCENARIO_ORDER):
        for j, metric in enumerate(METRIC_ORDER):
            strategy = lookup.loc[(scenario, metric)]
            matrix[i, j] = STRATEGY_ORDER.index(strategy)
            ax_matrix.add_patch(Rectangle(
                (j + 0.035, i + 0.035),
                0.93,
                0.93,
                facecolor=STRATEGY_COLORS[strategy],
                edgecolor="white",
                linewidth=0.8,
            ))

    ax_matrix.set_xlim(0, len(METRIC_ORDER))
    ax_matrix.set_ylim(len(SCENARIO_ORDER), 0)
    ax_matrix.set_aspect("equal")
    ax_matrix.set_xticks(np.arange(6) + 0.5, [METRIC_LABELS[m] for m in METRIC_ORDER], rotation=26, ha="right")
    ax_matrix.set_yticks(np.arange(5) + 0.5, [SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax_matrix.tick_params(length=0, labelsize=9.2, colors=MUTED)
    for index, tick in enumerate(ax_matrix.get_xticklabels()):
        tick.set_color("#9A711C" if index < 4 else "#3E759D")
        tick.set_fontweight("bold")
    ax_matrix.axvline(4, color=INK, lw=1.25, alpha=0.55)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    fig.legend(
        handles=[Patch(facecolor=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s]) for s in STRATEGY_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.58, 0.025),
        ncol=4,
        fontsize=9,
        columnspacing=1.55,
        handlelength=1.6,
    )
    return export_figure(fig, output_dir, "06_decision_map")