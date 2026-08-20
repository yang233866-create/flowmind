from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, Patch

from .design_system import (
    GRID, INK, MUTED, OBSERVED_COLOR, INFERRED_COLOR,
    add_figure_title, export_figure, panel_label,
)

FIGURE_TITLE = "方向交通需求、车辆组成与数据来源"
OBSERVED_LINESTYLE = "-"
INFERRED_LINESTYLE = (0, (5, 3))
DIRECTIONS = ["north", "south", "east", "west"]
DIRECTION_LABELS = {"north": "北向", "south": "南向", "east": "东向", "west": "西向"}
CLASSES = ["car", "bus", "truck", "motorcycle"]
CLASS_LABELS = {"car": "小客车", "bus": "公交车", "truck": "货车", "motorcycle": "摩托车"}
CLASS_COLORS = {"car": "#2563EB", "bus": "#0EA5A4", "truck": "#E59A36", "motorcycle": "#7C3AED"}


def _curve(ax, start, end, width, color, observed):
    x0, y0 = start
    x1, y1 = end
    path = Path([(x0, y0), (x0 + 0.27, y0), (x1 - 0.27, y1), (x1, y1)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=width,
                           alpha=0.44 if observed else 0.28,
                           linestyle=OBSERVED_LINESTYLE if observed else INFERRED_LINESTYLE,
                           capstyle="round", zorder=1))


def _add_node(
    ax, gid, x, y, width, height, label, value, facecolor, edgecolor,
    name_color, value_color, name_size,
):
    rectangle = Rectangle(
        (x, y - height / 2), width, height,
        facecolor=facecolor, edgecolor=edgecolor,
        lw=1.6 if edgecolor != "none" else 0,
        alpha=0.90 if facecolor != "white" else 1,
        gid=f"{gid}-box",
    )
    ax.add_patch(rectangle)
    name = ax.text(
        x + width / 2, y, label,
        ha="center", va="center", fontsize=name_size, color=name_color,
        weight="bold", gid=f"{gid}-name",
    )
    value_text = ax.text(
        x + width / 2, y - height / 2 - 0.018, value,
        ha="center", va="top", fontsize=8, color=value_color,
        gid=f"{gid}-value",
    )
    return rectangle, name, value_text


def build(data, output_dir, source_dir):
    approaches = data.traffic_state["approaches"]
    rows = []
    for direction in DIRECTIONS:
        approach = approaches[direction]
        for vehicle_class in CLASSES:
            share = float(approach["vehicle_mix"].get(vehicle_class, 0.0))
            rows.append({
                "direction": direction, "vehicle_class": vehicle_class,
                "flow_vph": float(approach["flow_vph"]), "share": share,
                "class_flow_vph": float(approach["flow_vph"]) * share,
                "observed": bool(approach["observed"]),
            })
    flow = pd.DataFrame(rows)
    flow.to_csv(source_dir / "11_direction_vehicle_class_flow.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.4), facecolor="#FFFFFF")
    gs = fig.add_gridspec(1, 2, left=0.055, right=0.97, bottom=0.22, top=0.82,
                          width_ratios=[1.08, 0.92], wspace=0.20)
    ax_flow = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    add_figure_title(fig, FIGURE_TITLE)

    panel_label(ax_flow, "A", "方向交通流与车辆类型组成")
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    direction_y = {"north": 0.82, "south": 0.61, "east": 0.36, "west": 0.15}
    class_y = {"car": 0.80, "bus": 0.58, "truck": 0.36, "motorcycle": 0.14}
    for direction in DIRECTIONS:
        observed = bool(approaches[direction]["observed"])
        edge = OBSERVED_COLOR if observed else INFERRED_COLOR
        y = direction_y[direction]
        _add_node(
            ax_flow, f"direction-{direction}", 0.005, y, 0.25, 0.10,
            DIRECTION_LABELS[direction],
            f"{approaches[direction]['flow_vph']:,.1f} veh/h",
            "white", edge, INK, MUTED, 9.8,
        )
    for vehicle_class in CLASSES:
        y = class_y[vehicle_class]
        total = flow.query("vehicle_class == @vehicle_class")["class_flow_vph"].sum()
        _add_node(
            ax_flow, f"class-{vehicle_class}", 0.76, y, 0.22, 0.10,
            CLASS_LABELS[vehicle_class], f"{total:,.0f} veh/h",
            CLASS_COLORS[vehicle_class], "none", "white",
            CLASS_COLORS[vehicle_class], 9.3,
        )
    maximum = max(float(flow["class_flow_vph"].max()), 1.0)
    for row in flow.itertuples(index=False):
        if row.class_flow_vph > 0:
            _curve(ax_flow, (0.25, direction_y[row.direction]), (0.78, class_y[row.vehicle_class]),
                   0.7 + 7.2 * row.class_flow_vph / maximum,
                   CLASS_COLORS[row.vehicle_class], bool(row.observed))

    panel_label(ax_bar, "B", "各方向车辆类型组成")
    left = [0.0] * len(DIRECTIONS)
    for vehicle_class in CLASSES:
        values = [float(flow.query("direction == @direction and vehicle_class == @vehicle_class")["share"].iloc[0]) * 100
                  for direction in DIRECTIONS]
        bars = ax_bar.barh(range(4), values, left=left, color=CLASS_COLORS[vehicle_class],
                           height=0.54, label=CLASS_LABELS[vehicle_class],
                           edgecolor="white", linewidth=0.7, alpha=0.92)
        for index, bar in enumerate(bars):
            if not approaches[DIRECTIONS[index]]["observed"]:
                bar.set_hatch("//")
                bar.set_edgecolor("white")
                bar.set_linewidth(0.9)
        left = [a + b for a, b in zip(left, values)]
    ax_bar.set_yticks(range(4), [DIRECTION_LABELS[d] for d in DIRECTIONS])
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("车辆类型占比（%）")
    ax_bar.grid(axis="x", color=GRID, lw=0.7, alpha=0.30)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(length=0, labelsize=9, colors=MUTED)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax_bar.spines[side].set_color(GRID)
        ax_bar.spines[side].set_linewidth(0.7)

    class_handles = [Patch(facecolor=CLASS_COLORS[c], label=CLASS_LABELS[c]) for c in CLASSES]
    source_handles = [
        Line2D([0], [0], color=OBSERVED_COLOR, lw=2, ls=OBSERVED_LINESTYLE, label="视觉观测"),
        Line2D([0], [0], color=INFERRED_COLOR, lw=2, ls=INFERRED_LINESTYLE, label="工程回退"),
    ]
    fig.legend(handles=class_handles, loc="lower center", bbox_to_anchor=(0.68, 0.07), ncol=4, fontsize=9)
    fig.legend(handles=source_handles, loc="lower center", bbox_to_anchor=(0.50, 0.015), ncol=2, fontsize=9)
    return export_figure(fig, output_dir, "11_perception_composition_flow")
