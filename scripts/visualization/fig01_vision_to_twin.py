from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

from collections.abc import Mapping
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from .design_system import (
    INK, OBSERVED_COLOR, INFERRED_COLOR, SURFACE,
    add_figure_title, export_figure, panel_label, style_axis,
)

FIGURE_TITLE = "视觉感知与方向交通流输入构建"
OBSERVED_LINESTYLE = "-"
INFERRED_LINESTYLE = (0, (5, 3))
DIRECTION_CN = {"north": "北向", "south": "南向", "east": "东向", "west": "西向"}


def crop_same_frame(
    image: np.ndarray,
    metadata: Mapping[str, object],
) -> np.ndarray | None:
    zoom = metadata.get("zoom")
    if zoom is None:
        return None
    x1, y1, x2, y2 = map(int, zoom["crop_xyxy"])
    return image[y1:y2, x1:x2].copy()


def draw_frame_evidence(
    ax,
    image: np.ndarray,
    metadata: Mapping[str, object],
):
    ax.imshow(image)
    ax.set_axis_off()
    title = "计数线布设示例" if metadata["no_detections"] else "视频检测与跟踪示例"
    panel_label(ax, "A", title)

    timestamp = metadata["frame"]["timestamp_sec"]
    ax.text(
        0.02,
        0.02,
        f"自动代表帧 · t = {timestamp:.2f} s",
        transform=ax.transAxes,
        fontsize=8,
        color="white",
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "#111827",
            "alpha": 0.72,
            "edgecolor": "none",
        },
    )

    crop = crop_same_frame(image, metadata)
    if crop is None:
        return None

    zoom = metadata["zoom"]
    x1, y1, x2, y2 = map(int, zoom["crop_xyxy"])
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="white",
            linewidth=1.2,
        )
    )
    inset = ax.inset_axes([0.54, 0.04, 0.43, 0.34])
    inset.imshow(crop)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title(f"轨迹 ID {zoom['track_id']} · 同帧局部", fontsize=8)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(1.2)
    return inset


def build(data, output_dir, source_dir):
    state = data.traffic_state
    approaches = state["approaches"]
    rows = []
    for direction, values in state["flow_profile"].items():
        for index, value in enumerate(values):
            rows.append({"direction": direction, "start_s": index * state["profile_bins_sec"],
                         "flow_vph": value, "observed": approaches[direction]["observed"]})
    pd.DataFrame(rows).to_csv(source_dir / "01_flow_profile.csv", index=False)
    (source_dir / "01_vision_frame_provenance.json").write_text(
        json.dumps(data.annotated_frame_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fig = plt.figure(figsize=(7.2, 4.4), facecolor="#FFFFFF")
    gs = fig.add_gridspec(2, 2, left=0.045, right=0.975, bottom=0.085, top=0.80,
                          width_ratios=[1.12, 1.0], height_ratios=[1.05, 0.95],
                          hspace=0.32, wspace=0.20)
    ax_image = fig.add_subplot(gs[:, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_profile = fig.add_subplot(gs[1, 1])
    add_figure_title(fig, FIGURE_TITLE)

    image = plt.imread(data.annotated_frame)
    draw_frame_evidence(ax_image, image, data.annotated_frame_meta)

    panel_label(ax_map, "B", "各方向流量与数据来源")
    ax_map.set_xlim(-1.55, 1.55)
    ax_map.set_ylim(-1.22, 1.22)
    ax_map.set_aspect("equal")
    ax_map.axis("off")
    road = "#E5E9EE"
    ax_map.add_patch(Rectangle((-0.25, -0.78), 0.50, 1.56, facecolor=road, edgecolor="none"))
    ax_map.add_patch(Rectangle((-0.90, -0.25), 1.80, 0.50, facecolor=road, edgecolor="none"))
    ax_map.add_patch(Rectangle((-0.25, -0.25), 0.50, 0.50, facecolor="#D5DCE4", edgecolor="none"))
    ax_map.plot([0, 0], [-0.78, -0.32], color="white", lw=1.1, ls=(0, (4, 4)))
    ax_map.plot([0, 0], [0.32, 0.78], color="white", lw=1.1, ls=(0, (4, 4)))
    ax_map.plot([-0.90, -0.32], [0, 0], color="white", lw=1.1, ls=(0, (4, 4)))
    ax_map.plot([0.32, 0.90], [0, 0], color="white", lw=1.1, ls=(0, (4, 4)))

    placements = {
        "north": ((0.10, 0.74), (0.10, 0.30), (0.0, 1.02), "center"),
        "south": ((-0.10, -0.74), (-0.10, -0.30), (0.0, -1.02), "center"),
        "east": ((0.94, -0.10), (0.34, -0.10), (1.06, 0.30), "left"),
        "west": ((-0.94, 0.10), (-0.34, 0.10), (-1.06, -0.30), "right"),
    }
    for direction, (start, end, text_xy, ha) in placements.items():
        observed = bool(approaches[direction]["observed"])
        color = OBSERVED_COLOR if observed else INFERRED_COLOR
        linestyle = OBSERVED_LINESTYLE if observed else INFERRED_LINESTYLE
        ax_map.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                         lw=2.4, color=color, linestyle=linestyle))
        ax_map.text(text_xy[0], text_xy[1],
                    f"{DIRECTION_CN[direction]}\n{approaches[direction]['flow_vph']:,.1f} veh/h",
                    ha=ha, va="center", fontsize=9, linespacing=1.25, color=INK)
    fig.legend(handles=[
        Line2D([0], [0], color=OBSERVED_COLOR, lw=2.2, ls=OBSERVED_LINESTYLE, label="视觉观测"),
        Line2D([0], [0], color=INFERRED_COLOR, lw=2.2, ls=INFERRED_LINESTYLE, label="工程回退"),
    ], loc="upper center", bbox_to_anchor=(0.79, 0.93), ncol=2, fontsize=9)

    panel_label(ax_profile, "C", "5 秒窗口方向流量")
    times = np.arange(5) * state["profile_bins_sec"]
    profile_colors = {"north": OBSERVED_COLOR, "south": "#2563EB"}
    for direction in ("north", "south"):
        values = np.asarray(state["flow_profile"][direction], dtype=float)
        ax_profile.step(times, values, where="post", lw=2.0, color=profile_colors[direction], label=DIRECTION_CN[direction])
        ax_profile.scatter(times, values, s=20, color=profile_colors[direction],
                           edgecolor=SURFACE, linewidth=0.7, zorder=3)
    style_axis(ax_profile, "y")
    ax_profile.set_xlim(0, 20)
    ax_profile.set_ylim(0, 3200)
    ax_profile.set_xticks([0, 5, 10, 15, 20])
    ax_profile.set_xlabel("视频时间（秒）")
    ax_profile.set_ylabel("折算流量（veh/h）")
    ax_profile.legend(loc="upper right", ncol=2, fontsize=9)
    return export_figure(fig, output_dir, "01_vision_to_twin")
