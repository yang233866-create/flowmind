"""Draw the FlowMind system architecture figure (paper-quality, SCI style).

Usage: python -m scripts.make_architecture_figure
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.sci_style import NPG, apply_style, save_fig

LAYERS = [
    ("Video Perception", "YOLO11 detection + ByteTrack tracking\nROI / counting-line statistics", NPG[3]),
    ("Traffic State", "TrafficState JSON (schema 1.1)\nflows / mix / turning / profile", NPG[1]),
    ("Scenario Mapping", "Scenario Generator\nroutes, flows, What-if modifiers", NPG[2]),
    ("Traffic Simulation", "Eclipse SUMO digital twin\n3 intersection templates", NPG[4]),
    ("Signal Optimization", "Fixed-Time / Actuated /\nDQN / PPO (SB3 + sumo-rl)", NPG[0]),
    ("Experiment & Decision", "Strategy Arena + Scenario Lab\nunified metrics, ranking, export", NPG[5]),
]

SIDE_NOTES = {
    1: "single cross-module interface",
    4: "identical metric pipeline for all strategies",
}


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(LAYERS) * 1.7 + 0.4)
    ax.axis("off")

    box_w, box_h, x0 = 6.4, 1.24, 1.1
    centers = []
    for i, (title, desc, color) in enumerate(LAYERS):
        y = (len(LAYERS) - 1 - i) * 1.7 + 0.5
        centers.append((x0 + box_w / 2, y + box_h / 2))
        ax.add_patch(FancyBboxPatch(
            (x0, y), box_w, box_h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=color, edgecolor="none", alpha=0.14,
        ))
        ax.add_patch(FancyBboxPatch(
            (x0, y), 0.14, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color, edgecolor="none",
        ))
        ax.text(x0 + 0.38, y + box_h - 0.24, f"{i + 1}. {title}",
                fontsize=10.5, fontweight="bold", va="top", color="#222222")
        ax.text(x0 + 0.38, y + 0.16, desc, fontsize=8.2, va="bottom", color="#444444")
        if i in SIDE_NOTES:
            ax.annotate(
                SIDE_NOTES[i],
                xy=(x0 + box_w + 0.15, y + box_h / 2),
                fontsize=7.5, style="italic", color="#666666",
                va="center", rotation=90,
            )

    for (x1, y1), (_, y2) in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch(
            (x1, y1 - box_h / 2 - 0.05), (x1, y2 + box_h / 2 + 0.05),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.2, color="#555555",
        ))

    # Feedback loop: Experiment layer -> Scenario Mapping (What-if replays)
    x_loop = x0 - 0.55
    y_top, y_bot = centers[2][1], centers[5][1]
    ax.add_patch(FancyArrowPatch(
        (x0 - 0.06, y_bot), (x0 - 0.06, y_top),
        connectionstyle=f"arc3,rad=0.0",
        arrowstyle="-|>", mutation_scale=12, linewidth=1.0,
        color=NPG[2], linestyle=(0, (4, 2)),
        path_effects=None,
    ))
    ax.plot([x_loop, x0 - 0.06], [y_bot, y_bot], color=NPG[2], linewidth=1.0,
            linestyle=(0, (4, 2)))
    ax.plot([x_loop, x_loop], [y_bot, y_top], color=NPG[2], linewidth=1.0,
            linestyle=(0, (4, 2)))
    ax.plot([x_loop, x0 - 0.06], [y_top, y_top], color=NPG[2], linewidth=1.0,
            linestyle=(0, (4, 2)))
    ax.text(x_loop - 0.18, (y_top + y_bot) / 2, "What-if scenario replay",
            fontsize=7.5, rotation=90, va="center", ha="center", color=NPG[2])

    ax.set_title("FlowMind AI: Perception-to-Decision Closed Loop", fontsize=12, pad=12)
    out = save_fig(fig, "figures/fig_architecture.png")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
