"""Render FlowMind SUMO templates as paper figures (lane-level geometry).

Usage: python -m scripts.plot_networks
Reads simulation/templates/*/net.net.xml via sumolib; draws lanes, colors
by allowed direction, marks the traffic light.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.sci_style import NPG, apply_style, save_fig
from simulation.sumo_home import ensure_sumo_home

TEMPLATES = ["cross_basic", "cross_leftturn", "arterial_minor"]
TITLES = {
    "cross_basic": "Standard cross (2 lanes)",
    "cross_leftturn": "Protected left turn (3 lanes)",
    "arterial_minor": "Arterial x minor road",
}


def draw_template(ax, tdir: Path) -> bool:
    import sumolib

    net_path = tdir / "net.net.xml"
    if not net_path.exists():
        return False
    net = sumolib.net.readNet(str(net_path), withInternal=True)

    for edge in net.getEdges(withInternal=True):
        internal = edge.getID().startswith(":")
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            xs, ys = zip(*shape)
            if internal:
                ax.plot(xs, ys, color="#BBBBBB", linewidth=0.7, alpha=0.6, zorder=1)
            else:
                ax.plot(xs, ys, color=NPG[3], linewidth=1.6, alpha=0.85, zorder=2,
                        solid_capstyle="round")

    for node in net.getNodes():
        if node.getType().startswith("traffic_light"):
            x, y = node.getCoord()
            ax.scatter([x], [y], s=60, marker="o", facecolor=NPG[0],
                       edgecolor="white", linewidth=1.0, zorder=5)

    meta_path = tdir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        offsets = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
        xs = [n.getCoord()[0] for n in net.getNodes()]
        ys = [n.getCoord()[1] for n in net.getNodes()]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys)) / 2
        for app, (dx, dy) in offsets.items():
            if app in meta.get("approaches", {}):
                n_lanes = meta["approaches"][app]["n_lanes"]
                ax.text(cx + dx * span * 1.12, cy + dy * span * 1.12,
                        f"{app.upper()[0]}\n{n_lanes} ln", ha="center", va="center",
                        fontsize=7.5, color="#333333")

    ax.set_aspect("equal")
    ax.axis("off")
    return True


def main() -> None:
    ensure_sumo_home()
    apply_style()
    root = Path("simulation/templates")
    available = [t for t in TEMPLATES if (root / t / "net.net.xml").exists()]
    if not available:
        print("No built templates found; run template build.py first.")
        return
    fig, axes = plt.subplots(1, len(available), figsize=(3.1 * len(available), 3.4))
    if len(available) == 1:
        axes = [axes]
    for ax, name in zip(axes, available):
        draw_template(ax, root / name)
        ax.set_title(TITLES.get(name, name), fontsize=10)
    fig.suptitle("SUMO intersection templates (traffic light marked in red)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    out = save_fig(fig, "figures/fig_sumo_networks.png")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
