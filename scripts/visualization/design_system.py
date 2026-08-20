from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findSystemFonts
from matplotlib.patches import FancyBboxPatch


BACKGROUND = "#FFFFFF"
SURFACE = "#FFFFFF"
INK = "#1F2933"
MUTED = "#64748B"
GRID = "#DCE3EA"
POSITIVE = "#0EA5A4"
RISK = "#C94B50"
AMBER = "#E59A36"
OBSERVED_COLOR = "#0EA5A4"
INFERRED_COLOR = "#E59A36"

MAIN_TITLE_SIZE = 17
PANEL_TITLE_SIZE = 11.5
PANEL_LETTER_SIZE = 11.5
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 9
HEATMAP_VALUE_SIZE = 10

STRATEGY_COLORS = {
    "fixed": "#8B98A8",
    "actuated": "#1E9E8F",
    "dqn": "#3569D4",
    "ppo": "#E46F51",
}
STRATEGY_LABELS = {
    "fixed": "Fixed 定时",
    "actuated": "Actuated 感应",
    "dqn": "DQN",
    "ppo": "PPO",
}
SCENARIO_LABELS = {
    "normal": "常态",
    "morning_peak": "早高峰",
    "evening_peak": "晚高峰",
    "event_surge": "活动激增",
    "lane_closure": "车道封闭",
}
FIGURE_STEMS = (
    "01_vision_to_twin",
    "02_strategy_tradeoffs",
    "03_scenario_robustness",
    "04_queue_dynamics",
    "05_training_evidence",
    "06_decision_map",
    "07_regret_landscape",
    "08_paired_transitions",
    "09_operating_state_density",
    "10_scenario_timeline_atlas",
    "11_perception_composition_flow",
)


def _font_family() -> str:
    candidates = ["Noto Sans SC", "Microsoft YaHei UI", "Microsoft YaHei", "SimHei"]
    installed = {Path(path).stem.lower(): path for path in findSystemFonts()}
    for candidate in candidates:
        key = candidate.replace(" ", "").lower()
        for stem, path in installed.items():
            if key in stem.replace(" ", ""):
                return FontProperties(fname=path).get_name()
    return "DejaVu Sans"


FONT_FAMILY = _font_family()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 9,
            "axes.unicode_minus": False,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.30,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": LEGEND_SIZE,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": BACKGROUND,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.16,
        }
    )


def add_figure_title(fig, title: str, subtitle: str | None = None, kicker: str | None = None) -> None:
    """Add only a neutral scientific figure title.

    subtitle and kicker remain optional for source compatibility but are deliberately
    not rendered in the paper figure suite.
    """
    fig.text(
        0.055,
        0.965,
        title,
        fontsize=MAIN_TITLE_SIZE,
        color=INK,
        weight="bold",
        ha="left",
        va="top",
    )


def panel_label(ax, label: str, title: str) -> None:
    y = 1.025
    ax.text(
        0.0,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_SIZE,
        weight="bold",
        color=POSITIVE,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.075,
        y,
        title,
        transform=ax.transAxes,
        fontsize=PANEL_TITLE_SIZE,
        weight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(False)
    if grid_axis in {"x", "y", "both"}:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.30, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(labelsize=TICK_LABEL_SIZE, length=0, colors=MUTED)


def rounded_card(ax, xy, width, height, facecolor=SURFACE, edgecolor=GRID, radius=0.02, alpha=1.0):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        alpha=alpha,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def strategy_handles(strategies=("fixed", "actuated", "dqn", "ppo")):
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
            markersize=6,
            linewidth=1.8,
        )
        for strategy in strategies
    ]


def export_figure(fig, output_dir: Path, stem: str, png_dpi: int = 600) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix, kwargs in (
        ("svg", {}),
        ("png", {"dpi": png_dpi}),
        ("pdf", {}),
    ):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, format=suffix, facecolor=fig.get_facecolor(), **kwargs)
        paths[suffix] = str(path)
    plt.close(fig)
    return paths


def write_manifest(output_dir: Path, stems=FIGURE_STEMS) -> Path:
    from PIL import Image

    records = []
    for stem in stems:
        record = {"stem": stem, "files": {}}
        for suffix in ("png", "svg", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            payload = path.read_bytes()
            item = {
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            if suffix == "png":
                with Image.open(path) as image:
                    item["pixels"] = [image.width, image.height]
                    item["dpi"] = list(image.info.get("dpi", ()))
            record["files"][suffix] = item
        records.append(record)
    manifest = output_dir.parent / "figure_manifest.json"
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


configure_matplotlib()
