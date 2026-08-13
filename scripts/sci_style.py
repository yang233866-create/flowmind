"""SCI figure style: Times New Roman, DPI>=300, Nature/NPG palette.

Usage:
    from scripts.sci_style import apply_style, NPG, save_fig
    apply_style()
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

NPG = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]

STRATEGY_COLORS = {
    "fixed": "#3C5488",
    "actuated": "#4DBBD5",
    "dqn": "#E64B35",
    "ppo": "#00A087",
}

STRATEGY_LABELS = {
    "fixed": "Fixed-Time",
    "actuated": "Actuated",
    "dqn": "DQN",
    "ppo": "PPO",
}

DIRECTION_COLORS = {
    "north": "#3C5488",
    "south": "#4DBBD5",
    "east": "#E64B35",
    "west": "#00A087",
}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=NPG),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    })


def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=max(dpi, 300))
    plt.close(fig)
    return path
