from __future__ import annotations

# Shared publication contract: font.family: sans-serif; svg.fonttype='none'; pdf.fonttype=42.
# design_system exports .svg, .pdf and .png with dpi=600; final width is 183 mm.
fig_width_mm = 183

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .design_system import (
    GRID,
    INK,
    MUTED,
    POSITIVE,
    STRATEGY_COLORS,
    export_figure,
)


FIGURE_TITLE = "DQN 与 PPO 训练过程"
PANEL_TITLES = {
    "A": "DQN 平均回合奖励",
    "B": "PPO 平均回合奖励",
    "C": "DQN 训练损失",
    "D": "PPO 价值损失",
}
REWARD_YLIM = (-24, -8)
REWARD_YTICKS = (-24, -20, -16, -12, -8)
TRAINING_XLIM = (0, 102)
TRAINING_XTICKS = (0, 20, 40, 60, 80, 100)
PNG_DPI = 600
RAW_LINEWIDTH = 0.9
RAW_ALPHA = 0.25
SMOOTH_LINEWIDTH = 2.4


def _smooth(frame, span):
    result = frame.copy()
    result["smooth"] = result["value"].ewm(span=span, adjust=False).mean()
    return result


def _style_axis(ax):
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="y", color="#E4E9EF", linewidth=0.7, alpha=0.35)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D7DEE6")
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(axis="both", colors="#5B6673", labelsize=9.2, length=0)
    ax.set_xlim(*TRAINING_XLIM)
    ax.set_xticks(TRAINING_XTICKS)
    ax.set_xlabel("训练步数（千步）", fontsize=10.0, color=INK, labelpad=7)


def _panel_title(ax, letter, title):
    ax.text(
        0.0,
        1.045,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        weight="bold",
        color=POSITIVE,
    )
    ax.text(
        0.055,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.3,
        weight="bold",
        color="#263238",
    )


def _plot_curve(ax, frame, color, ylabel, letter):
    smooth = _smooth(frame, max(5, len(frame) // 10))
    x = frame["step"].to_numpy(dtype=float) / 1000
    raw = frame["value"].to_numpy(dtype=float)
    trend = smooth["smooth"].to_numpy(dtype=float)

    ax.plot(
        x,
        raw,
        color=color,
        linewidth=RAW_LINEWIDTH,
        alpha=RAW_ALPHA,
        zorder=1,
    )
    ax.plot(
        x,
        trend,
        color=color,
        linewidth=SMOOTH_LINEWIDTH,
        alpha=1.0,
        zorder=2,
    )
    ax.scatter(
        [x[0], x[-1]],
        [raw[0], raw[-1]],
        color=color,
        s=22,
        edgecolor="#FFFFFF",
        linewidth=0.6,
        zorder=3,
    )
    _style_axis(ax)
    _panel_title(ax, letter, PANEL_TITLES[letter])
    ax.set_ylabel(ylabel, fontsize=10.0, color=INK, labelpad=7)
    return raw, trend


def _loss_limits(values):
    lower = min(0.0, float(np.min(values)))
    upper = float(np.max(values))
    padding = max((upper - lower) * 0.08, upper * 0.05, 0.01)
    return lower, upper + padding


def build(data, output_dir, source_dir):
    for algorithm, curves in data.training.items():
        for name, frame in curves.items():
            _smooth(frame, max(5, len(frame) // 10)).to_csv(
                source_dir / f"05_{algorithm}_{name}.csv", index=False
            )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.4, 5.6),
        facecolor="#FFFFFF",
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.845,
        hspace=0.47,
        wspace=0.28,
    )
    fig.suptitle(
        FIGURE_TITLE,
        x=0.105,
        y=0.965,
        ha="left",
        va="top",
        fontsize=17,
        weight="bold",
        color="#20262E",
    )

    _plot_curve(
        axes[0, 0],
        data.training["dqn"]["reward"],
        STRATEGY_COLORS["dqn"],
        "平均回合奖励",
        "A",
    )
    _plot_curve(
        axes[0, 1],
        data.training["ppo"]["reward"],
        STRATEGY_COLORS["ppo"],
        "平均回合奖励",
        "B",
    )
    axes[0, 0].set_ylim(*REWARD_YLIM)
    axes[0, 1].set_ylim(*REWARD_YLIM)
    axes[0, 0].set_yticks(REWARD_YTICKS)
    axes[0, 1].set_yticks(REWARD_YTICKS)

    dqn_loss, _ = _plot_curve(
        axes[1, 0],
        data.training["dqn"]["loss"],
        STRATEGY_COLORS["dqn"],
        "训练损失",
        "C",
    )
    ppo_loss, _ = _plot_curve(
        axes[1, 1],
        data.training["ppo"]["loss"],
        STRATEGY_COLORS["ppo"],
        "价值损失",
        "D",
    )
    axes[1, 0].set_ylim(*_loss_limits(dqn_loss))
    axes[1, 1].set_ylim(*_loss_limits(ppo_loss))

    return export_figure(
        fig,
        output_dir,
        "05_training_evidence",
        png_dpi=PNG_DPI,
    )
