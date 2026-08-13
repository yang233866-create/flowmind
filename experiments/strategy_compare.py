"""Batch-run strategies across scenarios and produce comparison figures.

CLI:
    python -m experiments.strategy_compare --scenarios all \
        --strategies fixed,actuated --seeds 3 [--template cross_basic]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.scenarios import SCENARIOS
from experiments.scenario_runner import RESULTS_ROOT, run_experiment

ARENA_CSV = Path("data/results/arena_summary.csv")

METRIC_LABELS = {
    "avg_waiting_s": "Avg waiting time (s)",
    "avg_travel_time_s": "Avg travel time (s)",
    "avg_queue_veh": "Avg queue (veh)",
    "max_queue_veh": "Max queue (veh)",
    "throughput_veh": "Throughput (veh)",
    "avg_speed_mps": "Avg speed (m/s)",
}
# Metrics where lower is better (for ranking / improvement signs)
LOWER_BETTER = {"avg_waiting_s", "avg_travel_time_s", "avg_queue_veh", "max_queue_veh", "teleports"}


def collect_summary() -> pd.DataFrame:
    """Scan all experiment dirs and rebuild the arena summary CSV."""
    rows = []
    if RESULTS_ROOT.exists():
        for exp_dir in sorted(RESULTS_ROOT.iterdir()):
            summary = exp_dir / "metrics_summary.json"
            if not summary.exists() or "__" not in exp_dir.name:
                continue
            parts = exp_dir.name.split("__")
            if len(parts) != 3 or not parts[2].startswith("s"):
                continue
            scenario, strategy, seed_tag = parts
            try:
                seed = int(seed_tag[1:])
            except ValueError:
                continue
            metrics = json.loads(summary.read_text(encoding="utf-8"))
            rows.append({"scenario": scenario, "strategy": strategy, "seed": seed,
                         "exp_id": exp_dir.name, **metrics})
    df = pd.DataFrame(rows)
    if not df.empty:
        ARENA_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ARENA_CSV, index=False)
    return df


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std over seeds per (scenario, strategy)."""
    metrics = [c for c in METRIC_LABELS if c in df.columns]
    return df.groupby(["scenario", "strategy"])[metrics].agg(["mean", "std"])


def make_figures(df: pd.DataFrame, fig_dir: Path = Path("figures")) -> list[Path]:
    import matplotlib.pyplot as plt

    from scripts.sci_style import STRATEGY_COLORS, STRATEGY_LABELS, apply_style, save_fig

    apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    strategies = [s for s in ("fixed", "actuated", "dqn", "ppo") if s in df["strategy"].unique()]
    scenarios = [s for s in list(SCENARIOS) if s in df["scenario"].unique()] or sorted(
        df["scenario"].unique()
    )

    # 1. Grouped bars: one panel per metric, strategies side by side per scenario
    panel_metrics = [m for m in ("avg_waiting_s", "avg_queue_veh", "avg_travel_time_s",
                                 "throughput_veh") if m in df.columns]
    if panel_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.6))
        for ax, metric in zip(axes.flat, panel_metrics):
            width = 0.8 / len(strategies)
            x = np.arange(len(scenarios))
            for i, strat in enumerate(strategies):
                sub = df[df["strategy"] == strat].groupby("scenario")[metric]
                mean = sub.mean().reindex(scenarios)
                std = sub.std().reindex(scenarios).fillna(0)
                ax.bar(x + (i - (len(strategies) - 1) / 2) * width, mean, width,
                       yerr=std, capsize=2, error_kw={"linewidth": 0.8},
                       color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat])
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8)
            ax.set_ylabel(METRIC_LABELS[metric])
        axes.flat[0].legend(ncol=min(len(strategies), 4), loc="upper left", fontsize=8)
        fig.tight_layout()
        made.append(save_fig(fig, fig_dir / "fig_arena_bars.png"))

    # 2. Heatmap: scenario x strategy avg waiting time
    if "avg_waiting_s" in df.columns and len(scenarios) > 1:
        pivot = (df.groupby(["scenario", "strategy"])["avg_waiting_s"].mean()
                 .unstack("strategy").reindex(index=scenarios, columns=strategies))
        fig, ax = plt.subplots(figsize=(1.4 + 1.1 * len(strategies), 0.9 + 0.55 * len(scenarios)))
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies])
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels([s.replace("_", " ") for s in scenarios])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Avg waiting time (s)", shrink=0.85)
        made.append(save_fig(fig, fig_dir / "fig_scenario_heatmap.png"))

    # 3. Radar: normalized multi-metric profile per strategy (normal scenario if present)
    radar_scenario = "normal" if "normal" in df["scenario"].values else scenarios[0]
    radar_metrics = [m for m in ("avg_waiting_s", "avg_travel_time_s", "avg_queue_veh",
                                 "max_queue_veh", "throughput_veh", "avg_speed_mps")
                     if m in df.columns]
    sub = df[df["scenario"] == radar_scenario]
    if len(radar_metrics) >= 3 and not sub.empty and len(strategies) > 1:
        means = sub.groupby("strategy")[radar_metrics].mean().reindex(strategies)
        norm = means.copy()
        for m in radar_metrics:
            col = means[m]
            rng = col.max() - col.min()
            scaled = (col - col.min()) / rng if rng > 0 else col * 0 + 0.5
            # 1 = best for every axis
            norm[m] = 1 - scaled if m in LOWER_BETTER else scaled
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"projection": "polar"})
        for strat in strategies:
            vals = norm.loc[strat].tolist() + [norm.loc[strat].iloc[0]]
            ax.plot(angles, vals, color=STRATEGY_COLORS[strat],
                    label=STRATEGY_LABELS[strat], linewidth=1.4)
            ax.fill(angles, vals, color=STRATEGY_COLORS[strat], alpha=0.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_LABELS[m].split(" (")[0] for m in radar_metrics], fontsize=8)
        ax.set_yticklabels([])
        ax.set_title(f"Normalized performance ({radar_scenario.replace('_', ' ')})", pad=16)
        ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05), fontsize=8)
        made.append(save_fig(fig, fig_dir / "fig_arena_radar.png"))

    # 4. Queue dynamics: total queue over time, one line per strategy (seed 0)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    plotted = False
    for strat in strategies:
        ts_path = RESULTS_ROOT / f"{radar_scenario}__{strat}__s0" / "timeseries.csv"
        if not ts_path.exists():
            continue
        ts = pd.read_csv(ts_path)
        if {"t", "queue_total"} <= set(ts.columns):
            smooth = ts["queue_total"].rolling(30, min_periods=1).mean()
            ax.plot(ts["t"] / 60, smooth, color=STRATEGY_COLORS[strat],
                    label=STRATEGY_LABELS[strat], linewidth=1.2)
            plotted = True
    if plotted:
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Total queue (veh, 30 s rolling mean)")
        ax.set_title(f"Queue dynamics ({radar_scenario.replace('_', ' ')})")
        ax.legend()
        made.append(save_fig(fig, fig_dir / "fig_queue_timeseries.png"))
    else:
        plt.close(fig)

    # 5. Improvement vs Fixed-Time baseline
    if "fixed" in strategies and len(strategies) > 1 and "avg_waiting_s" in df.columns:
        base = (df[df["strategy"] == "fixed"].groupby("scenario")["avg_waiting_s"].mean()
                .reindex(scenarios))
        fig, ax = plt.subplots(figsize=(7, 3.4))
        x = np.arange(len(scenarios))
        others = [s for s in strategies if s != "fixed"]
        width = 0.8 / len(others)
        for i, strat in enumerate(others):
            mean = (df[df["strategy"] == strat].groupby("scenario")["avg_waiting_s"].mean()
                    .reindex(scenarios))
            imp = (base - mean) / base * 100
            bars = ax.bar(x + (i - (len(others) - 1) / 2) * width, imp, width,
                          color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat])
            for b, v in zip(bars, imp):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + (1 if v >= 0 else -3),
                            f"{v:.0f}%", ha="center", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8)
        ax.set_ylabel("Waiting-time reduction vs Fixed (%)")
        ax.legend()
        made.append(save_fig(fig, fig_dir / "fig_before_after.png"))

    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="FlowMind Strategy Arena batch runner")
    ap.add_argument("--scenarios", default="all",
                    help="'all', comma list of registry names, or spec JSON paths")
    ap.add_argument("--strategies", default="fixed,actuated")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--template", default="cross_basic")
    ap.add_argument("--base-state", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--figures", action="store_true", help="regenerate comparison figures")
    ap.add_argument("--skip-run", action="store_true", help="only collect + figures")
    args = ap.parse_args()

    scenario_list = list(SCENARIOS) if args.scenarios == "all" else args.scenarios.split(",")
    strategy_list = args.strategies.split(",")

    if not args.skip_run:
        total = len(scenario_list) * len(strategy_list) * args.seeds
        n = 0
        for scenario in scenario_list:
            for strategy in strategy_list:
                for seed in range(args.seeds):
                    n += 1
                    print(f"[{n}/{total}] {scenario} / {strategy} / seed {seed}", flush=True)
                    try:
                        info = run_experiment(scenario, strategy, seed,
                                              template=args.template,
                                              base_state=args.base_state, force=args.force)
                        w = info["metrics"].get("avg_waiting_s", float("nan"))
                        tag = "cached" if info["cached"] else "done"
                        print(f"    {tag}: avg_waiting={w:.1f}s", flush=True)
                    except Exception as e:  # keep batch going, report at end
                        print(f"    FAILED: {e}", flush=True)

    df = collect_summary()
    if df.empty:
        print("No results found.")
        return
    print(f"\narena_summary.csv: {len(df)} runs -> {ARENA_CSV}")
    print(_agg(df).round(2).to_string())

    if args.figures or not args.skip_run:
        for p in make_figures(df):
            print(f"figure: {p}")


if __name__ == "__main__":
    main()
