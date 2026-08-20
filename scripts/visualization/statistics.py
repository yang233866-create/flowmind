from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


LOWER_IS_BETTER = {
    "avg_waiting_s",
    "avg_travel_time_s",
    "avg_queue_veh",
    "max_queue_veh",
}
HIGHER_IS_BETTER = {"throughput_veh", "avg_speed_mps"}


def _benefit(method: pd.Series, baseline: pd.Series, metric: str) -> pd.Series:
    if metric in LOWER_IS_BETTER:
        return baseline - method
    if metric in HIGHER_IS_BETTER:
        return method - baseline
    raise KeyError(f"Metric direction is not declared: {metric}")


def paired_effects(arena: pd.DataFrame, baseline: str = "fixed") -> pd.DataFrame:
    metrics = [*LOWER_IS_BETTER, *HIGHER_IS_BETTER]
    keys = ["scenario", "seed"]
    base = arena.loc[arena["strategy"] == baseline, keys + metrics].set_index(keys)
    rows = []
    for strategy in sorted(set(arena["strategy"]) - {baseline}):
        method = arena.loc[arena["strategy"] == strategy, keys + metrics].set_index(keys)
        method, matched = method.align(base, join="inner", axis=0)
        for metric in metrics:
            values = _benefit(method[metric], matched[metric], metric).astype(float)
            n = len(values)
            mean = float(values.mean())
            sem = float(stats.sem(values)) if n > 1 else 0.0
            half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
            rows.append(
                {
                    "strategy": strategy,
                    "baseline": baseline,
                    "metric": metric,
                    "n": n,
                    "benefit_mean": mean,
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                    "wins": int((values > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def scenario_benefits(arena: pd.DataFrame, baseline: str = "fixed") -> pd.DataFrame:
    grouped = arena.groupby(["scenario", "strategy"], observed=True).mean(numeric_only=True)
    base = grouped.xs(baseline, level="strategy")
    rows = []
    metrics = [*LOWER_IS_BETTER, *HIGHER_IS_BETTER]
    for (scenario, strategy), values in grouped.iterrows():
        if strategy == baseline:
            continue
        for metric in metrics:
            raw = float(values[metric])
            reference = float(base.loc[scenario, metric])
            benefit = reference - raw if metric in LOWER_IS_BETTER else raw - reference
            pct = benefit / abs(reference) * 100 if reference else np.nan
            rows.append(
                {
                    "scenario": scenario,
                    "strategy": strategy,
                    "metric": metric,
                    "benefit": benefit,
                    "benefit_pct": pct,
                }
            )
    return pd.DataFrame(rows)


def queue_exceedance(timeseries: dict[str, pd.DataFrame], arena: pd.DataFrame) -> pd.DataFrame:
    lookup = arena.set_index("exp_id")["strategy"].to_dict()
    rows = []
    thresholds = np.arange(0, 101, 2)
    for strategy in sorted(arena["strategy"].unique()):
        values = np.concatenate(
            [
                frame["queue_total"].to_numpy(dtype=float)
                for exp_id, frame in timeseries.items()
                if lookup[exp_id] == strategy
            ]
        )
        for threshold in thresholds:
            rows.append(
                {
                    "strategy": strategy,
                    "threshold": float(threshold),
                    "exceedance_pct": float((values > threshold).mean() * 100),
                }
            )
    return pd.DataFrame(rows)


def per_run_queue_summary(timeseries: dict[str, pd.DataFrame], arena: pd.DataFrame) -> pd.DataFrame:
    lookup = arena.set_index("exp_id")[["scenario", "strategy", "seed"]]
    rows = []
    for exp_id, frame in timeseries.items():
        meta = lookup.loc[exp_id]
        queue = frame["queue_total"].to_numpy(dtype=float)
        rows.append(
            {
                "exp_id": exp_id,
                "scenario": meta["scenario"],
                "strategy": meta["strategy"],
                "seed": int(meta["seed"]),
                "queue_mean": float(np.mean(queue)),
                "queue_p95": float(np.percentile(queue, 95)),
                "share_above_40_pct": float(np.mean(queue > 40) * 100),
            }
        )
    return pd.DataFrame(rows)


def metric_winners(arena: pd.DataFrame) -> pd.DataFrame:
    means = arena.groupby(["scenario", "strategy"], observed=True).mean(numeric_only=True)
    rows = []
    for scenario in arena["scenario"].unique():
        block = means.loc[scenario]
        for metric in [*LOWER_IS_BETTER, *HIGHER_IS_BETTER]:
            winner = block[metric].idxmin() if metric in LOWER_IS_BETTER else block[metric].idxmax()
            best = float(block.loc[winner, metric])
            for strategy, value in block[metric].items():
                value = float(value)
                if metric in LOWER_IS_BETTER:
                    regret = (value - best) / max(abs(best), 1e-9)
                else:
                    regret = (best - value) / max(abs(best), 1e-9)
                rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "strategy": strategy,
                        "winner": strategy == winner,
                        "normalized_regret": regret,
                    }
                )
    return pd.DataFrame(rows)

def normalized_regret_profiles(arena: pd.DataFrame) -> pd.DataFrame:
    """Return scenario-level distance from the real best strategy for each metric."""
    means = (
        arena.groupby(["scenario", "strategy"], observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    rows = []
    for scenario, block in means.groupby("scenario", observed=True):
        for metric in [*LOWER_IS_BETTER, *HIGHER_IS_BETTER]:
            values = block.set_index("strategy")[metric].astype(float)
            best = float(values.min() if metric in LOWER_IS_BETTER else values.max())
            denominator = max(abs(best), 1e-9)
            for strategy, value in values.items():
                regret = (
                    (float(value) - best) / denominator
                    if metric in LOWER_IS_BETTER
                    else (best - float(value)) / denominator
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "strategy": strategy,
                        "metric": metric,
                        "value": float(value),
                        "best_value": best,
                        "normalized_regret": max(0.0, float(regret)),
                    }
                )
    return pd.DataFrame(rows)


def paired_run_transitions(arena: pd.DataFrame, baseline: str = "fixed") -> pd.DataFrame:
    """Retain every scenario/seed pair and its direction-corrected benefit."""
    metrics = [*LOWER_IS_BETTER, *HIGHER_IS_BETTER]
    keys = ["scenario", "seed"]
    base = arena.loc[arena["strategy"] == baseline, keys + metrics].set_index(keys)
    rows = []
    for strategy in sorted(set(arena["strategy"]) - {baseline}):
        method = arena.loc[arena["strategy"] == strategy, keys + metrics].set_index(keys)
        method, matched = method.align(base, join="inner", axis=0)
        for (scenario, seed), values in method.iterrows():
            for metric in metrics:
                method_value = float(values[metric])
                baseline_value = float(matched.loc[(scenario, seed), metric])
                benefit = (
                    baseline_value - method_value
                    if metric in LOWER_IS_BETTER
                    else method_value - baseline_value
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": int(seed),
                        "strategy": strategy,
                        "baseline": baseline,
                        "metric": metric,
                        "baseline_value": baseline_value,
                        "method_value": method_value,
                        "benefit": float(benefit),
                    }
                )
    return pd.DataFrame(rows)


def aligned_queue_trajectories(
    timeseries: dict[str, pd.DataFrame],
    arena: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """Aggregate raw queue seconds into aligned windows, then summarize three seeds."""
    if window <= 0:
        raise ValueError("window must be a positive number of seconds")
    lookup = arena.set_index("exp_id")[["scenario", "strategy", "seed"]]
    run_rows = []
    for exp_id, frame in timeseries.items():
        meta = lookup.loc[exp_id]
        values = frame["queue_total"].to_numpy(dtype=float)
        bins = np.arange(len(values), dtype=int) // window
        grouped = pd.DataFrame({"time_bin": bins, "queue_total": values}).groupby(
            "time_bin", observed=True
        )["queue_total"].mean()
        for time_bin, queue in grouped.items():
            run_rows.append(
                {
                    "scenario": meta["scenario"],
                    "strategy": meta["strategy"],
                    "seed": int(meta["seed"]),
                    "time_s": int(time_bin) * window,
                    "queue_total": float(queue),
                }
            )
    run_level = pd.DataFrame(run_rows)
    return (
        run_level.groupby(["scenario", "strategy", "time_s"], observed=True)
        .agg(
            queue_mean=("queue_total", "mean"),
            queue_min=("queue_total", "min"),
            queue_max=("queue_total", "max"),
        )
        .reset_index()
    )