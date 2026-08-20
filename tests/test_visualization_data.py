from pathlib import Path

import numpy as np
import pytest

from scripts.visualization.data_loader import (
    REQUIRED_METRICS,
    load_visualization_data,
)
from scripts.visualization.statistics import paired_effects
from scripts.visualization import statistics as viz_stats


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_visualization_data(ROOT)


def test_arena_is_complete_factorial_evidence(bundle):
    arena = bundle.arena
    assert len(arena) == 60
    assert set(arena["scenario"]) == {
        "normal",
        "morning_peak",
        "evening_peak",
        "event_surge",
        "lane_closure",
    }
    assert set(arena["strategy"]) == {"fixed", "actuated", "dqn", "ppo"}
    assert set(arena["seed"]) == {0, 1, 2}
    assert not arena.duplicated(["scenario", "strategy", "seed"]).any()
    assert (arena.groupby(["scenario", "strategy"]).size() == 3).all()
    assert np.isfinite(arena[list(REQUIRED_METRICS)].to_numpy(dtype=float)).all()


def test_every_arena_run_has_traceable_artifacts(bundle):
    assert len(bundle.timeseries) == 60
    assert set(bundle.timeseries) == set(bundle.arena["exp_id"])
    assert len(bundle.run_meta) == 60
    for exp_id, frame in bundle.timeseries.items():
        assert not frame.empty, exp_id
        assert {"t", "queue_total", "queue_north", "queue_south", "queue_east", "queue_west"} <= set(frame)
        assert np.isfinite(frame["queue_total"].to_numpy(dtype=float)).all()


def test_perception_provenance_is_not_blurred(bundle):
    approaches = bundle.traffic_state["approaches"]
    assert approaches["north"]["observed"] is True
    assert approaches["south"]["observed"] is True
    assert approaches["east"]["observed"] is False
    assert approaches["west"]["observed"] is False
    assert bundle.traffic_state["flow_profile"]["east"] == []
    assert bundle.traffic_state["flow_profile"]["west"] == []


def test_formal_annotated_frame_has_validated_provenance(bundle):
    metadata = bundle.annotated_frame_meta
    assert metadata["selector"] == {
        "name": "active-tracks-trace-points",
        "version": "active-tracks-trace-points-v1",
    }

    frame = metadata["frame"]
    source = bundle.traffic_state["source"]
    assert frame["fps"] == source["fps"]
    assert 0 <= frame["index"] < source["frames"]
    assert frame["timestamp_sec"] == pytest.approx(
        frame["index"] / frame["fps"]
    )


def test_paired_effects_match_reader_facing_claims(bundle):
    effects = paired_effects(bundle.arena, baseline="fixed")
    wait = effects.query("strategy == 'actuated' and metric == 'avg_waiting_s'").iloc[0]
    assert wait["n"] == 15
    assert wait["benefit_mean"] == pytest.approx(32.67, abs=0.03)
    assert wait["ci_low"] == pytest.approx(26.12, abs=0.05)
    assert wait["ci_high"] == pytest.approx(39.22, abs=0.05)
    assert wait["wins"] == 15

    ppo_queue = effects.query("strategy == 'ppo' and metric == 'max_queue_veh'").iloc[0]
    assert ppo_queue["benefit_mean"] == pytest.approx(39.2, abs=0.05)
    assert ppo_queue["wins"] == 15

    ppo_throughput = effects.query("strategy == 'ppo' and metric == 'throughput_veh'").iloc[0]
    assert ppo_throughput["benefit_mean"] < 0
    assert ppo_throughput["wins"] == 0


def test_formal_training_runs_reach_approximately_100k_steps(bundle):
    assert set(bundle.training) == {"dqn", "ppo"}
    for algorithm, curves in bundle.training.items():
        assert curves["reward"]["step"].max() >= 99_000, algorithm
        assert np.isfinite(curves["reward"]["value"]).all()

def test_normalized_regret_profiles_are_complete_and_non_negative(bundle):
    assert hasattr(viz_stats, "normalized_regret_profiles")
    regret = viz_stats.normalized_regret_profiles(bundle.arena)
    assert len(regret) == 5 * 4 * len(REQUIRED_METRICS)
    assert np.isfinite(regret["normalized_regret"]).all()
    assert (regret["normalized_regret"] >= 0).all()
    minima = regret.groupby(["scenario", "metric"])["normalized_regret"].min()
    assert np.allclose(minima.to_numpy(), 0.0)


def test_paired_run_transitions_preserve_every_seed_pair(bundle):
    assert hasattr(viz_stats, "paired_run_transitions")
    transitions = viz_stats.paired_run_transitions(bundle.arena, baseline="fixed")
    assert len(transitions) == 3 * 5 * 3 * len(REQUIRED_METRICS)
    for metric in ("avg_waiting_s", "max_queue_veh", "throughput_veh"):
        assert len(transitions.query("metric == @metric")) == 45
    assert {"baseline_value", "method_value", "benefit"} <= set(transitions)


def test_aligned_queue_trajectories_cover_every_scenario_strategy(bundle):
    assert hasattr(viz_stats, "aligned_queue_trajectories")
    aligned = viz_stats.aligned_queue_trajectories(
        bundle.timeseries, bundle.arena, window=30
    )
    assert set(aligned["scenario"]) == set(bundle.arena["scenario"])
    assert set(aligned["strategy"]) == set(bundle.arena["strategy"])
    assert (aligned.groupby(["scenario", "strategy"]).size() == 60).all()
    assert (aligned["queue_min"] <= aligned["queue_mean"]).all()
    assert (aligned["queue_mean"] <= aligned["queue_max"]).all()
