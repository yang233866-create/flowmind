"""collect_summary must refuse to tabulate runs that are not comparable."""
from __future__ import annotations

import json
import os
import sys

import pytest

import experiments.scenario_runner as sr
import experiments.strategy_compare as sc

METRICS = {"avg_waiting_s": 42.0, "avg_travel_time_s": 90.0, "throughput_veh": 500.0,
           "avg_queue_veh": 12.0, "max_queue_veh": 30.0, "avg_speed_mps": 6.0,
           "teleports": 0.0}


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    root = tmp_path / "experiments"
    root.mkdir()
    monkeypatch.setattr(sc, "RESULTS_ROOT", root)
    monkeypatch.setattr(sc, "ARENA_CSV", tmp_path / "arena_summary.csv")
    return root


def _make_run(root, scenario, strategy, seed, state_sha=None, mtime=None, **metrics):
    d = root / f"{scenario}__{strategy}__s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    summary = d / "metrics_summary.json"
    summary.write_text(json.dumps({**METRICS, **metrics}), encoding="utf-8")
    if state_sha is not None:
        (d / sr.RUN_META).write_text(json.dumps({
            "base_state": f"data/traffic_states/{state_sha}.json",
            "base_state_sha1": state_sha,
            "model_sha1": None,
            "key": f"key_{scenario}_{strategy}_{seed}",
        }), encoding="utf-8")
    if mtime is not None:
        os.utime(summary, (mtime, mtime))
    return d


def test_provenance_columns_land_in_the_csv(results_root):
    _make_run(results_root, "normal", "fixed", 0, state_sha="aaa")
    df = sc.collect_summary()
    assert len(df) == 1
    for col in sc.PROVENANCE_COLS:
        assert col in df.columns
    assert df.iloc[0]["base_state_sha1"] == "aaa"
    assert "mtime" not in df.columns
    assert sc.ARENA_CSV.exists()


def test_unfingerprinted_runs_are_dropped_by_default(results_root, capsys):
    _make_run(results_root, "normal", "fixed", 0, state_sha="aaa")
    _make_run(results_root, "normal", "dqn", 0, state_sha=None)  # legacy dir
    df = sc.collect_summary()
    assert set(df["strategy"]) == {"fixed"}
    assert "normal__dqn__s0" in capsys.readouterr().out


def test_lax_mode_keeps_unfingerprinted_runs(results_root):
    _make_run(results_root, "normal", "fixed", 0, state_sha="aaa")
    _make_run(results_root, "normal", "dqn", 0, state_sha=None)
    df = sc.collect_summary(strict=False)
    assert set(df["strategy"]) == {"fixed", "dqn"}


def test_runs_from_a_different_base_state_are_dropped(results_root, capsys):
    """The published table compared synthetic_demo baselines to demo_001 RL runs."""
    for strategy in ("fixed", "actuated"):
        _make_run(results_root, "normal", strategy, 0, state_sha="old", mtime=1_000_000)
    for strategy in ("dqn", "ppo"):
        _make_run(results_root, "normal", strategy, 0, state_sha="new", mtime=2_000_000)

    df = sc.collect_summary()
    assert set(df["base_state_sha1"]) == {"new"}
    assert set(df["strategy"]) == {"dqn", "ppo"}
    out = capsys.readouterr().out
    assert "not comparable" in out
    assert "normal__fixed__s0" in out


def test_smoke_dirs_and_malformed_names_are_ignored(results_root):
    _make_run(results_root, "normal", "fixed", 0, state_sha="aaa")
    _make_run(results_root, "smoke_test", "fixed", 0, state_sha="aaa")
    (results_root / "not_an_experiment").mkdir()
    _make_run(results_root, "normal", "fixed", 99, state_sha="aaa")
    bad_seed = results_root / "normal__fixed__sX"
    bad_seed.mkdir()
    (bad_seed / "metrics_summary.json").write_text(json.dumps(METRICS), encoding="utf-8")

    df = sc.collect_summary()
    assert sorted(df["seed"]) == [0, 99]


def test_empty_root_returns_empty_frame(results_root):
    df = sc.collect_summary()
    assert df.empty
    assert not sc.ARENA_CSV.exists()


def test_a_failed_batch_exits_nonzero(results_root, monkeypatch, capsys):
    """A table assembled from a batch where every run crashed is not the table
    that was asked for -- exit 1 so callers cannot mistake it for success."""
    def boom(*a, **k):
        raise RuntimeError("SUMO not on PATH")

    monkeypatch.setattr(sc, "run_experiment", boom)
    monkeypatch.setattr(sc, "make_figures", lambda df: [])
    monkeypatch.setattr(sys, "argv", ["strategy_compare", "--scenarios", "normal",
                                      "--strategies", "fixed,actuated", "--seeds", "1"])
    with pytest.raises(SystemExit) as excinfo:
        sc.main()
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "2 of 2 runs FAILED" in out
    assert "SUMO not on PATH" in out


def test_a_clean_batch_does_not_raise(results_root, monkeypatch, capsys):
    def ok(scenario, strategy, seed, **k):
        _make_run(results_root, scenario, strategy, seed, state_sha="aaa")
        return {"metrics": METRICS, "cached": False}

    monkeypatch.setattr(sc, "run_experiment", ok)
    monkeypatch.setattr(sc, "make_figures", lambda df: [])
    monkeypatch.setattr(sys, "argv", ["strategy_compare", "--scenarios", "normal",
                                      "--strategies", "fixed,actuated", "--seeds", "1"])
    sc.main()
    assert "FAILED" not in capsys.readouterr().out
    assert sc.ARENA_CSV.exists()
