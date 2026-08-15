"""Experiment cache must key on the inputs, not on the directory name.

The published arena table mixed `fixed` runs built from one TrafficState with
`dqn` runs built from another, because the cache key and the summary scan both
only looked at `<scenario>__<strategy>__s<seed>`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import pytest

import experiments.scenario_runner as sr
import experiments.strategy_compare as sc


@dataclass
class _FakeResult:
    metrics: dict


@pytest.fixture
def sandbox(tmp_path, monkeypatch, profile_state):
    """Redirect RESULTS_ROOT and stub out route generation + SUMO."""
    root = tmp_path / "experiments"
    monkeypatch.setattr(sr, "RESULTS_ROOT", root)
    monkeypatch.setattr(sc, "RESULTS_ROOT", root)
    monkeypatch.setattr(sc, "ARENA_CSV", tmp_path / "arena_summary.csv")

    state = tmp_path / "state.json"
    state.write_text(json.dumps(profile_state), encoding="utf-8")

    calls = {"n": 0}

    def fake_generate_routes(state_path, template, out_dir, scenario=None, seed=0,
                             duration_sec=None):
        route = out_dir / "routes.rou.xml"
        route.write_text('<routes>\n</routes>\n', encoding="utf-8")
        return route

    def fake_run_episode(template, route_file, strategy, seed, out_dir, gui=False,
                         duration_sec=1800, lane_closures=None):
        calls["n"] += 1
        metrics = {"avg_waiting_s": 10.0 + calls["n"], "throughput_veh": 500.0}
        (out_dir / "metrics_summary.json").write_text(json.dumps(metrics), encoding="utf-8")
        return _FakeResult(metrics=metrics)

    import simulation.route_generator as rg
    import simulation.sumo_runner as srun
    monkeypatch.setattr(rg, "generate_routes", fake_generate_routes)
    monkeypatch.setattr(srun, "run_episode", fake_run_episode)
    return {"root": root, "state": state, "calls": calls}


def _run(sandbox, scenario="normal", strategy="fixed", seed=0, **kw):
    return sr.run_experiment(scenario, strategy, seed,
                             base_state=str(sandbox["state"]), **kw)


def test_first_run_writes_a_fingerprint(sandbox):
    info = _run(sandbox)
    assert info["cached"] is False
    meta = json.loads((info["out_dir"] / sr.RUN_META).read_text(encoding="utf-8"))
    assert meta["base_state_sha1"] and meta["key"]
    assert meta["model_sha1"] is None  # baseline strategy


def test_identical_inputs_hit_the_cache(sandbox):
    _run(sandbox)
    info = _run(sandbox)
    assert info["cached"] is True
    assert sandbox["calls"]["n"] == 1


def test_changed_base_state_invalidates_the_cache(sandbox, profile_state):
    _run(sandbox)
    profile_state["approaches"]["north"]["flow_vph"] = 2000.0
    sandbox["state"].write_text(json.dumps(profile_state), encoding="utf-8")
    info = _run(sandbox)
    assert info["cached"] is False
    assert sandbox["calls"]["n"] == 2


def test_results_without_a_fingerprint_are_rerun(sandbox):
    """Exactly the stale-directory case that contaminated the arena."""
    info = _run(sandbox)
    (info["out_dir"] / sr.RUN_META).unlink()
    again = _run(sandbox)
    assert again["cached"] is False


def test_force_reruns_even_when_fingerprint_matches(sandbox):
    _run(sandbox)
    assert _run(sandbox, force=True)["cached"] is False


def test_model_checkpoint_is_part_of_the_key(sandbox, tmp_path, monkeypatch):
    import optimization.rl_agents as rl
    ckpt = tmp_path / "dqn_cross_basic.zip"
    ckpt.write_bytes(b"weights-v1")
    monkeypatch.setattr(rl, "model_path", lambda strategy, template: ckpt)

    spec = {"base_state": str(sandbox["state"]), "flow_multipliers": {},
            "lane_closures": [], "duration_sec": 1800}
    first = sr.run_fingerprint(spec, "dqn", 0, "cross_basic")
    ckpt.write_bytes(b"weights-v2-retrained")
    second = sr.run_fingerprint(spec, "dqn", 0, "cross_basic")
    assert first["key"] != second["key"]


def test_scenario_spec_is_part_of_the_key(sandbox):
    base = {"base_state": str(sandbox["state"]), "flow_multipliers": {"east": 1.0},
            "lane_closures": [], "duration_sec": 1800}
    surge = {**base, "flow_multipliers": {"east": 2.2}}
    closed = {**base, "lane_closures": [{"approach": "east", "n_lanes": 1}]}
    longer = {**base, "duration_sec": 3600}
    keys = {sr.run_fingerprint(s, "fixed", 0, "cross_basic")["key"]
            for s in (base, surge, closed, longer)}
    assert len(keys) == 4
