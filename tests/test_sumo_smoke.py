"""End-to-end SUMO checks. Slow: `pytest -m "not sumo"` to skip.

These are the tests that would have caught the silent demand loss -- the defect
was invisible to anything that did not actually count vehicles.
"""
from __future__ import annotations

import pandas as pd
import pytest

from simulation.route_generator import generate_routes
from simulation.sumo_runner import run_episode

pytestmark = pytest.mark.sumo

DURATION = 180  # keep the suite usable; long enough to fill the intersection


@pytest.fixture
def routes(tmp_path, profile_state, write_state, normal_spec):
    return generate_routes(write_state(profile_state), "cross_basic",
                           tmp_path / "routes",
                           scenario={**normal_spec, "duration_sec": DURATION})


def test_vehicles_actually_reach_the_intersection(routes, tmp_path):
    """Regression for unsorted routes: throughput was 40 veh over 1800 s."""
    result = run_episode("cross_basic", routes, "fixed", 0, tmp_path / "run",
                         duration_sec=DURATION)
    assert result.metrics["throughput_veh"] > 20
    assert result.metrics["avg_travel_time_s"] > 0
    assert set(result.metrics) >= {"avg_waiting_s", "throughput_veh", "teleports"}


def test_baseline_timeseries_is_one_row_per_second(routes, tmp_path):
    result = run_episode("cross_basic", routes, "fixed", 0, tmp_path / "run",
                         duration_sec=DURATION)
    assert len(result.timeseries) == DURATION
    assert result.timeseries["t"].tolist() == list(range(1, DURATION + 1))


def test_demand_is_still_flowing_at_the_end_of_the_episode(routes, tmp_path):
    """The frozen trailing profile bin emptied the network after ~25 s."""
    result = run_episode("cross_basic", routes, "fixed", 0, tmp_path / "run",
                         duration_sec=DURATION)
    tail = result.timeseries[result.timeseries["t"] > DURATION - 30]
    assert tail["running_veh"].mean() > 5


def test_run_episode_refuses_an_unsorted_route_file(tmp_path):
    bad = tmp_path / "bad.rou.xml"
    bad.write_text(
        '<routes>\n'
        '    <flow id="a" begin="10" end="20" from="N_in" to="S_out" vehsPerHour="600"/>\n'
        '    <flow id="b" begin="0" end="10" from="N_in" to="S_out" vehsPerHour="600"/>\n'
        '</routes>\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not sorted"):
        run_episode("cross_basic", bad, "fixed", 0, tmp_path / "run", duration_sec=30)


def test_actuated_and_fixed_are_both_runnable(routes, tmp_path):
    out = {}
    for strategy in ("fixed", "actuated"):
        r = run_episode("cross_basic", routes, strategy, 0, tmp_path / strategy,
                        duration_sec=DURATION)
        out[strategy] = r.metrics
        assert (tmp_path / strategy / "metrics_summary.json").exists()
        assert pd.read_csv(tmp_path / strategy / "timeseries.csv").shape[0] == DURATION
    # both saw the same demand; a controller cannot change how many cars exist
    assert out["fixed"]["throughput_veh"] > 0 and out["actuated"]["throughput_veh"] > 0
