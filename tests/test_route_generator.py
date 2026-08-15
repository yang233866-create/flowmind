"""Route-generator invariants.

Every test here corresponds to a defect that silently produced valid-looking but
wrong results, so they assert on demand actually reaching SUMO -- not just on the
file parsing.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from simulation.route_generator import (
    DEFAULT_FLOW_VPH,
    _flow_bins,
    check_sorted,
    flow_begin_times,
    generate_routes,
    route_horizon,
)


def _flows(route_path):
    return list(ET.parse(str(route_path)).getroot().iter("flow"))


def _expected_vehicles(route_path) -> float:
    """Vehicles SUMO will insert, from vehsPerHour x bin length."""
    return sum(float(f.get("vehsPerHour")) * (float(f.get("end")) - float(f.get("begin"))) / 3600
               for f in _flows(route_path))


# --- sorting: SUMO silently discards out-of-order elements -------------------

def test_flows_are_sorted_by_begin(profile_state, write_state, normal_spec, tmp_path):
    route = generate_routes(write_state(profile_state), "cross_basic",
                            tmp_path / "out", scenario=normal_spec)
    times = flow_begin_times(route)
    assert times == sorted(times)
    check_sorted(route)  # must not raise


def test_check_sorted_rejects_unsorted_file(tmp_path):
    bad = tmp_path / "bad.rou.xml"
    bad.write_text(
        '<routes>\n'
        '    <flow id="a" begin="0" end="5" vehsPerHour="100"/>\n'
        '    <flow id="b" begin="10" end="15" vehsPerHour="100"/>\n'
        '    <flow id="c" begin="5" end="10" vehsPerHour="100"/>\n'
        '</routes>\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not sorted"):
        check_sorted(bad)


def test_check_sorted_ignores_vtype_children(tmp_path):
    ok = tmp_path / "ok.rou.xml"
    ok.write_text(
        '<routes>\n'
        '    <vType id="car" length="4.5"/>\n'
        '    <flow id="a" begin="0" end="5" vehsPerHour="100"/>\n'
        '    <vehicle id="v" depart="7"/>\n'
        '</routes>\n', encoding="utf-8")
    check_sorted(ok)
    assert flow_begin_times(ok) == [0.0, 7.0]


def test_flow_ids_are_unique(profile_state, write_state, normal_spec, tmp_path):
    """Duplicate flow ids make SUMO abort, not warn."""
    route = generate_routes(write_state(profile_state), "cross_basic",
                            tmp_path / "out", scenario=normal_spec)
    ids = [f.get("id") for f in _flows(route)]
    assert len(ids) == len(set(ids))


# --- profile extrapolation: trailing zero bins used to freeze demand --------

def test_profile_shorter_than_episode_falls_back_to_aggregate(profile_state):
    bins = _flow_bins(profile_state, "north", 1800.0)
    assert bins[:3] == [(0.0, 5.0, 720.0), (5.0, 10.0, 2880.0), (10.0, 15.0, 720.0)]
    # the tail is one bin at the approach's aggregate rate, not a frozen 0.0
    assert bins[-1] == (25.0, 1800.0, 1000.0)
    assert bins[-1][2] > 0


def test_demand_survives_the_whole_episode(profile_state, write_state, normal_spec, tmp_path):
    """The bug: 1800 s episode received ~25 s of demand and looked successful."""
    route = generate_routes(write_state(profile_state), "cross_basic",
                            tmp_path / "out", scenario=normal_spec)
    assert route_horizon(route) == 1800.0
    # north+south+east+west aggregate over 1800 s, minus the observed window
    assert _expected_vehicles(route) > 1000


def test_no_profile_gives_one_bin_per_approach(flat_state):
    assert _flow_bins(flat_state, "east", 600.0) == [(0.0, 600.0, 1100.0)]


def test_profile_longer_than_duration_is_truncated(profile_state):
    bins = _flow_bins(profile_state, "north", 12.0)
    assert bins == [(0.0, 5.0, 720.0), (5.0, 10.0, 2880.0), (10.0, 12.0, 720.0)]
    assert bins[-1][1] == 12.0


def test_bin_sec_larger_than_duration(profile_state):
    """A 900 s bin is still only backed by the 25 s the camera actually saw."""
    profile_state["profile_bins_sec"] = 900
    bins = _flow_bins(profile_state, "north", 300.0)
    assert bins == [(0.0, 25.0, 720.0), (25.0, 300.0, 1000.0)]


def test_partial_final_bin_is_not_replayed_at_full_width(profile_state):
    """The 5th bin of a 21.5 s clip spans 1.5 s; its vph was computed over 1.5 s.

    Replaying it for a full profile_bins_sec would inject 3.3x the vehicles the
    camera saw in that window.
    """
    profile_state["duration_sec"] = 21.5
    profile_state["flow_profile"]["north"] = [720.0, 720.0, 720.0, 720.0, 2400.0]
    bins = _flow_bins(profile_state, "north", 1800.0)
    assert bins[4] == (20.0, 21.5, 2400.0)
    assert bins[-1] == (21.5, 1800.0, 1000.0)  # aggregate fallback beyond the clip


def test_profile_bins_never_outrun_the_observed_window(profile_state):
    """A profile with more bins than duration_sec covers stops at the window."""
    profile_state["duration_sec"] = 12.0
    bins = _flow_bins(profile_state, "north", 1800.0)
    assert [b[:2] for b in bins[:3]] == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]
    assert bins[-1] == (12.0, 1800.0, 1000.0)


@pytest.mark.parametrize("bin_sec", [0, -5, None])
def test_degenerate_bin_sec_does_not_hang(profile_state, bin_sec):
    profile_state["profile_bins_sec"] = bin_sec
    bins = _flow_bins(profile_state, "north", 600.0)
    assert bins == [(0.0, 600.0, 1000.0)]


def test_empty_profile_list_uses_aggregate(profile_state):
    assert _flow_bins(profile_state, "east", 600.0) == [(0.0, 600.0, 400.0)]


def test_missing_approach_uses_default_flow(profile_state):
    profile_state["approaches"].pop("west")
    profile_state["flow_profile"]["west"] = []
    assert _flow_bins(profile_state, "west", 600.0) == [(0.0, 600.0, DEFAULT_FLOW_VPH)]


# --- duration override ------------------------------------------------------

def test_duration_override_beats_state_and_spec(profile_state, write_state, tmp_path):
    route = generate_routes(write_state(profile_state), "cross_basic",
                            tmp_path / "out", scenario={"duration_sec": 600},
                            duration_sec=1800)
    assert route_horizon(route) == 1800.0


def test_state_duration_is_the_fallback(flat_state, write_state, tmp_path):
    route = generate_routes(write_state(flat_state), "cross_basic", tmp_path / "out")
    assert route_horizon(route) == 600.0


# --- multipliers -------------------------------------------------------------

def test_flow_multiplier_scales_only_its_approach(flat_state, write_state, tmp_path, normal_spec):
    base = generate_routes(write_state(flat_state), "cross_basic", tmp_path / "a",
                           scenario={**normal_spec, "duration_sec": 600})
    spec = {**normal_spec, "duration_sec": 600,
            "flow_multipliers": {"north": 2.0, "south": 1.0, "east": 1.0, "west": 1.0}}
    doubled = generate_routes(write_state(flat_state), "cross_basic", tmp_path / "b",
                              scenario=spec)

    def per_approach(route):
        out = {}
        for f in _flows(route):
            a = f.get("id").split("_")[1]
            out[a] = out.get(a, 0.0) + float(f.get("vehsPerHour"))
        return out

    b, d = per_approach(base), per_approach(doubled)
    assert d["north"] == pytest.approx(2 * b["north"])
    for a in ("south", "east", "west"):
        assert d[a] == pytest.approx(b[a])
