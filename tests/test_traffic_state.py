"""TrafficState contract assembly: mix normalisation and the validator.

`normalize_mix` used to round after absorbing the float residual, which left a
3-way even split summing to 1.000002 -- a mix the validator then rejected.
"""
from __future__ import annotations

import math

import pytest

from vision.traffic_state import (
    APPROACHES,
    DEFAULT_FLOW_VPH,
    SCHEMA_VERSION,
    VEHICLE_TYPES,
    build_traffic_state,
    counts_to_vph,
    normalize_mix,
    validate_traffic_state,
)


@pytest.mark.parametrize("counts", [
    {"car": 1, "bus": 1, "truck": 1},                 # 3-way even split
    {"car": 1, "bus": 1, "truck": 1, "motorcycle": 1},
    {"car": 7, "bus": 3, "truck": 3, "motorcycle": 3},
    {"car": 1, "bus": 2},
    {"car": 0, "bus": 0, "truck": 0, "motorcycle": 1},
])
def test_mix_sums_to_exactly_one(counts):
    mix = normalize_mix(counts)
    assert set(mix) == set(VEHICLE_TYPES)
    assert math.isclose(sum(mix.values()), 1.0, abs_tol=1e-9)
    assert all(round(v, 6) == v for v in mix.values())


def test_even_split_passes_the_validator():
    """A mix the validator would previously flag as 'sums to 1.000002, not 1'."""
    state = build_traffic_state(
        "unit_mix", observed=["north"], counts={"north": 3},
        class_counts={"north": {"car": 1, "bus": 1, "truck": 1}},
        duration_sec=30.0)
    assert validate_traffic_state(state) == []


def test_all_zero_counts_fall_back_to_pure_car():
    assert normalize_mix({})["car"] == 1.0
    assert normalize_mix(None)["car"] == 1.0


def test_counts_to_vph_guards_zero_span():
    assert counts_to_vph(5, 0) == 0.0
    assert counts_to_vph(5, 30) == pytest.approx(600.0)


def test_unobserved_approach_mirrors_then_defaults():
    state = build_traffic_state("unit_mirror", observed=["north"], counts={"north": 10},
                                class_counts={"north": {"car": 10}}, duration_sec=60.0)
    assert state["approaches"]["south"]["flow_vph"] == state["approaches"]["north"]["flow_vph"]
    assert state["approaches"]["south"]["observed"] is False
    assert state["approaches"]["east"]["flow_vph"] == DEFAULT_FLOW_VPH
    assert validate_traffic_state(state) == []


def test_partial_last_bin_uses_its_real_span():
    """21.5 s of video with 5 s bins: the tail bin is 1.5 s, not 5 s."""
    state = build_traffic_state(
        "unit_span", observed=["north"], counts={"north": 9},
        class_counts={"north": {"car": 9}}, duration_sec=21.5,
        profile_counts={"north": [2, 2, 2, 2, 1]},
        profile_spans=[5.0, 5.0, 5.0, 5.0, 1.5], profile_bins_sec=5)
    profile = state["flow_profile"]["north"]
    assert profile[0] == pytest.approx(1440.0)
    assert profile[-1] == pytest.approx(2400.0)  # 1 veh / 1.5 s, not 1 / 5 s
    assert validate_traffic_state(state) == []


def test_validator_reports_a_broken_state():
    bad = {"schema_version": "0.9", "scenario_id": "", "duration_sec": 0,
           "approaches": {}, "turning_ratio": {}, "profile_bins_sec": 0}
    problems = validate_traffic_state(bad)
    assert any("schema_version" in p for p in problems)
    assert any("scenario_id" in p for p in problems)
    assert any("duration_sec" in p for p in problems)
    assert any(f"approaches.{d} missing" in p for d in APPROACHES for p in problems)


def test_valid_state_declares_the_current_schema():
    state = build_traffic_state("unit_schema", observed=[], counts={}, class_counts={},
                                duration_sec=10.0)
    assert state["schema_version"] == SCHEMA_VERSION
    assert validate_traffic_state(state) == []
