"""Shared fixtures. Tests run from the repo root (templates/states are relative)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STATES = REPO_ROOT / "data" / "traffic_states"


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    """Modules resolve data/ and simulation/templates/ relative to the cwd."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def profile_state() -> dict:
    """A vision-derived state: short observed window, profile ending in zeros.

    Mirrors data/traffic_states/demo_001.json, the shape that triggered both
    route-generator defects.
    """
    return {
        "scenario_id": "unit_profile",
        "duration_sec": 25.0,
        "profile_bins_sec": 5,
        "flow_profile": {
            "north": [720.0, 2880.0, 720.0, 0.0, 0.0],
            "south": [2160.0, 720.0, 720.0, 1440.0, 0.0],
            "east": [],
            "west": [],
        },
        "approaches": {
            "north": {"flow_vph": 1000.0, "observed": True},
            "south": {"flow_vph": 1200.0, "observed": True},
            "east": {"flow_vph": 400.0, "observed": False},
            "west": {"flow_vph": 400.0, "observed": False},
        },
    }


@pytest.fixture
def flat_state() -> dict:
    """A state with no flow_profile (synthetic_demo.json shape)."""
    return {
        "scenario_id": "unit_flat",
        "duration_sec": 600.0,
        "flow_profile": None,
        "approaches": {a: {"flow_vph": v, "observed": True} for a, v in
                       (("north", 900.0), ("south", 850.0),
                        ("east", 1100.0), ("west", 600.0))},
    }


@pytest.fixture
def write_state(tmp_path):
    def _write(state: dict, name: str = "state.json") -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(state), encoding="utf-8")
        return p
    return _write


@pytest.fixture
def normal_spec() -> dict:
    return {
        "name": "normal",
        "flow_multipliers": {"north": 1.0, "south": 1.0, "east": 1.0, "west": 1.0},
        "lane_closures": [],
        "duration_sec": 1800,
    }
