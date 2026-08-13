"""Built-in What-if scenario registry.

A scenario spec (see docs/CONTRACTS.md) modulates a base TrafficState:
flow multipliers per approach, optional lane closures, episode duration.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_STATE = "data/traffic_states/demo_001.json"
FALLBACK_STATE = "data/traffic_states/synthetic_demo.json"

SCENARIOS: dict[str, dict] = {
    "normal": {
        "flow_multipliers": {"north": 1.0, "south": 1.0, "east": 1.0, "west": 1.0},
        "lane_closures": [],
        "duration_sec": 1800,
    },
    "morning_peak": {
        "flow_multipliers": {"north": 1.6, "south": 1.3, "east": 1.1, "west": 1.1},
        "lane_closures": [],
        "duration_sec": 1800,
    },
    "evening_peak": {
        "flow_multipliers": {"north": 1.1, "south": 1.1, "east": 1.5, "west": 1.6},
        "lane_closures": [],
        "duration_sec": 1800,
    },
    "event_surge": {
        "flow_multipliers": {"north": 1.0, "south": 1.0, "east": 2.2, "west": 1.0},
        "lane_closures": [],
        "duration_sec": 1800,
    },
    "lane_closure": {
        "flow_multipliers": {"north": 1.0, "south": 1.0, "east": 1.0, "west": 1.0},
        "lane_closures": [{"approach": "east", "n_lanes": 1}],
        "duration_sec": 1800,
    },
}


def default_base_state() -> str:
    for p in (DEFAULT_STATE, FALLBACK_STATE):
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        "No TrafficState found. Run `python -m vision.analyze ...` first "
        f"(expected {DEFAULT_STATE} or {FALLBACK_STATE})."
    )


def resolve_scenario(name_or_path: str, base_state: str | None = None) -> dict:
    """Return a full scenario spec dict from a registry name or a JSON file path."""
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        spec = json.loads(p.read_text(encoding="utf-8"))
        spec.setdefault("name", p.stem)
    elif name_or_path in SCENARIOS:
        spec = {"name": name_or_path, **json.loads(json.dumps(SCENARIOS[name_or_path]))}
    else:
        raise KeyError(
            f"Unknown scenario '{name_or_path}'. Registry: {sorted(SCENARIOS)} "
            "or pass a path to a scenario spec JSON."
        )
    if base_state:
        spec["base_state"] = base_state
    spec.setdefault("base_state", default_base_state())
    spec.setdefault("flow_multipliers", {k: 1.0 for k in ("north", "south", "east", "west")})
    spec.setdefault("lane_closures", [])
    spec.setdefault("duration_sec", 1800)
    return spec
