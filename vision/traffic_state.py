"""TrafficState assembly (docs/CONTRACTS.md schema 1.1).

This module is the contract layer of `vision/`: it owns the canonical approach /
vehicle-type vocabulary and the rules for turning raw line-crossing counts into
the single cross-module JSON interface consumed by `simulation.route_generator`
and the Streamlit app. It deliberately has no perception dependencies (no
OpenCV / ultralytics / supervision) so it can be imported and validated cheaply.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "1.1"

APPROACHES: tuple[str, ...] = ("north", "south", "east", "west")
VEHICLE_TYPES: tuple[str, ...] = ("car", "bus", "truck", "motorcycle")

# An unobserved approach borrows the flow of the opposing approach: on a
# two-way road the counted direction is the best available estimate for the
# uncounted one (north <-> south, east <-> west).
MIRROR: dict[str, str] = {"north": "south", "south": "north",
                          "east": "west", "west": "east"}

DEFAULT_FLOW_VPH = 400.0
DEFAULT_TURNING = {"left": 0.15, "straight": 0.70, "right": 0.15}
DEFAULT_MIX = {"car": 1.0, "bus": 0.0, "truck": 0.0, "motorcycle": 0.0}
DEFAULT_PROFILE_BINS_SEC = 300


def normalize_mix(counts: Mapping[str, float] | None) -> dict[str, float]:
    """Normalise per-class counts to a mix summing to 1 over VEHICLE_TYPES.

    Classes outside the four contract types are expected to have been folded
    into `car` upstream; an all-zero input falls back to a pure-car mix.
    """
    raw = {vt: float((counts or {}).get(vt, 0.0)) for vt in VEHICLE_TYPES}
    total = sum(raw.values())
    if total <= 0:
        return dict(DEFAULT_MIX)
    # Round first, then absorb the residual into the dominant class: rounding
    # last can push the sum up to 2e-6 away from 1, which validate_traffic_state
    # then reports as a contract violation (e.g. a 3-way even split).
    mix = {vt: round(raw[vt] / total, 6) for vt in VEHICLE_TYPES}
    top = max(mix, key=mix.get)
    mix[top] = round(mix[top] + (1.0 - sum(mix.values())), 6)
    return mix


def counts_to_vph(count: float, span_sec: float) -> float:
    if span_sec <= 0:
        return 0.0
    return float(count) * 3600.0 / float(span_sec)


def build_traffic_state(
    scenario_id: str,
    *,
    observed: Iterable[str],
    counts: Mapping[str, int],
    class_counts: Mapping[str, Mapping[str, int]],
    duration_sec: float,
    video: str | None = None,
    fps: float | None = None,
    frames: int | None = None,
    queue_est: Mapping[str, float] | None = None,
    profile_counts: Mapping[str, Sequence[int]] | None = None,
    profile_spans: Sequence[float] | None = None,
    profile_bins_sec: int = DEFAULT_PROFILE_BINS_SEC,
    turning_ratio: Mapping[str, Mapping[str, float]] | None = None,
    analyzed_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a schema 1.1 TrafficState dict.

    Args:
        observed: approaches that had a count line configured; every other
            approach is emitted with `observed: false`.
        counts: total de-duplicated crossings per observed approach.
        class_counts: per-approach {vehicle_type: crossings}.
        duration_sec: length of the analysed window (not necessarily the whole
            video, e.g. when `--max-frames` truncates it). Flows are derived
            from this span.
        profile_counts / profile_spans: per-approach crossings per time bin and
            the actual wall-clock length of each bin, used for `flow_profile`.
    """
    observed_set = {d for d in observed if d in APPROACHES}
    duration_sec = float(duration_sec)

    flows: dict[str, float] = {}
    mixes: dict[str, dict[str, float]] = {}
    queues: dict[str, float | None] = {}
    profiles: dict[str, list[float]] = {}

    for d in APPROACHES:
        if d not in observed_set:
            continue
        flows[d] = round(counts_to_vph(counts.get(d, 0), duration_sec), 1)
        mixes[d] = normalize_mix((class_counts or {}).get(d))
        q = (queue_est or {}).get(d)
        queues[d] = None if q is None else round(float(q), 1)
        bins = list((profile_counts or {}).get(d) or [])
        if bins:
            spans = list(profile_spans or [])
            profiles[d] = [
                round(counts_to_vph(c, spans[i] if i < len(spans) else profile_bins_sec), 1)
                for i, c in enumerate(bins)
            ]
        else:
            profiles[d] = []

    # Fallback for approaches the video does not cover: mirror the opposing
    # approach when it was observed, otherwise use the contract default.
    observed_mix_pool: dict[str, float] = {}
    for d in observed_set:
        for vt, n in ((class_counts or {}).get(d) or {}).items():
            observed_mix_pool[vt] = observed_mix_pool.get(vt, 0.0) + float(n)

    approaches: dict[str, dict[str, Any]] = {}
    turning_out: dict[str, dict[str, float]] = {}
    profile_out: dict[str, list[float]] = {}

    for d in APPROACHES:
        if d in observed_set:
            approaches[d] = {
                "flow_vph": flows[d],
                "queue_est": queues[d],
                "vehicle_mix": mixes[d],
                "observed": True,
            }
            profile_out[d] = profiles[d]
        else:
            src = MIRROR[d]
            if src in observed_set:
                approaches[d] = {
                    "flow_vph": flows[src],
                    "queue_est": queues[src],
                    "vehicle_mix": dict(mixes[src]),
                    "observed": False,
                }
                profile_out[d] = list(profiles[src])
            else:
                approaches[d] = {
                    "flow_vph": DEFAULT_FLOW_VPH,
                    "queue_est": None,
                    "vehicle_mix": normalize_mix(observed_mix_pool) if observed_mix_pool
                                   else dict(DEFAULT_MIX),
                    "observed": False,
                }
                profile_out[d] = []

        given = (turning_ratio or {}).get(d) or {}
        ratios = {mv: float(given.get(mv, DEFAULT_TURNING[mv])) for mv in DEFAULT_TURNING}
        total = sum(ratios.values())
        if total > 0:
            ratios = {mv: round(v / total, 6) for mv, v in ratios.items()}
        turning_out[d] = ratios

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "source": {
            "video": video,
            "fps": None if fps is None else round(float(fps), 3),
            "frames": None if frames is None else int(frames),
            "duration_sec": round(duration_sec, 3),
            "analyzed_at": analyzed_at or datetime.now().isoformat(timespec="seconds"),
        },
        "duration_sec": round(duration_sec, 3),
        "approaches": approaches,
        "turning_ratio": turning_out,
        "profile_bins_sec": int(profile_bins_sec),
        "flow_profile": profile_out,
    }


def validate_traffic_state(state: Mapping[str, Any], *, tol: float = 1e-6) -> list[str]:
    """Return a list of contract violations (empty list == valid)."""
    problems: list[str] = []

    if state.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION!r}, "
                        f"got {state.get('schema_version')!r}")
    if not str(state.get("scenario_id") or "").strip():
        problems.append("scenario_id is empty")

    src = state.get("source")
    if not isinstance(src, Mapping):
        problems.append("source must be an object")
    else:
        for key in ("video", "fps", "frames", "duration_sec", "analyzed_at"):
            if key not in src:
                problems.append(f"source.{key} missing")

    duration = state.get("duration_sec")
    if not isinstance(duration, (int, float)) or duration <= 0:
        problems.append(f"duration_sec must be a positive number, got {duration!r}")

    approaches = state.get("approaches")
    if not isinstance(approaches, Mapping):
        problems.append("approaches must be an object")
        return problems

    for d in APPROACHES:
        app = approaches.get(d)
        if not isinstance(app, Mapping):
            problems.append(f"approaches.{d} missing")
            continue
        flow = app.get("flow_vph")
        if not isinstance(flow, (int, float)) or flow < 0:
            problems.append(f"approaches.{d}.flow_vph must be a non-negative number, "
                            f"got {flow!r}")
        if not isinstance(app.get("observed"), bool):
            problems.append(f"approaches.{d}.observed must be a bool")
        q = app.get("queue_est", "__missing__")
        if q == "__missing__":
            problems.append(f"approaches.{d}.queue_est missing (may be null)")
        elif q is not None and not isinstance(q, (int, float)):
            problems.append(f"approaches.{d}.queue_est must be a number or null")
        mix = app.get("vehicle_mix")
        if not isinstance(mix, Mapping):
            problems.append(f"approaches.{d}.vehicle_mix missing")
        else:
            missing = [vt for vt in VEHICLE_TYPES if vt not in mix]
            if missing:
                problems.append(f"approaches.{d}.vehicle_mix missing keys {missing}")
            total = sum(float(mix.get(vt, 0.0)) for vt in VEHICLE_TYPES)
            if not math.isclose(total, 1.0, abs_tol=max(tol, 1e-6)):
                problems.append(f"approaches.{d}.vehicle_mix sums to {total:.9f}, not 1")

    turning = state.get("turning_ratio")
    if not isinstance(turning, Mapping):
        problems.append("turning_ratio must be an object")
    else:
        for d in APPROACHES:
            ratios = turning.get(d)
            if not isinstance(ratios, Mapping):
                problems.append(f"turning_ratio.{d} missing")
                continue
            total = sum(float(ratios.get(mv, 0.0)) for mv in DEFAULT_TURNING)
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                problems.append(f"turning_ratio.{d} sums to {total:.9f}, not 1")

    bins = state.get("profile_bins_sec")
    if not isinstance(bins, int) or bins <= 0:
        problems.append(f"profile_bins_sec must be a positive int, got {bins!r}")

    profile = state.get("flow_profile")
    if profile is not None:
        if not isinstance(profile, Mapping):
            problems.append("flow_profile must be an object or null")
        else:
            for d in APPROACHES:
                if d not in profile:
                    problems.append(f"flow_profile.{d} missing")
                    continue
                seq = profile[d]
                if seq is None:
                    continue
                if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
                    problems.append(f"flow_profile.{d} must be a list")
                elif any(not isinstance(v, (int, float)) for v in seq):
                    problems.append(f"flow_profile.{d} contains non-numeric values")

    return problems


def write_traffic_state(state: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path
