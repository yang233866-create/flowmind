from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from vision.frame_evidence import load_and_validate_frame_evidence


REQUIRED_METRICS = (
    "avg_waiting_s",
    "avg_travel_time_s",
    "throughput_veh",
    "avg_queue_veh",
    "max_queue_veh",
    "avg_speed_mps",
)
EXPECTED_SCENARIOS = (
    "normal",
    "morning_peak",
    "evening_peak",
    "event_surge",
    "lane_closure",
)
EXPECTED_STRATEGIES = ("fixed", "actuated", "dqn", "ppo")
TRAINING_RUNS = {
    "dqn": Path("data/results/tb/dqn_cross_basic/DQN_3"),
    "ppo": Path("data/results/tb/ppo_cross_basic/PPO_4"),
}


@dataclass(frozen=True)
class VisualizationData:
    root: Path
    arena: pd.DataFrame
    timeseries: dict[str, pd.DataFrame]
    run_meta: dict[str, dict[str, Any]]
    traffic_state: dict[str, Any]
    scenarios: dict[str, dict[str, Any]]
    training: dict[str, dict[str, pd.DataFrame]]
    annotated_frame: Path
    annotated_frame_meta: dict[str, Any]


def _load_scalar(run_dir: Path, tag: str) -> pd.DataFrame:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        raise ValueError(f"TensorBoard tag missing: {tag} in {run_dir}")
    events = accumulator.Scalars(tag)
    frame = pd.DataFrame({"step": [e.step for e in events], "value": [e.value for e in events]})
    if frame.empty or not np.isfinite(frame[["step", "value"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid TensorBoard curve: {tag} in {run_dir}")
    return frame


def _load_training(root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for algorithm, relative in TRAINING_RUNS.items():
        run_dir = root / relative
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        loss_tag = "train/loss" if algorithm == "dqn" else "train/value_loss"
        result[algorithm] = {
            "reward": _load_scalar(run_dir, "rollout/ep_rew_mean"),
            "loss": _load_scalar(run_dir, loss_tag),
        }
    return result


def _validate_arena(arena: pd.DataFrame) -> None:
    required = {"scenario", "strategy", "seed", "exp_id", *REQUIRED_METRICS}
    missing = required - set(arena.columns)
    if missing:
        raise ValueError(f"arena_summary missing columns: {sorted(missing)}")
    if len(arena) != 60:
        raise ValueError(f"Expected 60 formal runs, found {len(arena)}")
    if arena.duplicated(["scenario", "strategy", "seed"]).any():
        raise ValueError("Duplicate scenario/strategy/seed keys in arena_summary")
    if set(arena["scenario"]) != set(EXPECTED_SCENARIOS):
        raise ValueError("Scenario set does not match the formal five-scenario design")
    if set(arena["strategy"]) != set(EXPECTED_STRATEGIES):
        raise ValueError("Strategy set does not match fixed/actuated/dqn/ppo")
    cells = arena.groupby(["scenario", "strategy"], observed=True).size()
    if not (cells == 3).all():
        raise ValueError("Every scenario/strategy cell must contain exactly three seeds")
    if not np.isfinite(arena[list(REQUIRED_METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Formal metrics contain NaN or infinite values")


def load_visualization_data(root: str | Path) -> VisualizationData:
    root = Path(root).resolve()
    arena_path = root / "data/results/arena_summary.csv"
    arena = pd.read_csv(arena_path)
    arena["strategy"] = arena["strategy"].str.lower()
    arena["scenario"] = arena["scenario"].str.lower()
    _validate_arena(arena)

    timeseries: dict[str, pd.DataFrame] = {}
    run_meta: dict[str, dict[str, Any]] = {}
    scenarios: dict[str, dict[str, Any]] = {}
    for row in arena.itertuples(index=False):
        exp_dir = root / "data/results/experiments" / row.exp_id
        ts_path = exp_dir / "timeseries.csv"
        meta_path = exp_dir / "run_meta.json"
        scenario_path = exp_dir / "scenario_spec.json"
        for path in (ts_path, meta_path, scenario_path):
            if not path.exists():
                raise FileNotFoundError(path)
        ts = pd.read_csv(ts_path)
        required_ts = {"t", "queue_total", "queue_north", "queue_south", "queue_east", "queue_west"}
        if required_ts - set(ts):
            raise ValueError(f"{ts_path} missing {sorted(required_ts - set(ts))}")
        if not np.isfinite(ts[list(required_ts)].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite queue trace in {ts_path}")
        timeseries[row.exp_id] = ts
        run_meta[row.exp_id] = json.loads(meta_path.read_text(encoding="utf-8"))
        if row.scenario not in scenarios:
            scenarios[row.scenario] = json.loads(scenario_path.read_text(encoding="utf-8"))

    state_path = root / "data/traffic_states/demo_001.json"
    traffic_state = json.loads(state_path.read_text(encoding="utf-8"))
    if set(traffic_state.get("approaches", {})) != {"north", "south", "east", "west"}:
        raise ValueError("Traffic state must include all four approaches")

    annotated_frame = root / "figures/fig_vision_annotated_frame.png"
    annotated_frame_meta_path = annotated_frame.with_suffix(".meta.json")
    video_path = root / str(traffic_state["source"]["video"]).replace("\\", "/")
    roi_path = root / "data/videos/demo_roi.json"
    if not annotated_frame.exists():
        raise FileNotFoundError(annotated_frame)
    annotated_frame_meta = load_and_validate_frame_evidence(
        annotated_frame,
        annotated_frame_meta_path,
        video_path,
        roi_path,
        traffic_state,
    )

    return VisualizationData(
        root=root,
        arena=arena.sort_values(["scenario", "strategy", "seed"]).reset_index(drop=True),
        timeseries=timeseries,
        run_meta=run_meta,
        traffic_state=traffic_state,
        scenarios=scenarios,
        training=_load_training(root),
        annotated_frame=annotated_frame,
        annotated_frame_meta=annotated_frame_meta,
    )
