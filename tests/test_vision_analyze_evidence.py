import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import supervision as sv

from vision import analyze


def _detections(track_ids: list[int]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(
            [
                [5 + 12 * index, 8, 15 + 12 * index, 18]
                for index, _track_id in enumerate(track_ids)
            ],
            dtype=float,
        ).reshape(-1, 4),
        class_id=np.full(len(track_ids), 2, dtype=int),
        tracker_id=np.asarray(track_ids, dtype=int),
    )


class FakeCapture:
    def __init__(
        self,
        sequences: list[list[int]],
        reported_frames: int | None = None,
    ) -> None:
        self.sequences = sequences
        self.reported_frames = (
            len(sequences) if reported_frames is None else reported_frames
        )
        self.frame_index = 0

    def get(self, property_id: int) -> float:
        values = {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_COUNT: float(self.reported_frames),
            cv2.CAP_PROP_FRAME_WIDTH: 64.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
        }
        return values.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frame_index >= len(self.sequences):
            return False, None
        self.frame_index += 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8).copy()

    def release(self) -> None:
        pass


class FakeDetector:
    def describe(self) -> dict[str, bool]:
        return {"fake": True}

    def detect(self, _frame: np.ndarray) -> sv.Detections:
        return sv.Detections.empty()

    def class_name(self, _class_id: object) -> str:
        return "car"


class FakeTracker:
    def __init__(self, sequences: list[list[int]]) -> None:
        self.sequences = sequences
        self.frame_index = 0

    def update(self, _detections_from_detector: sv.Detections) -> sv.Detections:
        tracked = _detections(self.sequences[self.frame_index])
        self.frame_index += 1
        return tracked


class FakeCounter:
    def __init__(self, roi_config: dict[str, Any], fps: float) -> None:
        del fps
        self.observed = tuple((roi_config.get("count_lines") or {}).keys())
        self.zones = {direction: object() for direction in self.observed}

    def get_observed(self) -> tuple[str, ...]:
        return self.observed

    def get_sense(self) -> str:
        return "both"

    def update(self, _tracked: sv.Detections, _frame_index: int) -> None:
        pass

    def get_counts(self) -> dict[str, int]:
        return {direction: 0 for direction in self.observed}

    def get_class_counts(self) -> dict[str, dict[str, int]]:
        return {direction: {} for direction in self.observed}

    def bin_flow_profile(
        self,
        *,
        bin_sec: float,
        total_duration_sec: float,
    ) -> tuple[dict[str, list[int]], list[float]]:
        del bin_sec, total_duration_sec
        return {direction: [] for direction in self.observed}, []


class CountingTraceAnnotator(sv.TraceAnnotator):
    annotate_calls = 0

    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
    ) -> np.ndarray:
        type(self).annotate_calls += 1
        return super().annotate(scene=scene, detections=detections)


class CountingLineAnnotator:
    annotate_calls = 0

    def __init__(self, **_kwargs: object) -> None:
        pass

    def annotate(self, scene: np.ndarray, _zone: object) -> np.ndarray:
        type(self).annotate_calls += 1
        scene[0:2, :, 1] = 255
        return scene


def _run_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    sequences: list[list[int]],
    roi_config: dict[str, Any],
    reported_frames: int | None = None,
) -> tuple[Path, Path, Path]:
    CountingTraceAnnotator.annotate_calls = 0
    CountingLineAnnotator.annotate_calls = 0
    monkeypatch.setattr(
        analyze.cv2,
        "VideoCapture",
        lambda _path: FakeCapture(sequences, reported_frames),
    )
    monkeypatch.setattr(analyze, "VehicleDetector", FakeDetector)
    monkeypatch.setattr(
        analyze,
        "VehicleTracker",
        lambda *, fps: FakeTracker(sequences),
    )
    monkeypatch.setattr(analyze, "DirectionalCounter", FakeCounter)
    monkeypatch.setattr(analyze.sv, "TraceAnnotator", CountingTraceAnnotator)
    monkeypatch.setattr(analyze.sv, "LineZoneAnnotator", CountingLineAnnotator)
    monkeypatch.setattr(analyze, "_generate_figures", lambda *_args: None)

    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake-video")
    state_path = tmp_path / "candidate.json"
    result = analyze.analyze_video(
        video_path=video_path,
        roi_config=roi_config,
        scenario_id="candidate",
        state_out=state_path,
        annotated_video=None,
        figures_dir=tmp_path,
    )
    assert result == state_path
    return (
        state_path,
        tmp_path / "fig_vision_annotated_frame.png",
        tmp_path / "fig_vision_annotated_frame.meta.json",
    )


def test_figures_select_richest_fully_annotated_frame_without_video_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _state_path, frame_path, meta_path = _run_analysis(
        monkeypatch,
        tmp_path,
        sequences=[[1], [1, 2], [1]],
        roi_config={
            "count_lines": {
                "north": [[0, 10], [63, 10]],
                "south": [[0, 30], [63, 30]],
            }
        },
    )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert frame_path.is_file()
    assert metadata["frame"]["index"] == 1
    assert metadata["active_track_count"] == 2
    assert metadata["active_trace_point_count"] == 3
    assert metadata["no_detections"] is False
    assert CountingTraceAnnotator.annotate_calls == 3
    assert CountingLineAnnotator.annotate_calls == 6


def test_empty_frames_fall_back_to_first_fully_annotated_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _state_path, frame_path, meta_path = _run_analysis(
        monkeypatch,
        tmp_path,
        sequences=[[], []],
        roi_config={"count_lines": {"north": [[0, 10], [63, 10]]}},
    )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    annotated = cv2.imread(str(frame_path))
    assert metadata["frame"]["index"] == 0
    assert metadata["no_detections"] is True
    assert metadata["zoom"] is None
    assert annotated is not None
    assert np.all(annotated[0:2, :, 1] == 255)
    assert CountingTraceAnnotator.annotate_calls == 2
    assert CountingLineAnnotator.annotate_calls == 2


def test_figures_require_at_least_one_configured_count_line(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "fig_vision_annotated_frame.png"
    meta_path = tmp_path / "fig_vision_annotated_frame.meta.json"

    with pytest.raises(ValueError) as exc_info:
        _run_analysis(
            monkeypatch,
            tmp_path,
            sequences=[[]],
            roi_config={"count_lines": {}},
        )

    assert str(exc_info.value) == (
        "figure evidence requires at least one configured count line"
    )
    assert not frame_path.exists()
    assert not meta_path.exists()


def test_figures_reject_video_without_readable_frames_before_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "fig_vision_annotated_frame.png"
    meta_path = tmp_path / "fig_vision_annotated_frame.meta.json"

    with pytest.raises(ValueError) as exc_info:
        _run_analysis(
            monkeypatch,
            tmp_path,
            sequences=[],
            reported_frames=3,
            roi_config={"count_lines": {"north": [[0, 10], [63, 10]]}},
        )

    assert str(exc_info.value) == "video contained no readable frames"
    assert not frame_path.exists()
    assert not meta_path.exists()
