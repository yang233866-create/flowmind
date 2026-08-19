# FlowMind Paper Figures 01, 10 and 11 Evidence Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Figure 01 show a real, automatically selected annotated inference frame with auditable provenance, and remove the text collisions in Figures 10 and 11 without changing any TrafficState field, experiment artifact, metric, curve, or statistical result.

**Architecture:** Add one shared `vision/frame_evidence.py` boundary that owns deterministic frame selection, same-frame zoom metadata, hashing, persistence, and validation. `vision.analyze` produces candidate visual evidence during normal inference; a separate promotion gate compares the candidate TrafficState with the formal state and verifies protected-data hashes before replacing only the formal frame PNG/JSON. The plotting layer consumes the validated evidence and makes geometry-only changes to Figures 10 and 11.

**Tech Stack:** Python 3.12, OpenCV 5.0, supervision 0.30, NumPy, Pillow, Matplotlib, pytest, existing FlowMind visualization/export helpers.

---

## Scope and safety guardrails

- Figure 11 uses the user-confirmed **B layout**: names stay inside node boxes; every `veh/h` value moves immediately below its own box.
- Figure 01 selects frames by the exact lexicographic score `(active_track_count, active_trace_point_count, -frame_index)`. Do not add confidence, object size, class, visual centering, crossing count, or manual taste to the score.
- `sv.TraceAnnotator.annotate()` updates trace history and must run exactly once per processed frame.
- The full frame and inset must be pixels from the same saved PNG. Do not reopen the video to choose a second inset frame.
- Formal `data/traffic_states/demo_001.json`, `data/results/arena_summary.csv`, and `data/results/experiments/**` are protected. A visual rerun writes into an isolated candidate directory and may promote only `figures/fig_vision_annotated_frame.png` and its new `.meta.json` companion.
- Figures 02–09 remain outside this task. In particular, this visual work does not repair the already identified mixed-network and endpoint-censoring problems in the arena data, and no final handoff may describe those quantitative figures as scientifically validated.
- The worktree is already dirty and several target visualization files are currently untracked. Stage only the exact paths named in each task; never run `git add .`, and inspect `git diff --cached --stat` before every commit.

## Task 1: Add the deterministic representative-frame selector

**Files:**

- Create: `vision/frame_evidence.py`
- Create: `tests/test_frame_evidence.py`

- [ ] **Step 1: Write the selector tests first.**

Create `tests/test_frame_evidence.py` with helpers and the four ranking/fallback tests below:

```python
import numpy as np

from vision.frame_evidence import (
    RepresentativeFrameSelector,
    TrackEvidence,
)


def _image(value: int) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def _track(
    track_id: int,
    points: int,
    box: tuple[float, float, float, float] = (5.0, 5.0, 15.0, 15.0),
) -> TrackEvidence:
    return TrackEvidence(
        track_id=track_id,
        class_name="car",
        bbox_xyxy=box,
        trace_points=tuple((float(i), float(i)) for i in range(points)),
    )


def test_more_active_tracks_beat_more_trace_points():
    selector = RepresentativeFrameSelector(fps=10.0)
    selector.consider(
        annotated_bgr=_image(1), frame_index=0, tracks=(_track(1, 60),)
    )
    selector.consider(
        annotated_bgr=_image(2),
        frame_index=1,
        tracks=(_track(1, 1), _track(2, 1)),
    )
    assert selector.result().frame_index == 1


def test_trace_points_break_equal_track_count():
    selector = RepresentativeFrameSelector(fps=10.0)
    selector.consider(
        annotated_bgr=_image(1), frame_index=0, tracks=(_track(1, 2),)
    )
    selector.consider(
        annotated_bgr=_image(2), frame_index=1, tracks=(_track(1, 3),)
    )
    assert selector.result().frame_index == 1


def test_earlier_frame_breaks_full_tie_and_image_is_copied():
    first = _image(1)
    selector = RepresentativeFrameSelector(fps=10.0)
    selector.consider(annotated_bgr=first, frame_index=0, tracks=(_track(1, 2),))
    selector.consider(
        annotated_bgr=_image(2), frame_index=1, tracks=(_track(1, 2),)
    )
    first[:] = 9

    result = selector.result()

    assert result.frame_index == 0
    assert result.timestamp_sec == 0.0
    assert np.all(result.annotated_bgr == 1)


def test_first_fully_annotated_frame_is_no_detection_fallback():
    selector = RepresentativeFrameSelector(fps=10.0)
    selector.consider(annotated_bgr=_image(1), frame_index=0, tracks=())
    selector.consider(annotated_bgr=_image(2), frame_index=1, tracks=())

    result = selector.result()

    assert result.frame_index == 0
    assert result.no_detections is True
```

- [ ] **Step 2: Run the focused test and verify the expected RED failure.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\selector-red `
  tests/test_frame_evidence.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'vision.frame_evidence'`.

- [ ] **Step 3: Implement the smallest selector model.**

Start `vision/frame_evidence.py` with these constants and complete data model:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


SELECTOR_NAME = "active-tracks-trace-points"
SELECTOR_VERSION = "active-tracks-trace-points-v1"
TRACE_LENGTH = 60
ZOOM_MARGIN_RATIO = 0.12


@dataclass(frozen=True)
class TrackEvidence:
    track_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    trace_points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RepresentativeFrame:
    annotated_bgr: np.ndarray = field(repr=False, compare=False)
    frame_index: int
    timestamp_sec: float
    fps: float
    tracks: tuple[TrackEvidence, ...]

    @property
    def active_track_count(self) -> int:
        return len(self.tracks)

    @property
    def active_trace_point_count(self) -> int:
        return sum(len(track.trace_points) for track in self.tracks)

    @property
    def score(self) -> tuple[int, int, int]:
        return (
            self.active_track_count,
            self.active_trace_point_count,
            -self.frame_index,
        )

    @property
    def no_detections(self) -> bool:
        return self.active_track_count == 0


class RepresentativeFrameSelector:
    def __init__(self, *, fps: float) -> None:
        self._fps = float(fps)
        if self._fps <= 0:
            raise ValueError("fps must be positive")
        self._best: RepresentativeFrame | None = None

    def consider(
        self,
        *,
        annotated_bgr: np.ndarray,
        frame_index: int,
        tracks: Sequence[TrackEvidence],
    ) -> None:
        candidate = RepresentativeFrame(
            annotated_bgr=annotated_bgr.copy(),
            frame_index=int(frame_index),
            timestamp_sec=float(frame_index) / self._fps,
            fps=self._fps,
            tracks=tuple(tracks),
        )
        if self._best is None or candidate.score > self._best.score:
            self._best = candidate

    def result(self) -> RepresentativeFrame:
        if self._best is None:
            raise ValueError("video contained no readable frames")
        return self._best
```

- [ ] **Step 4: Run the same test and verify GREEN.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\selector-green `
  tests/test_frame_evidence.py
```

Expected: the four selector tests pass.

- [ ] **Step 5: Commit only the selector boundary and its tests.**

```powershell
git add vision/frame_evidence.py tests/test_frame_evidence.py
git diff --cached --stat
git commit -m "feat: select representative annotated frame"
```

## Task 2: Snapshot active traces and persist a strict evidence contract

**Files:**

- Modify: `vision/frame_evidence.py`
- Modify: `tests/test_frame_evidence.py`

- [ ] **Step 1: Add a real-supervision trace snapshot test.**

```python
import supervision as sv

from vision.frame_evidence import snapshot_active_tracks


def test_snapshot_uses_visible_trace_history_and_copies_values():
    trace = sv.TraceAnnotator(trace_length=60)
    detections = sv.Detections(
        xyxy=np.asarray([[4, 5, 14, 15]], dtype=np.float32),
        class_id=np.asarray([2], dtype=int),
        tracker_id=np.asarray([7], dtype=int),
    )
    trace.annotate(np.zeros((32, 32, 3), dtype=np.uint8), detections)

    tracks = snapshot_active_tracks(
        detections, trace, class_name=lambda class_id: "car"
    )

    assert tracks == (
        TrackEvidence(
            track_id=7,
            class_name="car",
            bbox_xyxy=(4.0, 5.0, 14.0, 15.0),
            trace_points=((9.0, 10.0),),
        ),
    )
```

Add this second snapshot test so pending IDs and ordering are executable requirements:

```python
def test_snapshot_discards_pending_ids_and_sorts_numeric_ids():
    trace = sv.TraceAnnotator(trace_length=60)
    detections = sv.Detections(
        xyxy=np.asarray(
            [[20, 20, 30, 30], [1, 1, 4, 4], [4, 6, 14, 16]],
            dtype=np.float32,
        ),
        class_id=np.asarray([2, 2, 7], dtype=int),
        tracker_id=np.asarray([8, -1, 3], dtype=int),
    )
    trace.annotate(np.zeros((40, 40, 3), dtype=np.uint8), detections)

    tracks = snapshot_active_tracks(
        detections,
        trace,
        class_name=lambda class_id: "truck" if int(class_id) == 7 else "car",
    )

    assert [track.track_id for track in tracks] == [3, 8]
    assert [track.class_name for track in tracks] == ["truck", "car"]
    assert all(len(track.trace_points) == 1 for track in tracks)
```

- [ ] **Step 2: Run that test and verify RED because the function is absent.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\snapshot-red `
  tests/test_frame_evidence.py -k snapshot
```

Expected: collection fails because `snapshot_active_tracks` is not yet defined.

- [ ] **Step 3: Add the trace snapshot implementation.**

```python
from collections.abc import Callable

import supervision as sv


def snapshot_active_tracks(
    tracked: sv.Detections,
    trace_annotator: sv.TraceAnnotator,
    *,
    class_name: Callable[[object], str],
) -> tuple[TrackEvidence, ...]:
    if tracked.tracker_id is None:
        return ()
    records: list[TrackEvidence] = []
    for index, raw_track_id in enumerate(tracked.tracker_id):
        track_id = int(raw_track_id)
        if track_id < 0:
            continue
        raw_points = np.asarray(
            trace_annotator.trace.get(tracker_id=track_id), dtype=float
        ).reshape(-1, 2)
        points = tuple(
            (float(x), float(y)) for x, y in raw_points[-TRACE_LENGTH:]
        )
        raw_class_id = None if tracked.class_id is None else tracked.class_id[index]
        records.append(
            TrackEvidence(
                track_id=track_id,
                class_name=class_name(raw_class_id),
                bbox_xyxy=tuple(float(value) for value in tracked.xyxy[index]),
                trace_points=points,
            )
        )
    return tuple(sorted(records, key=lambda record: record.track_id))
```

- [ ] **Step 4: Add tests for deterministic zoom selection.**

Add these tests for the remaining tie-break, boundary, and fallback rules:

```python
from vision.frame_evidence import RepresentativeFrame, select_zoom


def test_zoom_prefers_trace_then_area_then_smaller_track_id():
    representative = RepresentativeFrame(
        annotated_bgr=np.zeros((100, 200, 3), dtype=np.uint8),
        frame_index=8,
        timestamp_sec=0.8,
        fps=10.0,
        tracks=(
            _track(7, 4, (20.0, 20.0, 40.0, 40.0)),
            _track(3, 4, (50.0, 20.0, 90.0, 60.0)),
        ),
    )

    zoom = select_zoom(representative)

    assert zoom is not None
    assert zoom["track_id"] == 3
    x1, y1, x2, y2 = zoom["crop_xyxy"]
    assert 0 <= x1 < x2 <= 200
    assert 0 <= y1 < y2 <= 100


def test_zoom_smaller_track_id_breaks_trace_and_area_tie():
    representative = RepresentativeFrame(
        annotated_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_index=2,
        timestamp_sec=0.2,
        fps=10.0,
        tracks=(
            _track(9, 3, (20.0, 20.0, 40.0, 40.0)),
            _track(4, 3, (50.0, 50.0, 70.0, 70.0)),
        ),
    )

    zoom = select_zoom(representative)

    assert zoom is not None
    assert zoom["track_id"] == 4


def test_zoom_uses_half_open_integer_bounds_and_clips_to_image():
    representative = RepresentativeFrame(
        annotated_bgr=np.zeros((50, 80, 3), dtype=np.uint8),
        frame_index=1,
        timestamp_sec=0.1,
        fps=10.0,
        tracks=(
            TrackEvidence(
                track_id=1,
                class_name="car",
                bbox_xyxy=(0.5, 1.5, 79.5, 49.5),
                trace_points=((0.0, 0.0), (79.0, 49.0)),
            ),
        ),
    )

    zoom = select_zoom(representative)

    assert zoom == {"track_id": 1, "crop_xyxy": [0, 0, 80, 50]}
    assert all(isinstance(value, int) for value in zoom["crop_xyxy"])


def test_no_detection_frame_has_no_zoom():
    representative = RepresentativeFrame(
        annotated_bgr=np.zeros((20, 30, 3), dtype=np.uint8),
        frame_index=0,
        timestamp_sec=0.0,
        fps=10.0,
        tracks=(),
    )
    assert select_zoom(representative) is None
```

- [ ] **Step 5: Implement zoom selection exactly once in the evidence module.**

```python
import math
from typing import Any


def _bbox_area(track: TrackEvidence) -> float:
    x1, y1, x2, y2 = track.bbox_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_zoom(
    representative: RepresentativeFrame,
    *,
    margin_ratio: float = ZOOM_MARGIN_RATIO,
) -> dict[str, Any] | None:
    if not representative.tracks:
        return None
    track = max(
        representative.tracks,
        key=lambda item: (len(item.trace_points), _bbox_area(item), -item.track_id),
    )
    x1, y1, x2, y2 = track.bbox_xyxy
    if track.trace_points:
        xs = [point[0] for point in track.trace_points]
        ys = [point[1] for point in track.trace_points]
        x1, x2 = min(x1, min(xs)), max(x2, max(xs))
        y1, y2 = min(y1, min(ys)), max(y2, max(ys))
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    image_height, image_width = representative.annotated_bgr.shape[:2]
    crop = [
        max(0, math.floor(x1 - width * margin_ratio)),
        max(0, math.floor(y1 - height * margin_ratio)),
        min(image_width, math.ceil(x2 + width * margin_ratio)),
        min(image_height, math.ceil(y2 + height * margin_ratio)),
    ]
    if crop[0] >= crop[2] or crop[1] >= crop[3]:
        return None
    return {"track_id": track.track_id, "crop_xyxy": crop}
```

- [ ] **Step 6: Add round-trip and tamper-rejection tests for PNG/JSON evidence.**

Add these complete fixtures and core failure cases to `tests/test_frame_evidence.py`:

```python
import copy
import json
import re
from pathlib import Path

import pytest

from vision.frame_evidence import (
    load_and_validate_frame_evidence,
    write_frame_evidence,
)


def _write_valid_evidence(tmp_path: Path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"deterministic-video-payload")
    roi = {"count_lines": {"north": [[0, 5], [29, 5]]}}
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps(roi), encoding="utf-8")
    representative = RepresentativeFrame(
        annotated_bgr=np.full((20, 30, 3), 127, dtype=np.uint8),
        frame_index=2,
        timestamp_sec=0.2,
        fps=10.0,
        tracks=(
            TrackEvidence(
                track_id=7,
                class_name="car",
                bbox_xyxy=(5.0, 6.0, 15.0, 16.0),
                trace_points=((8.0, 8.0), (10.0, 10.0)),
            ),
        ),
    )
    frame_path = tmp_path / "frame.png"
    meta_path = tmp_path / "frame.meta.json"
    write_frame_evidence(
        representative,
        frame_path=frame_path,
        meta_path=meta_path,
        video_path=video_path,
        roi_config=roi,
    )
    traffic_state = {
        "source": {
            "video": video_path.as_posix(),
            "fps": 10.0,
            "frames": 3,
            "duration_sec": 0.3,
        }
    }
    return frame_path, meta_path, video_path, roi_path, traffic_state


def _rewrite_metadata(meta_path: Path, mutate):
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    mutate(metadata)
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_frame_evidence_round_trip_and_hash_shapes(tmp_path):
    frame_path, meta_path, video_path, roi_path, state = _write_valid_evidence(
        tmp_path
    )

    metadata = load_and_validate_frame_evidence(
        frame_path, meta_path, video_path, roi_path, state
    )

    assert metadata["frame"]["index"] == 2
    assert metadata["active_track_count"] == 1
    assert metadata["active_trace_point_count"] == 2
    assert metadata["zoom"]["track_id"] == 7
    for digest in (
        metadata["video"]["sha256"],
        metadata["roi"]["sha256"],
        metadata["frame"]["annotated_sha256"],
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda meta: meta["frame"].__setitem__("width", 31), "frame.width"),
        (
            lambda meta: meta["frame"].__setitem__("timestamp_sec", 0.25),
            "frame.timestamp_sec",
        ),
        (lambda meta: meta["frame"].__setitem__("fps", 11.0), "frame.fps"),
        (lambda meta: meta["frame"].__setitem__("index", 3), "frame.index"),
        (
            lambda meta: meta.__setitem__("active_track_count", 2),
            "active_track_count",
        ),
        (
            lambda meta: meta.__setitem__("active_trace_point_count", 99),
            "active_trace_point_count",
        ),
        (lambda meta: meta.__setitem__("no_detections", True), "no_detections"),
        (lambda meta: meta["zoom"].__setitem__("track_id", 99), "zoom.track_id"),
        (
            lambda meta: meta["zoom"].__setitem__("crop_xyxy", [0, 0, 31, 20]),
            "zoom.crop_xyxy",
        ),
    ],
)
def test_metadata_tampering_is_rejected(tmp_path, mutate, error):
    frame_path, meta_path, video_path, roi_path, state = _write_valid_evidence(
        tmp_path
    )
    _rewrite_metadata(meta_path, mutate)

    with pytest.raises(ValueError, match=re.escape(error)):
        load_and_validate_frame_evidence(
            frame_path, meta_path, video_path, roi_path, state
        )


def test_duplicate_track_id_is_rejected(tmp_path):
    frame_path, meta_path, video_path, roi_path, state = _write_valid_evidence(
        tmp_path
    )

    def duplicate(metadata):
        metadata["tracks"].append(copy.deepcopy(metadata["tracks"][0]))
        metadata["active_track_count"] = 2
        metadata["active_trace_point_count"] = 4

    _rewrite_metadata(meta_path, duplicate)
    with pytest.raises(ValueError, match="tracks.track_id"):
        load_and_validate_frame_evidence(
            frame_path, meta_path, video_path, roi_path, state
        )


@pytest.mark.parametrize("target", ["frame", "video", "roi"])
def test_file_or_roi_tampering_is_rejected(tmp_path, target):
    frame_path, meta_path, video_path, roi_path, state = _write_valid_evidence(
        tmp_path
    )
    if target == "frame":
        frame_path.write_bytes(frame_path.read_bytes() + b"tamper")
        expected = "frame.annotated_sha256"
    elif target == "video":
        video_path.write_bytes(b"different-video")
        expected = "video.sha256"
    else:
        roi_path.write_text('{"count_lines": {}}', encoding="utf-8")
        expected = "roi.sha256"

    with pytest.raises(ValueError, match=re.escape(expected)):
        load_and_validate_frame_evidence(
            frame_path, meta_path, video_path, roi_path, state
        )
```

- [ ] **Step 7: Implement shared hashing, writer, and validator functions.**

Extend the module imports before adding these functions:

```python
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import cv2
from PIL import Image
```

Use these exact public signatures:

```python
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_frame_evidence(
    representative: RepresentativeFrame,
    *,
    frame_path: str | Path,
    meta_path: str | Path,
    video_path: str | Path,
    roi_config: Mapping[str, Any],
) -> tuple[Path, Path]:
    frame_path = Path(frame_path)
    meta_path = Path(meta_path)
    video_path = Path(video_path)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(frame_path), representative.annotated_bgr):
        raise OSError(f"failed to write representative frame: {frame_path}")
    height, width = representative.annotated_bgr.shape[:2]
    metadata = {
        "selector": {"name": SELECTOR_NAME, "version": SELECTOR_VERSION},
        "video": {"path": video_path.as_posix(), "sha256": sha256_file(video_path)},
        "roi": {"sha256": canonical_json_sha256(roi_config)},
        "frame": {
            "index": representative.frame_index,
            "timestamp_sec": representative.timestamp_sec,
            "fps": representative.fps,
            "width": width,
            "height": height,
            "annotated_sha256": sha256_file(frame_path),
        },
        "active_track_count": representative.active_track_count,
        "active_trace_point_count": representative.active_trace_point_count,
        "tracks": [
            {
                "track_id": track.track_id,
                "class_name": track.class_name,
                "bbox_xyxy": list(track.bbox_xyxy),
                "trace_points": [list(point) for point in track.trace_points],
            }
            for track in representative.tracks
        ],
        "zoom": select_zoom(representative),
        "no_detections": representative.no_detections,
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return frame_path, meta_path
```

Add the complete shared validator below. Downstream code must call this function instead of copying any of its checks:

```python
def _evidence_error(field: str, detail: str) -> ValueError:
    return ValueError(f"{field}: {detail}")


def _normalise_path(value: object) -> str:
    return str(value).replace("\\", "/")


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def load_and_validate_frame_evidence(
    frame_path: str | Path,
    meta_path: str | Path,
    video_path: str | Path,
    roi_path: str | Path,
    traffic_state: Mapping[str, Any],
) -> dict[str, Any]:
    frame_path = Path(frame_path)
    meta_path = Path(meta_path)
    video_path = Path(video_path)
    roi_path = Path(roi_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("selector") != {
        "name": SELECTOR_NAME,
        "version": SELECTOR_VERSION,
    }:
        raise _evidence_error("selector", "unsupported name or version")
    expected_frame_hash = metadata.get("frame", {}).get("annotated_sha256")
    if expected_frame_hash != sha256_file(frame_path):
        raise _evidence_error("frame.annotated_sha256", "PNG hash mismatch")
    if metadata.get("video", {}).get("sha256") != sha256_file(video_path):
        raise _evidence_error("video.sha256", "source video hash mismatch")
    roi_config = json.loads(roi_path.read_text(encoding="utf-8"))
    if metadata.get("roi", {}).get("sha256") != canonical_json_sha256(roi_config):
        raise _evidence_error("roi.sha256", "ROI hash mismatch")

    source = traffic_state.get("source")
    if not isinstance(source, Mapping):
        raise _evidence_error("traffic_state.source", "missing object")
    recorded_video = metadata.get("video", {}).get("path")
    if _normalise_path(recorded_video) != _normalise_path(source.get("video")):
        raise _evidence_error("video.path", "does not match TrafficState source")

    frame = metadata.get("frame")
    if not isinstance(frame, Mapping):
        raise _evidence_error("frame", "missing object")
    with Image.open(frame_path) as image:
        image_width, image_height = image.size
        if image.format != "PNG":
            raise _evidence_error("frame", "evidence image is not PNG")
    if frame.get("width") != image_width:
        raise _evidence_error("frame.width", "does not match PNG")
    if frame.get("height") != image_height:
        raise _evidence_error("frame.height", "does not match PNG")

    fps = frame.get("fps")
    if not _finite_number(fps) or float(fps) <= 0:
        raise _evidence_error("frame.fps", "must be positive and finite")
    if not _finite_number(source.get("fps")) or not math.isclose(
        float(fps), float(source["fps"]), rel_tol=0.0, abs_tol=1e-9
    ):
        raise _evidence_error("frame.fps", "does not match TrafficState")
    frame_index = frame.get("index")
    source_frames = source.get("frames")
    if (
        not isinstance(frame_index, int)
        or isinstance(frame_index, bool)
        or not isinstance(source_frames, int)
        or not 0 <= frame_index < source_frames
    ):
        raise _evidence_error("frame.index", "outside TrafficState frame range")
    timestamp = frame.get("timestamp_sec")
    if not _finite_number(timestamp) or not math.isclose(
        float(timestamp), frame_index / float(fps), rel_tol=0.0, abs_tol=1e-9
    ):
        raise _evidence_error("frame.timestamp_sec", "must equal index / fps")

    tracks = metadata.get("tracks")
    if not isinstance(tracks, list):
        raise _evidence_error("tracks", "must be a list")
    track_ids: list[int] = []
    trace_point_count = 0
    for track in tracks:
        if not isinstance(track, Mapping):
            raise _evidence_error("tracks", "entries must be objects")
        track_id = track.get("track_id")
        if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0:
            raise _evidence_error("tracks.track_id", "must be a non-negative int")
        track_ids.append(track_id)
        if not isinstance(track.get("class_name"), str):
            raise _evidence_error("tracks.class_name", "must be a string")
        bbox = track.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(
            _finite_number(value) for value in bbox
        ):
            raise _evidence_error("tracks.bbox_xyxy", "must contain four numbers")
        x1, y1, x2, y2 = map(float, bbox)
        if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
            raise _evidence_error("tracks.bbox_xyxy", "outside PNG bounds")
        points = track.get("trace_points")
        if not isinstance(points, list) or len(points) > TRACE_LENGTH:
            raise _evidence_error("tracks.trace_points", "invalid visible history")
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or not all(
                _finite_number(value) for value in point
            ):
                raise _evidence_error("tracks.trace_points", "invalid point")
            x, y = map(float, point)
            if not (0 <= x <= image_width and 0 <= y <= image_height):
                raise _evidence_error("tracks.trace_points", "point outside PNG")
        trace_point_count += len(points)
    if len(track_ids) != len(set(track_ids)):
        raise _evidence_error("tracks.track_id", "duplicate ID")
    if metadata.get("active_track_count") != len(tracks):
        raise _evidence_error("active_track_count", "does not match tracks")
    if metadata.get("active_trace_point_count") != trace_point_count:
        raise _evidence_error(
            "active_trace_point_count", "does not match visible trace points"
        )
    no_detections = metadata.get("no_detections")
    if not isinstance(no_detections, bool) or no_detections != (len(tracks) == 0):
        raise _evidence_error("no_detections", "inconsistent with tracks")

    zoom = metadata.get("zoom")
    if zoom is not None:
        if not isinstance(zoom, Mapping) or zoom.get("track_id") not in track_ids:
            raise _evidence_error("zoom.track_id", "does not identify an active track")
        crop = zoom.get("crop_xyxy")
        if (
            not isinstance(crop, list)
            or len(crop) != 4
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in crop)
        ):
            raise _evidence_error("zoom.crop_xyxy", "must contain four integers")
        x1, y1, x2, y2 = crop
        if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
            raise _evidence_error("zoom.crop_xyxy", "outside PNG bounds")
    return metadata
```

- [ ] **Step 8: Run the full evidence unit suite and verify GREEN.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\evidence-green `
  tests/test_frame_evidence.py
```

- [ ] **Step 9: Commit the evidence contract.**

```powershell
git add vision/frame_evidence.py tests/test_frame_evidence.py
git diff --cached --stat
git commit -m "feat: persist auditable frame evidence"
```

## Task 3: Integrate representative evidence into every figure-producing inference

**Files:**

- Modify: `vision/analyze.py:123-183`
- Modify: `vision/analyze.py:235-303`
- Create: `tests/test_vision_analyze_evidence.py`

- [ ] **Step 1: Write a three-frame integration test with no annotated-video writer.**

Create `tests/test_vision_analyze_evidence.py` with a complete fake three-frame pipeline. It uses real `sv.Detections` and a real trace implementation while avoiding YOLO:

```python
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from vision import analyze


def _detections(track_ids):
    boxes = np.asarray(
        [
            [5 + 12 * index, 8, 15 + 12 * index, 18]
            for index in range(len(track_ids))
        ],
        dtype=np.float32,
    ).reshape(-1, 4)
    return sv.Detections(
        xyxy=boxes,
        class_id=np.asarray([2] * len(track_ids), dtype=int),
        tracker_id=np.asarray(track_ids, dtype=int),
    )


def _install_pipeline_fakes(monkeypatch, sequences, reported_frames=None):
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in sequences]
    real_trace_annotator = sv.TraceAnnotator

    class FakeCapture:
        def __init__(self, path):
            self.index = 0

        def get(self, prop):
            values = {
                cv2.CAP_PROP_FPS: 10.0,
                cv2.CAP_PROP_FRAME_COUNT: len(frames)
                if reported_frames is None
                else reported_frames,
                cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48,
            }
            return values.get(prop, 0)

        def read(self):
            if self.index >= len(frames):
                return False, None
            frame = frames[self.index].copy()
            self.index += 1
            return True, frame

        def release(self):
            return None

    class FakeDetector:
        def describe(self):
            return {"fake": True}

        def detect(self, frame):
            return sv.Detections.empty()

        @staticmethod
        def class_name(class_id):
            return "car"

    class FakeTracker:
        def __init__(self, fps):
            self.index = 0

        def update(self, detections):
            result = sequences[self.index]
            self.index += 1
            return result

    class FakeCounter:
        def __init__(self, roi_config, fps):
            self.observed = tuple(roi_config.get("count_lines", {}))
            self.zones = {direction: object() for direction in self.observed}

        def get_observed(self):
            return self.observed

        def get_sense(self):
            return "both"

        def update(self, tracked, frame_idx):
            return None

        def get_counts(self):
            return {direction: 0 for direction in self.observed}

        def get_class_counts(self):
            return {direction: {} for direction in self.observed}

        def bin_flow_profile(self, bin_sec, total_duration_sec):
            return {direction: [] for direction in self.observed}, []

    class CountingTraceAnnotator(real_trace_annotator):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0
            type(self).instances.append(self)

        def annotate(self, scene, detections, custom_color_lookup=None):
            self.calls += 1
            return super().annotate(scene, detections, custom_color_lookup)

    class CountingLineAnnotator:
        instances = []

        def __init__(self, **kwargs):
            self.calls = 0
            type(self).instances.append(self)

        def annotate(self, scene, line_zone):
            self.calls += 1
            scene[0:2, :, 1] = 255
            return scene

    monkeypatch.setattr(analyze.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(analyze, "VehicleDetector", FakeDetector)
    monkeypatch.setattr(analyze, "VehicleTracker", FakeTracker)
    monkeypatch.setattr(analyze, "DirectionalCounter", FakeCounter)
    monkeypatch.setattr(analyze.sv, "TraceAnnotator", CountingTraceAnnotator)
    monkeypatch.setattr(analyze.sv, "LineZoneAnnotator", CountingLineAnnotator)
    monkeypatch.setattr(analyze, "_generate_figures", lambda *args, **kwargs: None)
    return CountingTraceAnnotator, CountingLineAnnotator


def _run_analysis(tmp_path, roi_config):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake-video")
    analyze.analyze_video(
        video_path=video_path,
        roi_config=roi_config,
        scenario_id="candidate",
        state_out=tmp_path / "candidate.json",
        annotated_video=None,
        figures_dir=tmp_path,
    )


def test_figures_dir_selects_richest_annotated_frame_without_video_writer(
    tmp_path, monkeypatch
):
    sequences = [_detections([1]), _detections([1, 2]), _detections([1])]
    trace_spy, line_spy = _install_pipeline_fakes(monkeypatch, sequences)
    roi_config = {
        "count_lines": {
            "north": [[0, 10], [63, 10]],
            "south": [[0, 30], [63, 30]],
        }
    }

    _run_analysis(tmp_path, roi_config)

    metadata = json.loads(
        (tmp_path / "fig_vision_annotated_frame.meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["frame"]["index"] == 1
    assert metadata["active_track_count"] == 2
    assert metadata["active_trace_point_count"] == 3
    assert metadata["no_detections"] is False
    assert (tmp_path / "fig_vision_annotated_frame.png").exists()
    assert trace_spy.instances[0].calls == 3
    assert sum(instance.calls for instance in line_spy.instances) == 6
```

The `trace_spy` assertion locks the requirement that `TraceAnnotator.annotate()` runs exactly once per readable frame. The line-spy total proves both configured count lines are rendered for all three frames even though `annotated_video=None`.

- [ ] **Step 2: Run the integration test and verify RED.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\analyze-red `
  tests/test_vision_analyze_evidence.py
```

Expected: no metadata is emitted and the current code saves the unannotated first frame.

- [ ] **Step 3: Move annotation behind a shared `needs_annotations` condition.**

Import the evidence interfaces, remove `first_frame`, create the selector only when `figures_dir` is requested, and use this loop body:

```python
selector = RepresentativeFrameSelector(fps=fps) if figures_dir is not None else None
needs_annotations = writer is not None or selector is not None

# Inside the frame loop, after tracker and counter updates:
if needs_annotations:
    annotated = frame.copy()
    annotated = trace_annotator.annotate(annotated, tracked)
    annotated = box_annotator.annotate(annotated, tracked)
    labels = [
        detector.class_name(class_id)
        for class_id in (
            tracked.class_id if tracked.class_id is not None else []
        )
    ]
    if labels:
        annotated = label_annotator.annotate(annotated, tracked, labels=labels)
    for direction in counter.get_observed():
        annotated = line_annotators[direction].annotate(
            annotated, counter.zones[direction]
        )
    if writer is not None:
        writer.write(annotated)
    if selector is not None:
        selector.consider(
            annotated_bgr=annotated,
            frame_index=frame_idx,
            tracks=snapshot_active_tracks(
                tracked,
                trace_annotator,
                class_name=detector.class_name,
            ),
        )
```

Immediately after constructing the selector and counter, add the tested guard:

```python
if selector is not None and not counter.get_observed():
    raise ValueError("figure evidence requires at least one configured count line")
```

- [ ] **Step 4: Persist the winning frame and remove the misleading raw-frame plot.**

Immediately after releasing the capture/writer and before TrafficState construction, resolve the selector so an unreadable video fails with the evidence-specific error before the zero-duration state error:

```python
representative = selector.result() if selector is not None else None
```

After creating `figures_dir`, write:

```python
if representative is None:
    raise RuntimeError("representative frame was not initialised")
write_frame_evidence(
    representative,
    frame_path=figures_dir / "fig_vision_annotated_frame.png",
    meta_path=figures_dir / "fig_vision_annotated_frame.meta.json",
    video_path=video_path,
    roi_config=roi_config,
)
_generate_figures(state, track_positions, (w, h), figures_dir, scenario_id)
```

Remove the `first_frame` parameter and the entire Matplotlib block titled `First Frame with Count Lines` from `_generate_figures()`. Keep flow timeline, vehicle mix, and heatmap calculations unchanged.

- [ ] **Step 5: Add the all-empty and unreadable-video integration cases.**

Add these exact tests to the same file:

```python
def test_no_detection_fallback_keeps_first_frame_and_count_lines(
    tmp_path, monkeypatch
):
    sequences = [_detections([]), _detections([])]
    _install_pipeline_fakes(monkeypatch, sequences)
    roi_config = {"count_lines": {"north": [[0, 10], [63, 10]]}}

    _run_analysis(tmp_path, roi_config)

    metadata = json.loads(
        (tmp_path / "fig_vision_annotated_frame.meta.json").read_text(
            encoding="utf-8"
        )
    )
    image = cv2.imread(str(tmp_path / "fig_vision_annotated_frame.png"))
    assert metadata["frame"]["index"] == 0
    assert metadata["no_detections"] is True
    assert metadata["zoom"] is None
    assert np.all(image[0:2, :, 1] == 255)


def test_empty_roi_fails_before_creating_evidence(tmp_path, monkeypatch):
    _install_pipeline_fakes(monkeypatch, [_detections([1])])

    with pytest.raises(
        ValueError, match="figure evidence requires at least one configured count line"
    ):
        _run_analysis(tmp_path, {"count_lines": {}})

    assert not (tmp_path / "fig_vision_annotated_frame.png").exists()
    assert not (tmp_path / "fig_vision_annotated_frame.meta.json").exists()


def test_unreadable_video_fails_without_partial_evidence(tmp_path, monkeypatch):
    _install_pipeline_fakes(monkeypatch, [], reported_frames=3)

    with pytest.raises(ValueError, match="video contained no readable frames"):
        _run_analysis(
            tmp_path,
            {"count_lines": {"north": [[0, 10], [63, 10]]}},
        )

    assert not (tmp_path / "fig_vision_annotated_frame.png").exists()
    assert not (tmp_path / "fig_vision_annotated_frame.meta.json").exists()
```

- [ ] **Step 6: Run vision regressions and verify GREEN.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\vision-green `
  tests/test_frame_evidence.py `
  tests/test_vision_analyze_evidence.py `
  tests/test_counter.py `
  tests/test_traffic_state.py
```

- [ ] **Step 7: Commit only integration code and tests.**

```powershell
git add vision/analyze.py vision/frame_evidence.py tests/test_vision_analyze_evidence.py
git diff --cached --stat
git commit -m "feat: integrate representative frame evidence"
```

## Task 4: Add the protected-data promotion gate

**Files:**

- Create: `scripts/promote_vision_evidence.py`
- Create: `tests/test_vision_evidence_gate.py`

- [ ] **Step 1: Write semantic-view tests before the gate.**

Start `tests/test_vision_evidence_gate.py` with the following imports, state factory, and semantic tests. The projection deliberately excludes raw counts because the formal state does not persist them:

```python
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.promote_vision_evidence import (
    _replace_pair_with_rollback,
    main,
    promote_vision_evidence,
    semantic_differences,
    snapshot_protected_data,
    traffic_state_semantic_view,
    verify_protected_data,
)
from vision.frame_evidence import (
    RepresentativeFrame,
    TrackEvidence,
    write_frame_evidence,
)


def _state():
    mix = {"car": 1.0, "bus": 0.0, "truck": 0.0, "motorcycle": 0.0}
    turns = {"left": 0.15, "straight": 0.70, "right": 0.15}
    return {
        "schema_version": "1.1",
        "scenario_id": "demo_001",
        "duration_sec": 1.0,
        "approaches": {
            direction: {"observed": direction in {"north", "south"}, "flow_vph": 400.0, "queue_est": None, "vehicle_mix": dict(mix)}
            for direction in ("north", "south", "east", "west")
        },
        "turning_ratio": {
            direction: dict(turns)
            for direction in ("north", "south", "east", "west")
        },
        "flow_profile": {
            "north": [400.0], "south": [400.0], "east": [], "west": []
        },
        "profile_bins_sec": 5,
        "source": {
            "video": "data/videos/demo.mp4",
            "fps": 10.0,
            "frames": 10,
            "duration_sec": 1.0,
            "analyzed_at": "2026-08-19T00:00:00",
        },
    }


def _set_nested(value, path, replacement):
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


def test_semantic_view_ignores_timestamp_and_source_path_only():
    reference = _state()
    candidate = copy.deepcopy(reference)
    candidate["source"]["analyzed_at"] = "2099-01-01T00:00:00"
    candidate["source"]["video"] = "temporary/candidate.mp4"
    assert semantic_differences(reference, candidate) == []


@pytest.mark.parametrize(
    ("path", "replacement", "expected_path"),
    [
        (("duration_sec",), 2.0, "duration_sec"),
        (("approaches", "north", "flow_vph"), 999.0, "approaches.north.flow_vph"),
        (("approaches", "east", "observed"), True, "approaches.east.observed"),
        (("turning_ratio", "north", "left"), 0.25, "turning_ratio.north.left"),
        (("flow_profile", "north"), [0.0], "flow_profile.north[0]"),
        (("profile_bins_sec",), 10, "profile_bins_sec"),
        (("source", "fps"), 11.0, "source.fps"),
        (("source", "frames"), 9, "source.frames"),
        (("source", "duration_sec"), 0.9, "source.duration_sec"),
    ],
)
def test_semantic_gate_reports_each_scientific_change(
    path, replacement, expected_path
):
    reference = _state()
    candidate = copy.deepcopy(reference)
    _set_nested(candidate, path, replacement)
    assert expected_path in semantic_differences(reference, candidate)
```

- [ ] **Step 2: Run the gate test and verify RED because the module is absent.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\gate-red `
  tests/test_vision_evidence_gate.py
```

- [ ] **Step 3: Implement deterministic file/tree snapshots.**

Use these protected paths:

```python
PROTECTED_FILES = (
    "data/traffic_states/demo_001.json",
    "data/results/arena_summary.csv",
)
PROTECTED_TREES = ("data/results/experiments",)
```

The tree digest must include each relative POSIX path and its file digest so renames cannot collide:

```python
def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
```

Add the snapshot and verification functions exactly as follows:

```python
def snapshot_protected_data(root: Path) -> dict[str, dict[str, str]]:
    root = Path(root).resolve()
    return {
        "files": {
            relative: sha256_file(root / relative) for relative in PROTECTED_FILES
        },
        "trees": {
            relative: sha256_tree(root / relative) for relative in PROTECTED_TREES
        },
    }


def verify_protected_data(
    root: Path, expected: Mapping[str, Any]
) -> None:
    actual = snapshot_protected_data(root)
    changed = []
    for group in ("files", "trees"):
        expected_group = expected.get(group, {})
        actual_group = actual.get(group, {})
        for relative in sorted(set(expected_group) | set(actual_group)):
            if expected_group.get(relative) != actual_group.get(relative):
                changed.append(relative)
    if changed:
        raise ValueError("protected data changed: " + ", ".join(changed))
```

Add this executable snapshot test:

```python
def test_protected_snapshot_detects_file_and_tree_changes(tmp_path):
    root = tmp_path / "root"
    (root / "data/traffic_states").mkdir(parents=True)
    (root / "data/results/experiments/run").mkdir(parents=True)
    (root / "data/traffic_states/demo_001.json").write_text("{}", encoding="utf-8")
    (root / "data/results/arena_summary.csv").write_text("a\n1\n", encoding="utf-8")
    experiment = root / "data/results/experiments/run/result.json"
    experiment.write_text("{}", encoding="utf-8")
    snapshot = snapshot_protected_data(root)

    verify_protected_data(root, snapshot)
    experiment.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="data/results/experiments"):
        verify_protected_data(root, snapshot)
```

- [ ] **Step 4: Write failure-atomic promotion tests.**

Add a complete temporary repository factory and the promotion tests:

```python
def _promotion_case(tmp_path):
    root = tmp_path / "root"
    (root / "data/traffic_states").mkdir(parents=True)
    (root / "data/results/experiments/run").mkdir(parents=True)
    (root / "data/videos").mkdir(parents=True)
    (root / "figures").mkdir(parents=True)
    reference = _state()
    reference_path = root / "data/traffic_states/demo_001.json"
    reference_path.write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    (root / "data/results/arena_summary.csv").write_text(
        "scenario,strategy,seed\nnormal,fixed,0\n", encoding="utf-8"
    )
    (root / "data/results/experiments/run/result.json").write_text(
        '{"value": 1}\n', encoding="utf-8"
    )
    video_path = root / "data/videos/demo.mp4"
    video_path.write_bytes(b"formal-video")
    roi = {"count_lines": {"north": [[0, 5], [29, 5]]}}
    roi_path = root / "data/videos/demo_roi.json"
    roi_path.write_text(json.dumps(roi), encoding="utf-8")
    formal_frame = root / "figures/fig_vision_annotated_frame.png"
    formal_meta = root / "figures/fig_vision_annotated_frame.meta.json"
    formal_frame.write_bytes(b"old-frame")
    formal_meta.write_bytes(b"old-meta")

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    candidate_state = copy.deepcopy(reference)
    candidate_state["source"]["analyzed_at"] = "2026-08-19T01:00:00"
    candidate_state_path = candidate_dir / "demo_001.json"
    candidate_state_path.write_text(
        json.dumps(candidate_state, indent=2) + "\n", encoding="utf-8"
    )
    representative = RepresentativeFrame(
        annotated_bgr=np.full((20, 30, 3), 80, dtype=np.uint8),
        frame_index=2,
        timestamp_sec=0.2,
        fps=10.0,
        tracks=(
            TrackEvidence(
                track_id=5,
                class_name="car",
                bbox_xyxy=(5.0, 6.0, 15.0, 16.0),
                trace_points=((8.0, 8.0),),
            ),
        ),
    )
    candidate_frame = candidate_dir / "frame.png"
    candidate_meta = candidate_dir / "frame.meta.json"
    write_frame_evidence(
        representative,
        frame_path=candidate_frame,
        meta_path=candidate_meta,
        video_path=video_path,
        roi_config=roi,
    )
    metadata = json.loads(candidate_meta.read_text(encoding="utf-8"))
    metadata["video"]["path"] = "data/videos/demo.mp4"
    candidate_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "root": root,
        "reference_path": reference_path,
        "formal_frame": formal_frame,
        "formal_meta": formal_meta,
        "candidate_state": candidate_state_path,
        "candidate_frame": candidate_frame,
        "candidate_meta": candidate_meta,
        "snapshot": snapshot_protected_data(root),
    }


def _promote(case):
    return promote_vision_evidence(
        root=case["root"],
        candidate_state_path=case["candidate_state"],
        candidate_frame_path=case["candidate_frame"],
        candidate_meta_path=case["candidate_meta"],
        protected_snapshot=case["snapshot"],
    )


def test_scientific_mismatch_does_not_touch_formal_assets(tmp_path):
    case = _promotion_case(tmp_path)
    old_frame = case["formal_frame"].read_bytes()
    old_meta = case["formal_meta"].read_bytes()
    candidate = json.loads(case["candidate_state"].read_text(encoding="utf-8"))
    candidate["approaches"]["north"]["flow_vph"] = 999.0
    case["candidate_state"].write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(ValueError, match="approaches.north.flow_vph"):
        _promote(case)

    assert case["formal_frame"].read_bytes() == old_frame
    assert case["formal_meta"].read_bytes() == old_meta


def test_protected_change_does_not_touch_formal_assets(tmp_path):
    case = _promotion_case(tmp_path)
    old_pair = (
        case["formal_frame"].read_bytes(),
        case["formal_meta"].read_bytes(),
    )
    (case["root"] / "data/results/arena_summary.csv").write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="arena_summary.csv"):
        _promote(case)

    assert case["formal_frame"].read_bytes() == old_pair[0]
    assert case["formal_meta"].read_bytes() == old_pair[1]


def test_invalid_candidate_provenance_does_not_touch_formal_assets(tmp_path):
    case = _promotion_case(tmp_path)
    old_pair = (
        case["formal_frame"].read_bytes(),
        case["formal_meta"].read_bytes(),
    )
    case["candidate_frame"].write_bytes(
        case["candidate_frame"].read_bytes() + b"tamper"
    )

    with pytest.raises(ValueError, match="frame.annotated_sha256"):
        _promote(case)

    assert case["formal_frame"].read_bytes() == old_pair[0]
    assert case["formal_meta"].read_bytes() == old_pair[1]


def test_successful_promotion_changes_only_visual_pair(tmp_path):
    case = _promotion_case(tmp_path)
    protected_before = copy.deepcopy(case["snapshot"])
    state_before = case["reference_path"].read_bytes()
    candidate_frame = case["candidate_frame"].read_bytes()
    candidate_meta = case["candidate_meta"].read_bytes()

    promoted_frame, promoted_meta = _promote(case)

    assert promoted_frame.read_bytes() == candidate_frame
    assert promoted_meta.read_bytes() == candidate_meta
    assert case["reference_path"].read_bytes() == state_before
    assert snapshot_protected_data(case["root"]) == protected_before


def test_pair_helper_rolls_back_when_final_verification_fails(tmp_path):
    source_frame = tmp_path / "candidate.png"
    source_meta = tmp_path / "candidate.json"
    destination_frame = tmp_path / "formal.png"
    destination_meta = tmp_path / "formal.json"
    source_frame.write_bytes(b"new-frame")
    source_meta.write_bytes(b"new-meta")
    destination_frame.write_bytes(b"old-frame")
    destination_meta.write_bytes(b"old-meta")

    def fail_after_replace():
        raise ValueError("post-replace verification failed")

    with pytest.raises(ValueError, match="post-replace"):
        _replace_pair_with_rollback(
            source_frame,
            source_meta,
            destination_frame,
            destination_meta,
            fail_after_replace,
        )

    assert destination_frame.read_bytes() == b"old-frame"
    assert destination_meta.read_bytes() == b"old-meta"


def test_snapshot_and_verify_cli_round_trip(tmp_path):
    case = _promotion_case(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"

    assert main(
        ["snapshot", "--root", str(case["root"]), "--out", str(snapshot_path)]
    ) == 0
    assert main(
        [
            "verify",
            "--root",
            str(case["root"]),
            "--snapshot",
            str(snapshot_path),
        ]
    ) == 0
```

- [ ] **Step 5: Implement the gate and three CLI subcommands.**

Create `scripts/promote_vision_evidence.py` with these imports, constants, and semantic projection:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vision.frame_evidence import (
    load_and_validate_frame_evidence,
    sha256_file,
)


PROTECTED_FILES = (
    "data/traffic_states/demo_001.json",
    "data/results/arena_summary.csv",
)
PROTECTED_TREES = ("data/results/experiments",)


def traffic_state_semantic_view(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": state["schema_version"],
        "duration_sec": state["duration_sec"],
        "approaches": {
            direction: {
                key: state["approaches"][direction][key]
                for key in ("observed", "flow_vph", "queue_est", "vehicle_mix")
            }
            for direction in ("north", "south", "east", "west")
        },
        "turning_ratio": state["turning_ratio"],
        "flow_profile": state["flow_profile"],
        "profile_bins_sec": state["profile_bins_sec"],
        "source": {
            key: state["source"][key]
            for key in ("fps", "frames", "duration_sec")
        },
    }
```

Use these public functions together with `sha256_tree`, `snapshot_protected_data`, and `verify_protected_data` from Step 3:

```python
def semantic_differences(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[str]:
    reference_view = traffic_state_semantic_view(reference)
    candidate_view = traffic_state_semantic_view(candidate)
    return _dotted_differences(reference_view, candidate_view)


def promote_vision_evidence(
    *,
    root: Path,
    candidate_state_path: Path,
    candidate_frame_path: Path,
    candidate_meta_path: Path,
    protected_snapshot: Mapping[str, Any],
) -> tuple[Path, Path]:
    reference_state_path = root / "data/traffic_states/demo_001.json"
    reference = json.loads(reference_state_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_state_path.read_text(encoding="utf-8"))
    differences = semantic_differences(reference, candidate)
    if differences:
        raise ValueError("TrafficState semantic mismatch: " + ", ".join(differences))
    verify_protected_data(root, protected_snapshot)
    video_relative = str(reference["source"]["video"]).replace("\\", "/")
    video_path = root / Path(video_relative)
    roi_path = root / "data/videos/demo_roi.json"
    load_and_validate_frame_evidence(
        candidate_frame_path,
        candidate_meta_path,
        video_path,
        roi_path,
        candidate,
    )
    destination_frame = root / "figures/fig_vision_annotated_frame.png"
    destination_meta = root / "figures/fig_vision_annotated_frame.meta.json"
    def verify_after_replace():
        load_and_validate_frame_evidence(
            destination_frame,
            destination_meta,
            video_path,
            roi_path,
            candidate,
        )
        verify_protected_data(root, protected_snapshot)
    _replace_pair_with_rollback(
        candidate_frame_path,
        candidate_meta_path,
        destination_frame,
        destination_meta,
        verify_after_replace,
    )
    return destination_frame, destination_meta
```

Implement the dotted comparator without numeric tolerances so a scientific-field drift cannot be rounded away:

```python
def _dotted_differences(reference, candidate, prefix=""):
    if isinstance(reference, dict) and isinstance(candidate, dict):
        differences = []
        for key in sorted(set(reference) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                differences.append(path)
            else:
                differences.extend(
                    _dotted_differences(reference[key], candidate[key], path)
                )
        return differences
    if isinstance(reference, list) and isinstance(candidate, list):
        differences = []
        if len(reference) != len(candidate):
            differences.append(prefix)
            return differences
        for index, (reference_item, candidate_item) in enumerate(
            zip(reference, candidate)
        ):
            differences.extend(
                _dotted_differences(
                    reference_item, candidate_item, f"{prefix}[{index}]"
                )
            )
        return differences
    return [] if reference == candidate else [prefix]
```

Add the complete failure-atomic pair replacement. It runs the final verification before discarding backups:

```python
def _replace_pair_with_rollback(
    source_frame: Path,
    source_meta: Path,
    destination_frame: Path,
    destination_meta: Path,
    verify_after_replace: Callable[[], None],
) -> None:
    destination_frame.parent.mkdir(parents=True, exist_ok=True)
    destination_meta.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged_frame = destination_frame.with_name(
        f".{destination_frame.name}.{token}.tmp"
    )
    staged_meta = destination_meta.with_name(f".{destination_meta.name}.{token}.tmp")
    previous = {
        destination_frame: destination_frame.read_bytes()
        if destination_frame.exists()
        else None,
        destination_meta: destination_meta.read_bytes()
        if destination_meta.exists()
        else None,
    }
    try:
        shutil.copy2(source_frame, staged_frame)
        shutil.copy2(source_meta, staged_meta)
        os.replace(staged_frame, destination_frame)
        os.replace(staged_meta, destination_meta)
        verify_after_replace()
    except Exception:
        for destination, payload in previous.items():
            if payload is None:
                destination.unlink(missing_ok=True)
                continue
            restore = destination.with_name(f".{destination.name}.{token}.restore")
            restore.write_bytes(payload)
            os.replace(restore, destination)
        raise
    finally:
        staged_frame.unlink(missing_ok=True)
        staged_meta.unlink(missing_ok=True)
```

Finish the module with the exact CLI parser and dispatcher:

```python
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protect and promote vision evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--root", type=Path, required=True)
    snapshot.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--snapshot", type=Path, required=True)
    promote = commands.add_parser("promote")
    promote.add_argument("--root", type=Path, required=True)
    promote.add_argument("--candidate-state", type=Path, required=True)
    promote.add_argument("--candidate-frame", type=Path, required=True)
    promote.add_argument("--candidate-meta", type=Path, required=True)
    promote.add_argument("--snapshot", type=Path, required=True)
    return parser


def _read_snapshot(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "snapshot":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(snapshot_protected_data(root), indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    protected_snapshot = _read_snapshot(args.snapshot)
    if args.command == "verify":
        verify_protected_data(root, protected_snapshot)
        return 0
    promote_vision_evidence(
        root=root,
        candidate_state_path=args.candidate_state,
        candidate_frame_path=args.candidate_frame,
        candidate_meta_path=args.candidate_meta,
        protected_snapshot=protected_snapshot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The finished CLI supports these concrete commands:

```text
python -m scripts.promote_vision_evidence snapshot --root . --out outputs/vision_rerun_candidate/protected_before.json
python -m scripts.promote_vision_evidence promote --root . --candidate-state outputs/vision_rerun_candidate/demo_001.json --candidate-frame outputs/vision_rerun_candidate/figures/fig_vision_annotated_frame.png --candidate-meta outputs/vision_rerun_candidate/figures/fig_vision_annotated_frame.meta.json --snapshot outputs/vision_rerun_candidate/protected_before.json
python -m scripts.promote_vision_evidence verify --root . --snapshot outputs/vision_rerun_candidate/protected_before.json
```

- [ ] **Step 6: Run all gate tests and verify GREEN.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\gate-green `
  tests/test_vision_evidence_gate.py
```

Expected: every semantic, snapshot, rollback, promotion, and CLI test passes.

- [ ] **Step 7: Commit only the gate and tests.**

```powershell
git add scripts/promote_vision_evidence.py tests/test_vision_evidence_gate.py
git diff --cached --stat
git commit -m "feat: gate vision evidence on unchanged data"
```

## Task 5: Perform the isolated visual rerun and promote only evidence assets

**Files:**

- Modify after successful gate only: `figures/fig_vision_annotated_frame.png`
- Create after successful gate only: `figures/fig_vision_annotated_frame.meta.json`
- Candidate-only, never commit: `outputs/vision_rerun_candidate/**`

- [ ] **Step 1: Snapshot the current dirty protected data before inference.**

```powershell
.\.venv\Scripts\python.exe -m scripts.promote_vision_evidence snapshot `
  --root . `
  --out outputs\vision_rerun_candidate\protected_before.json
```

- [ ] **Step 2: Run the real visual pipeline in isolation.**

```powershell
.\.venv\Scripts\python.exe -m vision.analyze `
  --video data\videos\demo.mp4 `
  --roi-config data\videos\demo_roi.json `
  --state-out outputs\vision_rerun_candidate\demo_001.json `
  --figures-dir outputs\vision_rerun_candidate\figures
```

Do not pass the formal TrafficState or formal figure directory to this command.

- [ ] **Step 3: Validate and promote only the PNG/JSON pair.**

```powershell
.\.venv\Scripts\python.exe -m scripts.promote_vision_evidence promote `
  --root . `
  --candidate-state outputs\vision_rerun_candidate\demo_001.json `
  --candidate-frame outputs\vision_rerun_candidate\figures\fig_vision_annotated_frame.png `
  --candidate-meta outputs\vision_rerun_candidate\figures\fig_vision_annotated_frame.meta.json `
  --snapshot outputs\vision_rerun_candidate\protected_before.json
```

If this command reports any semantic or protected-data difference, stop. Do not hand-copy the image, weaken the comparator, or overwrite the formal TrafficState.

- [ ] **Step 4: Verify that Git sees no protected-data changes introduced by this task.**

Record the pre-existing dirty paths separately, then assert this task added only the two formal evidence assets. Inspect both metadata and image dimensions:

```powershell
git status --short
.\.venv\Scripts\python.exe -c "import json; from pathlib import Path; from PIL import Image; p=Path('figures/fig_vision_annotated_frame.png'); m=json.loads(Path('figures/fig_vision_annotated_frame.meta.json').read_text(encoding='utf-8')); im=Image.open(p); print(im.size, m['frame'], m['active_track_count'], m['active_trace_point_count'], m['zoom'])"
```

- [ ] **Step 5: Commit only the promoted evidence pair.**

```powershell
git add figures/fig_vision_annotated_frame.png figures/fig_vision_annotated_frame.meta.json
git diff --cached --stat
git commit -m "chore: refresh audited vision evidence"
```

## Task 6: Require provenance in the data loader and render Figure 01 from one frame

**Files:**

- Modify: `scripts/visualization/data_loader.py:35-45`
- Modify: `scripts/visualization/data_loader.py:123-141`
- Modify: `scripts/visualization/fig01_vision_to_twin.py:1-48`
- Modify: `tests/test_visualization_data.py`
- Modify: `tests/test_paper_figure_suite_contract.py`

- [ ] **Step 1: Add the validated metadata field test.**

Extend `VisualizationData` with `annotated_frame_meta: dict[str, Any]` and first add this failing test:

```python
def test_formal_annotated_frame_has_validated_provenance(bundle):
    meta = bundle.annotated_frame_meta
    assert meta["selector"] == {
        "name": "active-tracks-trace-points",
        "version": "active-tracks-trace-points-v1",
    }
    assert meta["frame"]["fps"] == bundle.traffic_state["source"]["fps"]
    assert 0 <= meta["frame"]["index"] < bundle.traffic_state["source"]["frames"]
    assert meta["frame"]["timestamp_sec"] == pytest.approx(
        meta["frame"]["index"] / meta["frame"]["fps"]
    )
```

- [ ] **Step 2: Run the focused loader test and verify RED.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\loader-red `
  tests/test_visualization_data.py::test_formal_annotated_frame_has_validated_provenance
```

- [ ] **Step 3: Load the frame and metadata through the one shared validator.**

```python
from vision.frame_evidence import load_and_validate_frame_evidence

# Add to VisualizationData:
annotated_frame_meta: dict[str, Any]

annotated_frame = root / "figures/fig_vision_annotated_frame.png"
annotated_frame_meta_path = root / "figures/fig_vision_annotated_frame.meta.json"
video_relative = str(traffic_state["source"]["video"]).replace("\\", "/")
video_path = root / Path(video_relative)
roi_path = root / "data/videos/demo_roi.json"
annotated_frame_meta = load_and_validate_frame_evidence(
    annotated_frame,
    annotated_frame_meta_path,
    video_path,
    roi_path,
    traffic_state,
)

# Add this keyword beside the existing annotated_frame keyword in the return call:
annotated_frame_meta=annotated_frame_meta,
```

- [ ] **Step 4: Add same-frame crop and no-detection rendering tests.**

```python
def test_figure01_zoom_is_pixel_slice_of_the_same_frame():
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    metadata = {"zoom": {"track_id": 7, "crop_xyxy": [2, 1, 7, 5]}}

    crop = fig01_vision_to_twin.crop_same_frame(image, metadata)

    np.testing.assert_array_equal(crop, image[1:5, 2:7])
    assert not np.shares_memory(crop, image)


def test_figure01_omits_zoom_when_metadata_has_no_crop():
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    assert fig01_vision_to_twin.crop_same_frame(
        image, {"zoom": None, "no_detections": True}
    ) is None
```

Add the rendered-panel tests below; they inspect the actual inset artist, not source strings:

```python
import matplotlib.pyplot as plt
import numpy as np


def _frame_metadata(no_detections=False, zoom=None):
    return {
        "frame": {"timestamp_sec": 0.2},
        "no_detections": no_detections,
        "zoom": zoom,
    }


def test_figure01_rendered_inset_contains_exact_same_frame_slice():
    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    metadata = _frame_metadata(
        zoom={"track_id": 7, "crop_xyxy": [2, 3, 12, 10]}
    )
    figure, axis = plt.subplots()

    inset = fig01_vision_to_twin.draw_frame_evidence(axis, image, metadata)

    assert inset is not None
    np.testing.assert_array_equal(
        np.asarray(inset.images[0].get_array()), image[3:10, 2:12]
    )
    assert len(axis.child_axes) == 1
    plt.close(figure)


def test_figure01_no_detection_panel_has_no_inset_and_downgraded_title():
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    metadata = _frame_metadata(no_detections=True, zoom=None)
    figure, axis = plt.subplots()

    inset = fig01_vision_to_twin.draw_frame_evidence(axis, image, metadata)

    assert inset is None
    assert axis.child_axes == []
    assert any(text.get_text() == "计数线布设示例" for text in axis.texts)
    plt.close(figure)
```

- [ ] **Step 5: Implement the pure crop and panel renderer.**

```python
def crop_same_frame(image: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray | None:
    zoom = metadata.get("zoom")
    if zoom is None:
        return None
    x1, y1, x2, y2 = map(int, zoom["crop_xyxy"])
    return image[y1:y2, x1:x2].copy()


def draw_frame_evidence(ax, image: np.ndarray, metadata: Mapping[str, Any]):
    ax.imshow(image)
    ax.set_axis_off()
    no_detections = bool(metadata["no_detections"])
    panel_title = "计数线布设示例" if no_detections else "视频检测与跟踪示例"
    panel_label(ax, "A", panel_title)
    timestamp = float(metadata["frame"]["timestamp_sec"])
    ax.text(
        0.02,
        0.02,
        f"自动代表帧 · t = {timestamp:.2f} s",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="white",
        bbox={"facecolor": "#1F2933", "alpha": 0.78, "edgecolor": "none", "pad": 2.5},
    )
    crop = crop_same_frame(image, metadata)
    if crop is None:
        return None
    x1, y1, x2, y2 = map(int, metadata["zoom"]["crop_xyxy"])
    ax.add_patch(
        Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="#FFFFFF", linewidth=1.5,
        )
    )
    inset = ax.inset_axes([0.54, 0.04, 0.43, 0.34])
    inset.imshow(crop)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title(
        f"轨迹 ID {metadata['zoom']['track_id']} · 同帧局部", fontsize=8
    )
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("#FFFFFF")
        spine.set_linewidth(1.2)
    return inset
```

Replace the existing panel-A `imshow`/`panel_label` pair in `build()` with this exact block:

```python
image = plt.imread(data.annotated_frame)
draw_frame_evidence(ax_image, image, data.annotated_frame_meta)
(source_dir / "01_vision_frame_provenance.json").write_text(
    json.dumps(data.annotated_frame_meta, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
```

Add `import json` and `from collections.abc import Mapping` to the Figure 01 module. Do not reopen the video.

- [ ] **Step 6: Run Figure 01 and loader tests and verify GREEN.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\fig01-green `
  tests/test_frame_evidence.py `
  tests/test_visualization_data.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\fig01-contract `
  tests/test_paper_figure_suite_contract.py -k "figure01 or provenance"
```

- [ ] **Step 7: Commit only the loader, Figure 01, and tests.**

```powershell
git add scripts/visualization/data_loader.py scripts/visualization/fig01_vision_to_twin.py tests/test_visualization_data.py tests/test_paper_figure_suite_contract.py
git diff --cached --stat
git commit -m "feat: render validated figure 01 evidence"
```

## Task 7: Fix Figure 10 with a rendered-geometry contract

**Files:**

- Modify: `scripts/visualization/fig10_scenario_timeline_atlas.py:36-68`
- Create: `tests/test_figure_geometry.py`

- [ ] **Step 1: Add reusable rendered-bbox helpers and capture Figure 10.**

Create `tests/test_figure_geometry.py` with these complete imports, fixture, and helpers. The monkeypatch keeps `build()` from closing the captured figure:

```python
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
import pytest

from scripts.visualization.data_loader import load_visualization_data
from scripts.visualization import (
    fig10_scenario_timeline_atlas,
    fig11_perception_composition_flow,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_visualization_data(ROOT)


def _capture_figure(module, bundle, tmp_path, monkeypatch):
    captured = {}

    def capture(fig, output_dir, stem):
        fig.canvas.draw()
        captured["figure"] = fig
        return {"svg": "", "png": "", "pdf": ""}

    monkeypatch.setattr(module, "export_figure", capture)
    module.build(bundle, tmp_path, tmp_path)
    return captured["figure"]


def _intersection_area(first, second):
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def _separation(first, second):
    dx = max(first.x0 - second.x1, second.x0 - first.x1, 0.0)
    dy = max(first.y0 - second.y1, second.y0 - first.y1, 0.0)
    return (dx * dx + dy * dy) ** 0.5
```

- [ ] **Step 2: Write the Figure 10 failure assertions.**

Add the complete Figure 10 test:

```python
def test_figure10_unit_label_has_zero_overlap_and_four_point_clearance(
    bundle, tmp_path, monkeypatch
):
    figure = _capture_figure(
        fig10_scenario_timeline_atlas, bundle, tmp_path, monkeypatch
    )
    renderer = figure.canvas.get_renderer()
    unit = next(text for text in figure.texts if text.get_text() == "队列（辆）")
    unit_bbox = unit.get_window_extent(renderer)
    scenario_texts = [text for axis in figure.axes for text in axis.texts]
    assert len(scenario_texts) == 10
    assert all(
        _intersection_area(unit_bbox, text.get_window_extent(renderer)) == 0
        for text in scenario_texts
    )
    minimum_gap_px = figure.dpi * 4.0 / 72.0
    assert _separation(
        unit_bbox, figure.axes[0].get_window_extent(renderer)
    ) >= minimum_gap_px
    assert _separation(
        unit_bbox, figure.legends[0].get_window_extent(renderer)
    ) >= minimum_gap_px
    assert figure.bbox.contains(unit_bbox.x0, unit_bbox.y0)
    assert figure.bbox.contains(unit_bbox.x1, unit_bbox.y1)
    plt.close(figure)
```

Do not use `figure.get_tightbbox()` as a containment oracle: it is derived from the artists themselves and would make a clipping check nearly tautological. Task 9 adds a saved-output border test for the final tight-bbox export.

- [ ] **Step 3: Run only the Figure 10 geometry test and verify RED.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\fig10-red `
  tests/test_figure_geometry.py -k figure10
```

Expected: the existing vertical `supylabel` intersects at least the evening-peak label/condition.

- [ ] **Step 4: Make the one-line geometry fix.**

Delete:

```python
fig.supylabel("队列（辆）", x=0.045, fontsize=10, color=INK)
```

Add:

```python
fig.text(
    0.15,
    0.835,
    "队列（辆）",
    ha="left",
    va="bottom",
    fontsize=10,
    color=INK,
    gid="queue-unit-label",
)
```

Do not change the axes limits, ticks, curves, uncertainty bands, scenario strings, or legend.

- [ ] **Step 5: Rerun the Figure 10 test and verify GREEN.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\fig10-green `
  tests/test_figure_geometry.py -k figure10
```

Expected: the one Figure 10 test passes.

- [ ] **Step 6: Commit the geometry fix and its test file.**

```powershell
git add scripts/visualization/fig10_scenario_timeline_atlas.py tests/test_figure_geometry.py
git diff --cached --stat
git commit -m "fix: remove figure 10 label collision"
```

## Task 8: Implement Figure 11 option B and prove every label is separated

**Files:**

- Modify: `scripts/visualization/fig11_perception_composition_flow.py:61-101`
- Modify: `tests/test_figure_geometry.py`

- [ ] **Step 1: Add Figure 11 node-geometry tests before production changes.**

Add this artist lookup and full contract test to `tests/test_figure_geometry.py`:

```python
def _by_gid(axis, gid):
    return next(artist for artist in axis.get_children() if artist.get_gid() == gid)


def test_figure11_option_b_node_geometry_is_exact(
    bundle, tmp_path, monkeypatch
):
    figure = _capture_figure(
        fig11_perception_composition_flow, bundle, tmp_path, monkeypatch
    )
    axis = figure.axes[0]
    renderer = figure.canvas.get_renderer()
    axis_bbox = axis.get_window_extent(renderer)
    expected = {
        "direction-north": (0.005, 0.25, 0.10),
        "direction-south": (0.005, 0.25, 0.10),
        "direction-east": (0.005, 0.25, 0.10),
        "direction-west": (0.005, 0.25, 0.10),
        "class-car": (0.76, 0.22, 0.10),
        "class-bus": (0.76, 0.22, 0.10),
        "class-truck": (0.76, 0.22, 0.10),
        "class-motorcycle": (0.76, 0.22, 0.10),
    }
    rectangles = []
    names = []
    values = []
    minimum_gap_px = figure.dpi * 2.0 / 72.0
    for key, (expected_x, expected_width, expected_height) in expected.items():
        rectangle = _by_gid(axis, f"{key}-box")
        name = _by_gid(axis, f"{key}-name")
        value = _by_gid(axis, f"{key}-value")
        rectangles.append(rectangle)
        names.append(name)
        values.append(value)
        assert rectangle.get_x() == pytest.approx(expected_x)
        assert rectangle.get_width() == pytest.approx(expected_width)
        assert rectangle.get_height() == pytest.approx(expected_height)
        assert name.get_position()[0] == pytest.approx(
            expected_x + expected_width / 2
        )
        assert value.get_position()[1] == pytest.approx(
            rectangle.get_y() - 0.018
        )
        assert value.get_fontsize() == pytest.approx(8.0)
        assert value.get_verticalalignment() == "top"
        rectangle_bbox = rectangle.get_window_extent(renderer)
        name_bbox = name.get_window_extent(renderer)
        value_bbox = value.get_window_extent(renderer)
        assert name_bbox.x0 >= rectangle_bbox.x0 + minimum_gap_px
        assert name_bbox.x1 <= rectangle_bbox.x1 - minimum_gap_px
        assert name_bbox.y0 >= rectangle_bbox.y0 + minimum_gap_px
        assert name_bbox.y1 <= rectangle_bbox.y1 - minimum_gap_px
        assert rectangle_bbox.y0 - value_bbox.y1 >= minimum_gap_px
        assert _intersection_area(rectangle_bbox, value_bbox) == 0
        assert axis_bbox.contains(value_bbox.x0, value_bbox.y0)
        assert axis_bbox.contains(value_bbox.x1, value_bbox.y1)

    for index, value in enumerate(values):
        value_bbox = value.get_window_extent(renderer)
        assert all(
            _intersection_area(value_bbox, rectangle.get_window_extent(renderer)) == 0
            for rectangle in rectangles
        )
        other_texts = names + values[:index] + values[index + 1 :]
        assert all(
            _intersection_area(value_bbox, text.get_window_extent(renderer)) == 0
            for text in other_texts
        )

    flow_paths = [
        patch for patch in axis.patches if isinstance(patch, PathPatch)
    ]
    assert flow_paths
    direction_centres = {0.82, 0.61, 0.36, 0.15}
    class_centres = {0.80, 0.58, 0.36, 0.14}
    for path in flow_paths:
        vertices = path.get_path().vertices
        start_x, start_y = vertices[0]
        end_x, end_y = vertices[-1]
        assert 0.005 <= start_x <= 0.255
        assert 0.76 <= end_x <= 0.98
        assert any(start_y == pytest.approx(value) for value in direction_centres)
        assert any(end_y == pytest.approx(value) for value in class_centres)
    plt.close(figure)
```

- [ ] **Step 2: Run the Figure 11 geometry test and verify RED.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\fig11-red `
  tests/test_figure_geometry.py -k figure11
```

Expected: RED because the current nodes do not expose the required GIDs or B-layout geometry. The previously measured renderer audit already establishes that all eight current name/value pairs overlap; the final test can pass only after the artists adopt the exact B contract.

- [ ] **Step 3: Add one helper that implements option B for both node families.**

```python
def _add_node(
    ax,
    *,
    gid: str,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    value: str,
    facecolor: str,
    edgecolor: str,
    name_color: str,
    value_color: str,
    name_size: float,
):
    rectangle = Rectangle(
        (x, y - height / 2),
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        lw=1.6 if edgecolor != "none" else 0.0,
        alpha=0.90 if facecolor != "white" else 1.0,
    )
    rectangle.set_gid(f"{gid}-box")
    ax.add_patch(rectangle)
    name = ax.text(
        x + width / 2,
        y,
        label,
        ha="center",
        va="center",
        fontsize=name_size,
        color=name_color,
        weight="bold",
        gid=f"{gid}-name",
    )
    value_artist = ax.text(
        x + width / 2,
        y - height / 2 - 0.018,
        value,
        ha="center",
        va="top",
        fontsize=8,
        color=value_color,
        gid=f"{gid}-value",
    )
    return rectangle, name, value_artist
```

- [ ] **Step 4: Replace both duplicated node blocks with the helper.**

Replace the two existing node loops with these exact calls:

```python
for direction in DIRECTIONS:
    observed = bool(approaches[direction]["observed"])
    edge = OBSERVED_COLOR if observed else INFERRED_COLOR
    y = direction_y[direction]
    _add_node(
        ax_flow,
        gid=f"direction-{direction}",
        x=0.005,
        y=y,
        width=0.25,
        height=0.10,
        label=DIRECTION_LABELS[direction],
        value=f"{approaches[direction]['flow_vph']:,.1f} veh/h",
        facecolor="white",
        edgecolor=edge,
        name_color=INK,
        value_color=MUTED,
        name_size=9.8,
    )

for vehicle_class in CLASSES:
    y = class_y[vehicle_class]
    total = flow.query(
        "vehicle_class == @vehicle_class"
    )["class_flow_vph"].sum()
    _add_node(
        ax_flow,
        gid=f"class-{vehicle_class}",
        x=0.76,
        y=y,
        width=0.22,
        height=0.10,
        label=CLASS_LABELS[vehicle_class],
        value=f"{total:,.0f} veh/h",
        facecolor=CLASS_COLORS[vehicle_class],
        edgecolor="none",
        name_color="white",
        value_color=CLASS_COLORS[vehicle_class],
        name_size=9.3,
    )
```

Keep the existing flow endpoints `(0.25, direction_y)` and `(0.78, class_y)`. They remain inside the new node x intervals. Keep `_curve` width calculations, class shares, source line styles, bar chart, colors, legends, and exported source CSV unchanged.

- [ ] **Step 5: Run Figure 11 geometry tests against the fixed B parameters.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\fig11-green `
  tests/test_figure_geometry.py -k figure11
```

Expected: PASS with the fixed `0.018` offset and 8 pt value font. If it fails on the configured Matplotlib/font stack, stop and report the measured bbox gap rather than silently changing the approved geometry.

- [ ] **Step 6: Run both geometry tests together and verify GREEN.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\geometry-green `
  tests/test_figure_geometry.py
```

- [ ] **Step 7: Commit Figure 11 option B.**

```powershell
git add scripts/visualization/fig11_perception_composition_flow.py tests/test_figure_geometry.py
git diff --cached --stat
git commit -m "fix: move figure 11 values outside nodes"
```

## Task 9: Re-export only Figures 01, 10, and 11 and perform visual QA

**Files:**

- Modify generated assets only: `outputs/figures/01_vision_to_twin.{svg,pdf,png}`
- Modify generated assets only: `outputs/figures/10_scenario_timeline_atlas.{svg,pdf,png}`
- Modify generated assets only: `outputs/figures/11_perception_composition_flow.{svg,pdf,png}`
- Modify generated source only: `outputs/source_data/01_vision_frame_provenance.json`
- Must remain byte-identical: `outputs/source_data/01_flow_profile.csv`
- Must remain byte-identical: `outputs/source_data/10_aligned_queue_trajectories_30s.csv`
- Must remain byte-identical: `outputs/source_data/11_direction_vehicle_class_flow.csv`
- Modify: `tests/test_visualization_exports.py`
- Modify if present: `outputs/qa/FIGURE_QA.md`

- [ ] **Step 1: Rebuild only the three scoped figures into an isolated candidate directory.**

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from scripts.visualization.data_loader import load_visualization_data; from scripts.visualization.fig01_vision_to_twin import build as b1; from scripts.visualization.fig10_scenario_timeline_atlas import build as b10; from scripts.visualization.fig11_perception_composition_flow import build as b11; root=Path('.').resolve(); data=load_visualization_data(root); out=root/'outputs/revised_candidate/figures'; src=root/'outputs/revised_candidate/source_data'; [builder(data,out,src) for builder in (b1,b10,b11)]"
```

Do not run the all-eleven-figure entry point in this task; that would needlessly rewrite out-of-scope quantitative figures.

- [ ] **Step 2: Prove the three regenerated scientific source CSVs are byte-identical before promoting figures.**

```powershell
$sourceNames = @(
  '01_flow_profile.csv',
  '10_aligned_queue_trajectories_30s.csv',
  '11_direction_vehicle_class_flow.csv'
)
foreach ($name in $sourceNames) {
  $formal = Join-Path 'outputs\source_data' $name
  $candidate = Join-Path 'outputs\revised_candidate\source_data' $name
  if (-not (Test-Path -LiteralPath $formal)) { throw "Missing formal source: $formal" }
  if (-not (Test-Path -LiteralPath $candidate)) { throw "Missing candidate source: $candidate" }
  $formalHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $formal).Hash
  $candidateHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash
  if ($formalHash -ne $candidateHash) { throw "Scientific source changed: $name" }
}
```

Expected: the command returns without output. Any mismatch blocks promotion.

- [ ] **Step 3: Promote only the nine figure files and the new provenance JSON.**

```powershell
$stems = @('01_vision_to_twin', '10_scenario_timeline_atlas', '11_perception_composition_flow')
foreach ($stem in $stems) {
  foreach ($suffix in @('svg', 'pdf', 'png')) {
    Copy-Item -Force -LiteralPath "outputs\revised_candidate\figures\$stem.$suffix" -Destination "outputs\figures\$stem.$suffix"
  }
}
Copy-Item -Force -LiteralPath 'outputs\revised_candidate\source_data\01_vision_frame_provenance.json' -Destination 'outputs\source_data\01_vision_frame_provenance.json'
```

Do not copy the three candidate CSVs; the formal copies remain untouched.

- [ ] **Step 4: Add a saved-output clipping regression.**

Append this test to `tests/test_visualization_exports.py`:

```python
import numpy as np


def test_revised_pngs_have_white_outer_border_without_clipped_artists():
    for stem in (
        "01_vision_to_twin",
        "10_scenario_timeline_atlas",
        "11_perception_composition_flow",
    ):
        with Image.open(FIGURE_DIR / f"{stem}.png") as image:
            pixels = np.asarray(image.convert("RGB"))
        border = np.concatenate(
            (
                pixels[:2, :, :].reshape(-1, 3),
                pixels[-2:, :, :].reshape(-1, 3),
                pixels[:, :2, :].reshape(-1, 3),
                pixels[:, -2:, :].reshape(-1, 3),
            ),
            axis=0,
        )
        assert int(border.min()) >= 250, stem
```

- [ ] **Step 5: Run export contract tests.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\exports `
  tests/test_visualization_exports.py `
  tests/test_paper_figure_suite_contract.py `
  tests/test_figure_geometry.py
```

Confirm SVG text remains editable, PDFs open, PNGs are 600 DPI and at least 3000 px wide, and all three have white corners.

- [ ] **Step 6: Render the three PDFs or inspect their 600-DPI PNGs at final 183 mm size.**

During execution, invoke the required visual inspection on these exact files at original detail:

```text
view_image C:\Users\y\FlowMind\outputs\figures\01_vision_to_twin.png detail=original
view_image C:\Users\y\FlowMind\outputs\figures\10_scenario_timeline_atlas.png detail=original
view_image C:\Users\y\FlowMind\outputs\figures\11_perception_composition_flow.png detail=original
```

Use the PDF skill/render workflow during implementation because this is visual QA. Inspect:

- Figure 01: detection boxes, trace, both count lines, timestamp, full-frame crop marker, and inset are readable; the inset is the marked same-frame region.
- Figure 10: the horizontal `队列（辆）` label has clear space from the first axes, scenario descriptions, and legend.
- Figure 11: all eight names are centered inside their boxes; all eight values are directly below the correct box; no value collides with a neighboring node, curve, or chart boundary.

- [ ] **Step 7: Update QA truthfully.**

Use `apply_patch` to make these exact factual edits in `outputs/qa/FIGURE_QA.md`:

1. Replace `Final visual contact sheets inspected after the last regeneration; no clipping, missing glyphs, overlapping legends or inverted semantic colors were observed.` with `The earlier contact-sheet statement was superseded on 2026-08-19 after rendered-bbox tests found text collisions in Figures 10 and 11; those two collisions are covered by the revision checks below.`
2. In `Image-integrity disclosure for Figure 01`, replace `crop: none in the formal panel` with `crop: one deterministic inset from the same saved representative-frame PNG; crop_xyxy and track_id are recorded in outputs/source_data/01_vision_frame_provenance.json.`
3. Replace `stitching: none` with `stitching: none; the full frame and inset are from one frame index and one PNG.`
4. Append this exact section:

```markdown
## Figure 01/10/11 evidence revision (2026-08-19)

- Figure 01 representative-frame selector: `active-tracks-trace-points-v1`; the lexicographic score is `(active_track_count, active_trace_point_count, -frame_index)`.
- The authoritative selected frame index, timestamp, FPS, dimensions, active track count, visible trace-point count, video SHA-256, ROI SHA-256, annotated PNG SHA-256 and same-frame inset crop are stored without duplication in `outputs/source_data/01_vision_frame_provenance.json`.
- The isolated rerun passed exact TrafficState semantic comparison. `data/traffic_states/demo_001.json`, `data/results/arena_summary.csv` and `data/results/experiments/**` matched the pre-rerun protected snapshot after promotion.
- Figure 10 passed the rendered zero-intersection and 4 pt clearance contract for the horizontal queue-unit label.
- Figure 11 uses option B: all eight names remain inside their node boxes and all eight `veh/h` values sit below their corresponding boxes; rendered tests enforce 2 pt clearance and zero text/node intersection.
- The three regenerated scientific source CSVs are byte-identical to their formal copies. Only the Figure 01 provenance JSON is new source material.
- This revision demonstrates traceable pipeline output, not perception accuracy. No hand-labeled mAP, tracking-ID, or counting-error evaluation is available.
- Figures 02–09 are not scientifically validated by this visual-only revision. The mixed-network arena comparison and endpoint-censoring issues require a separate experiment rerun and metric repair.
```

- [ ] **Step 8: Commit the saved-output regression; leave the currently untracked output package unstaged.**

```powershell
git add tests/test_visualization_exports.py
git diff --cached --stat
git commit -m "test: verify revised figure export boundaries"
git status --short outputs
```

Expected: only the test is committed. `outputs/**` remains a local deliverable because the current repository does not track that tree; never use `git add -f`.

## Task 10: Final regression, invariance audit, and handoff

**Files:**

- Modify only if a factual mismatch is found: `docs/superpowers/specs/2026-08-19-paper-figure-01-10-11-revision-design.md`
- Modify only if a completed checklist is desired: this plan file

- [ ] **Step 1: Run the complete scoped regression suite.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .codex-test-tmp\figure-revision-final `
  tests/test_frame_evidence.py `
  tests/test_vision_analyze_evidence.py `
  tests/test_vision_evidence_gate.py `
  tests/test_visualization_data.py `
  tests/test_paper_figure_suite_contract.py `
  tests/test_visualization_exports.py `
  tests/test_figure_geometry.py `
  tests/test_counter.py `
  tests/test_traffic_state.py
```

- [ ] **Step 2: Re-run the protected snapshot verification.**

```powershell
.\.venv\Scripts\python.exe -m scripts.promote_vision_evidence verify `
  --root . `
  --snapshot outputs\vision_rerun_candidate\protected_before.json
```

The CLI implementation in Task 4 must include this `verify` subcommand as a thin call to `verify_protected_data`; add its CLI test before using it here.

- [ ] **Step 3: Inspect the final diff without disturbing unrelated work.**

```powershell
git status --short
git diff --check -- vision/frame_evidence.py vision/analyze.py scripts/promote_vision_evidence.py scripts/visualization/data_loader.py scripts/visualization/fig01_vision_to_twin.py scripts/visualization/fig10_scenario_timeline_atlas.py scripts/visualization/fig11_perception_composition_flow.py tests/test_frame_evidence.py tests/test_vision_analyze_evidence.py tests/test_vision_evidence_gate.py tests/test_visualization_data.py tests/test_paper_figure_suite_contract.py tests/test_figure_geometry.py
git diff --stat
git log --oneline -10
```

Verify no commit includes `data/traffic_states/demo_001.json`, `data/results/arena_summary.csv`, any experiment artifact, model weight, candidate rerun directory, or Figure 02–09 output.

- [ ] **Step 4: Request a code review using `superpowers:requesting-code-review`.**

The reviewer must check the approved score and option B geometry against the design, inspect all new provenance failure paths, and confirm tests assert rendered geometry rather than source-code strings.

```text
Review range: a247aca..HEAD
Required checks: selector score; one trace update/frame; evidence schema; rollback; protected hashes; Figure 01 same-frame crop; Figure 10 rendered gaps; Figure 11 B geometry; source CSV byte identity.
```

- [ ] **Step 5: Report the result with precise boundaries.**

State which visual assets changed, the selected frame metadata, the protected-data hash result, test count, and preview paths. Explicitly state that this revision improves evidence traceability and layout only; it does not change experimental numbers or resolve the separate scientific validity blockers in Figures 02–09.

Gather the exact dynamic values for that report with:

```powershell
.\.venv\Scripts\python.exe -c "import json; from pathlib import Path; m=json.loads(Path('figures/fig_vision_annotated_frame.meta.json').read_text(encoding='utf-8')); print(json.dumps({'selector':m['selector'],'frame':m['frame'],'active_track_count':m['active_track_count'],'active_trace_point_count':m['active_trace_point_count'],'zoom':m['zoom']},ensure_ascii=False,indent=2))"
.\.venv\Scripts\python.exe -m scripts.promote_vision_evidence verify --root . --snapshot outputs\vision_rerun_candidate\protected_before.json
```
