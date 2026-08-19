import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import supervision as sv

from vision.frame_evidence import (
    RepresentativeFrame,
    RepresentativeFrameSelector,
    TrackEvidence,
    load_and_validate_frame_evidence,
    select_zoom,
    sha256_file,
    snapshot_active_tracks,
    write_frame_evidence,
)


def _image(value: int) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def _track(track_id: int, trace_length: int) -> TrackEvidence:
    return TrackEvidence(
        track_id=track_id,
        class_name="car",
        bbox_xyxy=(5.0, 5.0, 15.0, 15.0),
        trace_points=tuple((float(i), float(i)) for i in range(trace_length)),
    )


def _representative(
    *tracks: TrackEvidence,
    image_shape: tuple[int, int, int] = (50, 80, 3),
) -> RepresentativeFrame:
    return RepresentativeFrame(
        annotated_bgr=np.zeros(image_shape, dtype=np.uint8),
        frame_index=0,
        timestamp_sec=0.0,
        fps=10.0,
        tracks=tracks,
    )


def _write_valid_evidence(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"deterministic-video-payload")
    roi_config = {"count_lines": {"north": [[0, 5], [29, 5]]}}
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps(roi_config), encoding="utf-8")
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
    meta_path = tmp_path / "meta.json"
    write_frame_evidence(
        representative,
        frame_path=frame_path,
        meta_path=meta_path,
        video_path=video_path,
        roi_config=roi_config,
    )
    traffic_state = {
        "source": {
            "video": video_path.as_posix(),
            "fps": 10.0,
            "frames": 3,
            "duration": 0.3,
        }
    }
    return frame_path, meta_path, video_path, roi_path, traffic_state


def _read_metadata(meta_path: Path) -> dict[str, Any]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _write_metadata(meta_path: Path, metadata: dict[str, Any]) -> None:
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _set_nested(metadata: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = metadata
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_snapshot_active_tracks_uses_real_trace_annotator_centres() -> None:
    tracked = sv.Detections(
        xyxy=np.array([[4.0, 5.0, 14.0, 15.0]]),
        class_id=np.array([0]),
        tracker_id=np.array([7]),
    )
    trace_annotator = sv.TraceAnnotator(trace_length=60)
    trace_annotator.annotate(scene=np.zeros((20, 20, 3)), detections=tracked)

    result = snapshot_active_tracks(
        tracked,
        trace_annotator,
        class_name=lambda class_id: "car",
    )

    assert result == (
        TrackEvidence(
            track_id=7,
            class_name="car",
            bbox_xyxy=(4.0, 5.0, 14.0, 15.0),
            trace_points=((9.0, 10.0),),
        ),
    )


def test_snapshot_active_tracks_drops_pending_ids_and_sorts_numerically() -> None:
    tracked = sv.Detections(
        xyxy=np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [10.0, 10.0, 20.0, 20.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
        class_id=np.array([2, 5, 7]),
        tracker_id=np.array([8, -1, 3]),
    )
    trace_annotator = sv.TraceAnnotator(trace_length=60)
    trace_annotator.annotate(scene=np.zeros((40, 40, 3)), detections=tracked)

    result = snapshot_active_tracks(
        tracked,
        trace_annotator,
        class_name=lambda class_id: "truck" if class_id == 7 else "car",
    )

    assert result == (
        TrackEvidence(
            track_id=3,
            class_name="truck",
            bbox_xyxy=(20.0, 20.0, 30.0, 30.0),
            trace_points=((25.0, 25.0),),
        ),
        TrackEvidence(
            track_id=8,
            class_name="car",
            bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
            trace_points=((5.0, 5.0),),
        ),
    )


def test_snapshot_active_tracks_without_tracker_ids_is_empty() -> None:
    tracked = sv.Detections(xyxy=np.array([[4.0, 5.0, 14.0, 15.0]]))

    result = snapshot_active_tracks(
        tracked,
        sv.TraceAnnotator(trace_length=60),
        class_name=lambda class_id: "car",
    )

    assert result == ()


def test_select_zoom_prefers_larger_bbox_when_trace_lengths_tie() -> None:
    representative = _representative(
        TrackEvidence(
            track_id=1,
            class_name="car",
            bbox_xyxy=(10.0, 10.0, 20.0, 20.0),
            trace_points=((12.0, 12.0), (18.0, 18.0)),
        ),
        TrackEvidence(
            track_id=2,
            class_name="truck",
            bbox_xyxy=(30.0, 10.0, 50.0, 30.0),
            trace_points=((35.0, 15.0), (45.0, 25.0)),
        ),
    )

    assert select_zoom(representative) == {
        "track_id": 2,
        "crop_xyxy": [27, 7, 53, 33],
    }


def test_select_zoom_prefers_smaller_track_id_on_full_tie() -> None:
    representative = _representative(
        TrackEvidence(
            track_id=9,
            class_name="car",
            bbox_xyxy=(10.0, 10.0, 20.0, 20.0),
            trace_points=((12.0, 12.0),),
        ),
        TrackEvidence(
            track_id=2,
            class_name="car",
            bbox_xyxy=(30.0, 10.0, 40.0, 20.0),
            trace_points=((32.0, 12.0),),
        ),
    )

    assert select_zoom(representative)["track_id"] == 2


def test_select_zoom_clamps_expanded_envelope_to_image_bounds() -> None:
    representative = _representative(
        TrackEvidence(
            track_id=1,
            class_name="car",
            bbox_xyxy=(0.5, 1.5, 79.5, 49.5),
            trace_points=((0.0, 0.0), (79.0, 49.0)),
        ),
        image_shape=(50, 80, 3),
    )

    result = select_zoom(representative)

    assert result == {"track_id": 1, "crop_xyxy": [0, 0, 80, 50]}
    assert all(isinstance(value, int) for value in result["crop_xyxy"])


def test_select_zoom_without_detections_is_none() -> None:
    assert select_zoom(_representative()) is None


def test_frame_evidence_round_trip_has_expected_counts_zoom_and_digests(
    tmp_path: Path,
) -> None:
    frame_path, meta_path, video_path, roi_path, traffic_state = (
        _write_valid_evidence(tmp_path)
    )

    metadata = load_and_validate_frame_evidence(
        frame_path,
        meta_path,
        video_path,
        roi_path,
        traffic_state,
    )

    assert metadata["frame"]["index"] == 2
    assert metadata["active_track_count"] == 1
    assert metadata["active_trace_point_count"] == 2
    assert metadata["zoom"]["track_id"] == 7
    for digest in (
        metadata["frame"]["annotated_sha256"],
        metadata["video"]["sha256"],
        metadata["roi"]["sha256"],
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


@pytest.mark.parametrize(
    ("path", "value", "expected_field"),
    [
        (("frame", "width"), 31, "frame.width"),
        (("frame", "timestamp_sec"), 0.25, "frame.timestamp_sec"),
        (("frame", "fps"), 11, "frame.fps"),
        (("frame", "index"), 3, "frame.index"),
        (("active_track_count",), 2, "active_track_count"),
        (("active_trace_point_count",), 99, "active_trace_point_count"),
        (("no_detections",), True, "no_detections"),
        (("zoom", "track_id"), 99, "zoom.track_id"),
        (("zoom", "crop_xyxy"), [0, 0, 31, 20], "zoom.crop_xyxy"),
    ],
)
def test_frame_evidence_rejects_tampered_contract_fields(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
    expected_field: str,
) -> None:
    frame_path, meta_path, video_path, roi_path, traffic_state = (
        _write_valid_evidence(tmp_path)
    )
    metadata = _read_metadata(meta_path)
    _set_nested(metadata, path, value)
    _write_metadata(meta_path, metadata)

    with pytest.raises(ValueError, match=rf"^{re.escape(expected_field)}:"):
        load_and_validate_frame_evidence(
            frame_path,
            meta_path,
            video_path,
            roi_path,
            traffic_state,
        )


def test_frame_evidence_rejects_non_deterministic_active_zoom_track(
    tmp_path: Path,
) -> None:
    frame_path, meta_path, video_path, roi_path, traffic_state = (
        _write_valid_evidence(tmp_path)
    )
    metadata = _read_metadata(meta_path)
    metadata["tracks"].append(
        {
            "track_id": 9,
            "class_name": "truck",
            "bbox_xyxy": [20.0, 2.0, 25.0, 7.0],
            "trace_points": [[22.0, 4.0]],
        }
    )
    metadata["active_track_count"] = 2
    metadata["active_trace_point_count"] = 3
    metadata["zoom"] = {
        "track_id": 9,
        "crop_xyxy": [19, 1, 26, 8],
    }
    _write_metadata(meta_path, metadata)

    with pytest.raises(ValueError, match=r"^zoom\.track_id:"):
        load_and_validate_frame_evidence(
            frame_path,
            meta_path,
            video_path,
            roi_path,
            traffic_state,
        )


def test_frame_evidence_rejects_non_deterministic_zoom_crop(
    tmp_path: Path,
) -> None:
    frame_path, meta_path, video_path, roi_path, traffic_state = (
        _write_valid_evidence(tmp_path)
    )
    metadata = _read_metadata(meta_path)
    metadata["zoom"]["crop_xyxy"] = [4, 4, 17, 18]
    _write_metadata(meta_path, metadata)

    with pytest.raises(ValueError, match=r"^zoom\.crop_xyxy:"):
        load_and_validate_frame_evidence(
            frame_path,
            meta_path,
            video_path,
            roi_path,
            traffic_state,
        )


def test_frame_evidence_rejects_missing_zoom_for_active_tracks(
    tmp_path: Path,
) -> None:
    frame_path, meta_path, video_path, roi_path, traffic_state = (
        _write_valid_evidence(tmp_path)
    )
    metadata = _read_metadata(meta_path)
    metadata["zoom"] = None
    _write_metadata(meta_path, metadata)

    with pytest.raises(ValueError, match=r"^zoom:"):
        load_and_validate_frame_evidence(
            frame_path,
            meta_path,
            video_path,
            roi_path,
            traffic_state,
        )


def test_frame_evidence_rejects_duplicate_track_ids(tmp_path: Path) -> None:
    frame_path, meta_path, video_path, roi_path, traffic_state = (
        _write_valid_evidence(tmp_path)
    )
    metadata = _read_metadata(meta_path)
    metadata["tracks"].append(copy.deepcopy(metadata["tracks"][0]))
    metadata["active_track_count"] = 2
    metadata["active_trace_point_count"] = 4
    _write_metadata(meta_path, metadata)

    with pytest.raises(ValueError, match=r"^tracks\.track_id:"):
        load_and_validate_frame_evidence(
            frame_path,
            meta_path,
            video_path,
            roi_path,
            traffic_state,
        )


def test_frame_evidence_rejects_tampered_png_bytes(tmp_path: Path) -> None:
    frame_path, meta_path, video_path, roi_path, traffic_state = (
        _write_valid_evidence(tmp_path)
    )
    with frame_path.open("ab") as frame_file:
        frame_file.write(b"tamper")

    with pytest.raises(ValueError, match=r"^frame\.annotated_sha256:"):
        load_and_validate_frame_evidence(
            frame_path,
            meta_path,
            video_path,
            roi_path,
            traffic_state,
        )


def test_frame_evidence_rejects_tampered_video_bytes(tmp_path: Path) -> None:
    frame_path, meta_path, video_path, roi_path, traffic_state = (
        _write_valid_evidence(tmp_path)
    )
    video_path.write_bytes(b"changed-video-payload")

    with pytest.raises(ValueError, match=r"^video\.sha256:"):
        load_and_validate_frame_evidence(
            frame_path,
            meta_path,
            video_path,
            roi_path,
            traffic_state,
        )


def test_frame_evidence_rejects_tampered_roi_config(tmp_path: Path) -> None:
    frame_path, meta_path, video_path, roi_path, traffic_state = (
        _write_valid_evidence(tmp_path)
    )
    roi_path.write_text(
        json.dumps({"count_lines": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"^roi\.sha256:"):
        load_and_validate_frame_evidence(
            frame_path,
            meta_path,
            video_path,
            roi_path,
            traffic_state,
        )


def test_active_track_count_takes_priority_over_trace_point_count() -> None:
    selector = RepresentativeFrameSelector(fps=30.0)
    selector.consider(
        annotated_bgr=_image(0),
        frame_index=0,
        tracks=(_track(1, 60),),
    )
    selector.consider(
        annotated_bgr=_image(1),
        frame_index=1,
        tracks=(_track(1, 1), _track(2, 1)),
    )

    result = selector.result()

    assert result.frame_index == 1
    assert result.active_track_count == 2
    assert result.active_trace_point_count == 2


def test_trace_point_count_breaks_ties_between_equal_track_counts() -> None:
    selector = RepresentativeFrameSelector(fps=25.0)
    selector.consider(
        annotated_bgr=_image(0),
        frame_index=0,
        tracks=(_track(1, 2), _track(2, 2)),
    )
    selector.consider(
        annotated_bgr=_image(1),
        frame_index=1,
        tracks=(_track(1, 3), _track(2, 2)),
    )

    result = selector.result()

    assert result.frame_index == 1
    assert result.active_track_count == 2
    assert result.active_trace_point_count == 5


def test_earlier_frame_breaks_full_tie_and_owns_an_image_copy() -> None:
    selector = RepresentativeFrameSelector(fps=20.0)
    selector.consider(
        annotated_bgr=_image(9),
        frame_index=4,
        tracks=(_track(1, 2), _track(2, 1)),
    )
    earlier_image = _image(3)
    selector.consider(
        annotated_bgr=earlier_image,
        frame_index=2,
        tracks=(_track(1, 2), _track(2, 1)),
    )
    earlier_image[:] = 99

    result = selector.result()

    assert result.frame_index == 2
    assert result.timestamp_sec == 0.1
    assert result.score == (2, 3, -2)
    np.testing.assert_array_equal(result.annotated_bgr, _image(3))


def test_no_detections_returns_first_fully_annotated_frame() -> None:
    selector = RepresentativeFrameSelector(fps=10.0)
    first_image = _image(7)
    selector.consider(annotated_bgr=first_image, frame_index=0, tracks=())
    selector.consider(annotated_bgr=_image(8), frame_index=1, tracks=())

    result = selector.result()

    assert result.frame_index == 0
    assert result.no_detections is True
    np.testing.assert_array_equal(result.annotated_bgr, first_image)
