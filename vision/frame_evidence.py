from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import supervision as sv
from PIL import Image


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
        self.fps = float(fps)
        if self.fps <= 0:
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
            timestamp_sec=float(frame_index) / self.fps,
            fps=self.fps,
            tracks=tuple(tracks),
        )
        if self._best is None or candidate.score > self._best.score:
            self._best = candidate

    def result(self) -> RepresentativeFrame:
        if self._best is None:
            raise ValueError("video contained no readable frames")
        return self._best


def snapshot_active_tracks(
    tracked: sv.Detections,
    trace_annotator: sv.TraceAnnotator,
    *,
    class_name: Callable[[object], str],
) -> tuple[TrackEvidence, ...]:
    if tracked.tracker_id is None:
        return ()

    snapshots: list[TrackEvidence] = []
    for index, bbox_xyxy in enumerate(tracked.xyxy):
        track_id = int(tracked.tracker_id[index])
        if track_id < 0:
            continue
        class_id = None if tracked.class_id is None else tracked.class_id[index]
        trace = np.asarray(
            trace_annotator.trace.get(tracker_id=track_id), dtype=float
        ).reshape(-1, 2)[-TRACE_LENGTH:]
        snapshots.append(
            TrackEvidence(
                track_id=track_id,
                class_name=class_name(class_id),
                bbox_xyxy=tuple(float(value) for value in bbox_xyxy),
                trace_points=tuple(
                    (float(point[0]), float(point[1])) for point in trace
                ),
            )
        )

    return tuple(sorted(snapshots, key=lambda track: track.track_id))


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

    selected = max(
        representative.tracks,
        key=lambda track: (
            len(track.trace_points),
            _bbox_area(track),
            -track.track_id,
        ),
    )
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = selected.bbox_xyxy
    x_coordinates = [bbox_x1, bbox_x2]
    y_coordinates = [bbox_y1, bbox_y2]
    for trace_x, trace_y in selected.trace_points:
        x_coordinates.append(trace_x)
        y_coordinates.append(trace_y)

    envelope_x1 = min(x_coordinates)
    envelope_y1 = min(y_coordinates)
    envelope_x2 = max(x_coordinates)
    envelope_y2 = max(y_coordinates)
    width = max(1.0, envelope_x2 - envelope_x1)
    height = max(1.0, envelope_y2 - envelope_y1)
    left = math.floor(envelope_x1 - width * margin_ratio)
    top = math.floor(envelope_y1 - height * margin_ratio)
    right = math.ceil(envelope_x2 + width * margin_ratio)
    bottom = math.ceil(envelope_y2 + height * margin_ratio)

    image_height, image_width = representative.annotated_bgr.shape[:2]
    left = max(0, min(image_width, left))
    top = max(0, min(image_height, top))
    right = max(0, min(image_width, right))
    bottom = max(0, min(image_height, bottom))
    if right <= left or bottom <= top:
        return None

    return {
        "track_id": selected.track_id,
        "crop_xyxy": [left, top, right, bottom],
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

    image_height, image_width = representative.annotated_bgr.shape[:2]
    metadata = {
        "selector": {
            "name": SELECTOR_NAME,
            "version": SELECTOR_VERSION,
        },
        "video": {
            "path": video_path.as_posix(),
            "sha256": sha256_file(video_path),
        },
        "roi": {
            "sha256": canonical_json_sha256(roi_config),
        },
        "frame": {
            "index": representative.frame_index,
            "timestamp_sec": representative.timestamp_sec,
            "fps": representative.fps,
            "width": image_width,
            "height": image_height,
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
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return frame_path, meta_path


def _evidence_error(field: str, detail: str) -> ValueError:
    return ValueError(f"{field}: {detail}")


def _normalise_path(value: object) -> str:
    return str(value).replace("\\", "/")


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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

    expected_selector = {
        "name": SELECTOR_NAME,
        "version": SELECTOR_VERSION,
    }
    if not isinstance(metadata, Mapping) or metadata.get("selector") != expected_selector:
        raise _evidence_error("selector", "does not match the required selector")

    frame_metadata = metadata.get("frame")
    stored_frame_digest = (
        frame_metadata.get("annotated_sha256")
        if isinstance(frame_metadata, Mapping)
        else None
    )
    if stored_frame_digest != sha256_file(frame_path):
        raise _evidence_error(
            "frame.annotated_sha256", "does not match the PNG contents"
        )

    video_metadata = metadata.get("video")
    stored_video_digest = (
        video_metadata.get("sha256")
        if isinstance(video_metadata, Mapping)
        else None
    )
    if stored_video_digest != sha256_file(video_path):
        raise _evidence_error("video.sha256", "does not match the video contents")

    roi_metadata = metadata.get("roi")
    stored_roi_digest = (
        roi_metadata.get("sha256") if isinstance(roi_metadata, Mapping) else None
    )
    roi_config = json.loads(roi_path.read_text(encoding="utf-8"))
    if stored_roi_digest != canonical_json_sha256(roi_config):
        raise _evidence_error("roi.sha256", "does not match the ROI configuration")

    source = (
        traffic_state.get("source")
        if isinstance(traffic_state, Mapping)
        else None
    )
    if not isinstance(source, Mapping):
        raise _evidence_error("traffic_state.source", "must be a mapping")

    metadata_video_path = (
        video_metadata.get("path") if isinstance(video_metadata, Mapping) else None
    )
    source_video_path = source.get("video")
    if (
        not isinstance(metadata_video_path, str)
        or source_video_path is None
        or _normalise_path(metadata_video_path)
        != _normalise_path(source_video_path)
    ):
        raise _evidence_error("video.path", "does not match traffic_state.source.video")

    if not isinstance(frame_metadata, Mapping):
        raise _evidence_error("frame", "must be a mapping")
    with Image.open(frame_path) as image:
        if image.format != "PNG":
            raise _evidence_error("frame.format", "must be PNG")
        image_width, image_height = image.size

    frame_width = frame_metadata.get("width")
    if not _strict_int(frame_width) or frame_width != image_width:
        raise _evidence_error("frame.width", "does not match the PNG width")
    frame_height = frame_metadata.get("height")
    if not _strict_int(frame_height) or frame_height != image_height:
        raise _evidence_error("frame.height", "does not match the PNG height")

    frame_fps = frame_metadata.get("fps")
    source_fps = source.get("fps")
    if (
        not _finite_number(frame_fps)
        or float(frame_fps) <= 0.0
        or not _finite_number(source_fps)
        or not math.isclose(
            float(frame_fps),
            float(source_fps),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise _evidence_error(
            "frame.fps", "must be positive, finite, and match the source fps"
        )

    frame_index = frame_metadata.get("index")
    source_frames = source.get("frames")
    if (
        not _strict_int(frame_index)
        or not _strict_int(source_frames)
        or not 0 <= frame_index < source_frames
    ):
        raise _evidence_error("frame.index", "must be within the source frame range")

    timestamp_sec = frame_metadata.get("timestamp_sec")
    expected_timestamp = frame_index / float(frame_fps)
    if not _finite_number(timestamp_sec) or not math.isclose(
        float(timestamp_sec),
        expected_timestamp,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise _evidence_error(
            "frame.timestamp_sec", "must equal frame.index divided by frame.fps"
        )

    tracks = metadata.get("tracks")
    if not isinstance(tracks, list):
        raise _evidence_error("tracks", "must be a list")

    active_track_ids: set[int] = set()
    trace_point_count = 0
    for track in tracks:
        if not isinstance(track, Mapping):
            raise _evidence_error("tracks", "each track must be a mapping")

        track_id = track.get("track_id")
        if (
            not _strict_int(track_id)
            or track_id < 0
            or track_id in active_track_ids
        ):
            raise _evidence_error(
                "tracks.track_id", "must be a unique non-negative integer"
            )
        active_track_ids.add(track_id)

        if not isinstance(track.get("class_name"), str):
            raise _evidence_error("tracks.class_name", "must be a string")

        bbox_xyxy = track.get("bbox_xyxy")
        if (
            not isinstance(bbox_xyxy, list)
            or len(bbox_xyxy) != 4
            or not all(_finite_number(value) for value in bbox_xyxy)
        ):
            raise _evidence_error(
                "tracks.bbox_xyxy", "must contain four finite numbers"
            )
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
            float(value) for value in bbox_xyxy
        )
        if not (
            0.0 <= bbox_x1 < bbox_x2 <= image_width
            and 0.0 <= bbox_y1 < bbox_y2 <= image_height
        ):
            raise _evidence_error(
                "tracks.bbox_xyxy", "must be within the image bounds"
            )

        trace_points = track.get("trace_points")
        if not isinstance(trace_points, list) or len(trace_points) > TRACE_LENGTH:
            raise _evidence_error(
                "tracks.trace_points", "must be a list no longer than TRACE_LENGTH"
            )
        for point in trace_points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(_finite_number(value) for value in point)
            ):
                raise _evidence_error(
                    "tracks.trace_points", "each point must contain two finite numbers"
                )
            point_x, point_y = (float(value) for value in point)
            if not (
                0.0 <= point_x <= image_width
                and 0.0 <= point_y <= image_height
            ):
                raise _evidence_error(
                    "tracks.trace_points", "each point must be within the image bounds"
                )
        trace_point_count += len(trace_points)

    active_track_count = metadata.get("active_track_count")
    if (
        not _strict_int(active_track_count)
        or active_track_count != len(tracks)
    ):
        raise _evidence_error(
            "active_track_count", "must equal the number of active tracks"
        )

    stored_trace_point_count = metadata.get("active_trace_point_count")
    if (
        not _strict_int(stored_trace_point_count)
        or stored_trace_point_count != trace_point_count
    ):
        raise _evidence_error(
            "active_trace_point_count",
            "must equal the number of active trace points",
        )

    no_detections = metadata.get("no_detections")
    if not isinstance(no_detections, bool) or no_detections != (not tracks):
        raise _evidence_error(
            "no_detections", "must state whether the track list is empty"
        )

    zoom = metadata.get("zoom")
    if zoom is not None:
        if not isinstance(zoom, Mapping):
            raise _evidence_error("zoom", "must be a mapping or null")
        zoom_track_id = zoom.get("track_id")
        if (
            not _strict_int(zoom_track_id)
            or zoom_track_id not in active_track_ids
        ):
            raise _evidence_error("zoom.track_id", "must identify an active track")

        crop_xyxy = zoom.get("crop_xyxy")
        if (
            not isinstance(crop_xyxy, list)
            or len(crop_xyxy) != 4
            or not all(_strict_int(value) for value in crop_xyxy)
        ):
            raise _evidence_error(
                "zoom.crop_xyxy", "must contain four integer coordinates"
            )
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
        if not (
            0 <= crop_x1 < crop_x2 <= image_width
            and 0 <= crop_y1 < crop_y2 <= image_height
        ):
            raise _evidence_error(
                "zoom.crop_xyxy", "must be a valid half-open image crop"
            )

    return dict(metadata)
