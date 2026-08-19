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
