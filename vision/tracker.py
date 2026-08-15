"""ByteTrack wrapper for multi-object tracking across video frames.

Usage:
    tracker = VehicleTracker(fps=25.0)
    for frame in video:
        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        # tracked.tracker_id is now populated with stable IDs
"""
from __future__ import annotations

import supervision as sv


class VehicleTracker:
    """Thin wrapper around supervision.ByteTrack for vehicle tracking."""

    def __init__(
        self,
        fps: float = 30.0,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ) -> None:
        """
        Args:
            fps: video frame rate (used for lost_track_buffer calculation).
            track_activation_threshold: detection confidence threshold for track
                activation. Higher = more stable, fewer false tracks.
            lost_track_buffer: frames to buffer when a track is lost. Higher =
                better occlusion handling, fewer ID switches.
            minimum_matching_threshold: IOU threshold for matching tracks with
                detections. Lower = better accuracy but more ID switches; higher
                = better completeness but more drift.
        """
        self.fps = float(fps)
        self.tracker = sv.ByteTrack(
            frame_rate=int(round(self.fps)),
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with new frame's detections, returning tracked dets."""
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Reset tracker state (call when starting a new video)."""
        self.tracker.reset()
