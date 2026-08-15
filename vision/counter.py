"""Four-direction line-crossing counter with vehicle-type breakdown.

Given an ROI configuration with up to four count lines (north/south/east/west),
tracks vehicles crossing each line and aggregates counts by direction and type.
The direction label indicates which approach the vehicle is coming FROM (i.e.,
the inbound direction), following the TrafficState schema convention.

Counting sense: `sv.LineZone` splits crossings into `in`/`out` by the side of
the line the vehicle came from, which depends on how the line was drawn. On a
two-way road both senses fire -- inbound traffic *and* the outbound traffic
leaving the intersection -- so the default `"both"` roughly doubles the measured
approach flow. Set `count_sense` in the ROI config to `"in"` or `"out"` (or a
per-direction dict) once the line orientation is known:

    {"count_lines": {...}, "count_sense": {"north": "in", "east": "both"}}

Usage:
    counter = DirectionalCounter(roi_config, fps=25.0)
    for frame_idx, frame in enumerate(video):
        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        counter.update(tracked, frame_idx)

    counts = counter.get_counts()  # {direction: total_count}
    class_counts = counter.get_class_counts()  # {direction: {vtype: count}}
    timestamps = counter.get_timestamps()  # {direction: [(tracker_id, frame_idx)]}
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import supervision as sv

from vision.detector import VehicleDetector

APPROACHES = ("north", "south", "east", "west")
SENSES = ("in", "out", "both")
DEFAULT_SENSE = "both"


class DirectionalCounter:
    """ROI-based four-direction line-crossing counter with per-type breakdown."""

    def __init__(self, roi_config: Mapping[str, Any], fps: float = 30.0) -> None:
        """
        Args:
            roi_config: dict with a "count_lines" key containing
                {direction: [[x1, y1], [x2, y2]], ...}. Missing directions are
                unobserved. Direction = approach the vehicle is coming from.
                Optional "count_sense": "in" | "out" | "both", or a per-direction
                dict of the same; see the module docstring.
            fps: video frame rate (for timestamp calculations).
        """
        self.fps = float(fps)
        self.roi_config = dict(roi_config)

        count_lines = self.roi_config.get("count_lines") or {}
        self.zones: dict[str, sv.LineZone] = {}
        self.observed: set[str] = set()
        self.sense: dict[str, str] = {}

        raw_sense = self.roi_config.get("count_sense") or DEFAULT_SENSE

        for direction in APPROACHES:
            coords = count_lines.get(direction)
            if not coords or len(coords) != 2:
                continue
            (x1, y1), (x2, y2) = coords
            try:
                start = sv.Point(float(x1), float(y1))
                end = sv.Point(float(x2), float(y2))
                # LineZone uses bottom-center anchor by default for vehicle bbox
                self.zones[direction] = sv.LineZone(
                    start=start,
                    end=end,
                    triggering_anchors=[sv.Position.BOTTOM_CENTER],
                )
                self.observed.add(direction)
            except Exception:
                continue
            s = (raw_sense.get(direction, DEFAULT_SENSE)
                 if isinstance(raw_sense, Mapping) else raw_sense)
            s = str(s).lower()
            if s not in SENSES:
                raise ValueError(
                    f"count_sense for {direction!r} must be one of {SENSES}, got {s!r}")
            self.sense[direction] = s

        # Per-direction cumulative counts
        self._counts: dict[str, int] = {d: 0 for d in self.observed}
        self._class_counts: dict[str, dict[str, int]] = {
            d: defaultdict(int) for d in self.observed
        }

        # Deduplication: {direction: {tracker_id: frame_idx_of_crossing}}
        self._seen: dict[str, dict[int, int]] = {d: {} for d in self.observed}

        # Crossing timestamps for flow_profile binning: {direction: [(tracker_id, frame_idx)]}
        self._timestamps: dict[str, list[tuple[int, int]]] = {
            d: [] for d in self.observed
        }

    def update(self, detections: sv.Detections, frame_idx: int) -> None:
        """Process one frame's tracked detections and update crossing counts.

        Args:
            detections: must have `tracker_id` and `class_id` populated.
            frame_idx: 0-based frame index for timestamp recording.
        """
        if not len(detections):
            return
        if detections.tracker_id is None:
            return

        for direction in self.observed:
            zone = self.zones[direction]
            crossed_in, crossed_out = zone.trigger(detections)

            # Which crossing sense counts as this approach's inbound demand is a
            # property of how the line was drawn, so it comes from the config.
            # "both" is the permissive default and double counts a two-way road.
            sense = self.sense.get(direction, DEFAULT_SENSE)
            if sense == "in":
                mask = crossed_in
            elif sense == "out":
                mask = crossed_out
            else:
                mask = crossed_in | crossed_out

            for i in np.where(mask)[0]:
                tid = int(detections.tracker_id[i])

                # Dedup: only count the first crossing for each tracker_id
                if tid in self._seen[direction]:
                    continue

                self._seen[direction][tid] = frame_idx
                self._counts[direction] += 1
                self._timestamps[direction].append((tid, frame_idx))

                # Record vehicle type
                if detections.class_id is not None:
                    class_id = int(detections.class_id[i])
                    vtype = VehicleDetector.class_name(class_id)
                    self._class_counts[direction][vtype] += 1

    def get_counts(self) -> dict[str, int]:
        """Return total crossings per observed direction."""
        return dict(self._counts)

    def get_class_counts(self) -> dict[str, dict[str, int]]:
        """Return per-type crossings per observed direction."""
        return {d: dict(counts) for d, counts in self._class_counts.items()}

    def get_timestamps(self) -> dict[str, list[tuple[int, int]]]:
        """Return [(tracker_id, frame_idx)] per direction for flow_profile binning."""
        return {d: list(ts) for d, ts in self._timestamps.items()}

    def get_observed(self) -> list[str]:
        """Return list of directions that had count lines configured."""
        return sorted(self.observed)

    def get_sense(self) -> dict[str, str]:
        """Return the crossing sense actually used per observed direction."""
        return dict(self.sense)

    def bin_flow_profile(
        self,
        bin_sec: float,
        total_duration_sec: float,
    ) -> tuple[dict[str, list[int]], list[float]]:
        """Bin crossing timestamps into time windows.

        Args:
            bin_sec: target bin width in seconds.
            total_duration_sec: total video duration (may exceed last crossing).

        Returns:
            (profile_counts, profile_spans) where:
                profile_counts: {direction: [count_in_bin_0, count_in_bin_1, ...]}
                profile_spans: [actual_span_sec_bin_0, ...] (last bin may be shorter)
        """
        n_bins = max(1, int(np.ceil(total_duration_sec / bin_sec)))

        profile_counts: dict[str, list[int]] = {}
        for direction in self.observed:
            bins = [0] * n_bins
            for _, frame_idx in self._timestamps[direction]:
                t_sec = frame_idx / self.fps
                bin_idx = min(int(t_sec / bin_sec), n_bins - 1)
                bins[bin_idx] += 1
            profile_counts[direction] = bins

        # Actual span of each bin (last bin may be shorter than bin_sec)
        spans = [bin_sec] * n_bins
        if n_bins > 0:
            last_span = total_duration_sec - (n_bins - 1) * bin_sec
            spans[-1] = max(0.0, last_span)

        return profile_counts, spans
