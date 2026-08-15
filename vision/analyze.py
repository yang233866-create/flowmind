"""Vision analysis CLI: video → ROI-based counting → TrafficState JSON + figures.

CLI:
    python -m vision.analyze --video data/videos/demo.mp4 \\
        --roi-config data/videos/demo_roi.json \\
        --state-out data/traffic_states/demo_001.json \\
        [--annotated-video data/videos/demo_annotated.mp4] \\
        [--figures-dir figures] \\
        [--max-frames 1500]

ROI config format (JSON):
    {
      "count_lines": {
        "north": [[x1, y1], [x2, y2]],
        "south": [[x1, y1], [x2, y2]],
        "east": [[x1, y1], [x2, y2]],
        "west": [[x1, y1], [x2, y2]]
      }
    }

Directions not listed in count_lines are treated as unobserved and filled per
schema 1.1 fallback rules (mirror opposing direction or default to 400 vph).

Outputs:
    - TrafficState JSON (schema 1.1) at --state-out
    - Optional annotated video at --annotated-video (boxes + tracks + count lines)
    - Four PNG figures in --figures-dir (if given):
        * fig_vision_flow_timeline.png
        * fig_vision_vehicle_mix.png
        * fig_vision_annotated_frame.png
        * fig_vision_track_heatmap.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from tqdm import tqdm

# Add repo root to sys.path so we can import from scripts/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.sci_style import DIRECTION_COLORS, apply_style, save_fig
from vision.counter import DirectionalCounter
from vision.detector import VehicleDetector
from vision.tracker import VehicleTracker
from vision.traffic_state import (
    build_traffic_state,
    validate_traffic_state,
    write_traffic_state,
)

APPROACHES = ("north", "south", "east", "west")
DIRECTION_LABELS = {"north": "North", "south": "South", "east": "East", "west": "West"}


def analyze_video(
    video_path: str | Path,
    roi_config: dict,
    scenario_id: str,
    state_out: str | Path,
    annotated_video: str | Path | None = None,
    figures_dir: str | Path | None = None,
    max_frames: int | None = None,
    allow_invalid: bool = False,
) -> Path:
    """Run full vision pipeline: detect → track → count → TrafficState + figures.

    Args:
        allow_invalid: write the TrafficState even if it violates the schema 1.1
            contract. Off by default -- an invalid state silently poisons every
            simulation built from it.

    Returns:
        Path to the written TrafficState JSON.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames and max_frames > 0:
        total_frames = min(total_frames, max_frames)

    print(f"[analyze] video: {video_path.name}")
    print(f"[analyze]   {w}×{h} @ {fps:.1f} fps, {total_frames} frames")
    print(f"[analyze]   duration: {total_frames / fps:.1f} s")

    detector = VehicleDetector()
    print(f"[analyze] detector: {detector.describe()}")

    tracker = VehicleTracker(fps=fps)
    print(f"[analyze] tracker: ByteTrack @ {fps:.1f} fps")

    counter = DirectionalCounter(roi_config, fps=fps)
    print(f"[analyze] counter: observed directions = {counter.get_observed()}")
    print(f"[analyze]   crossing sense = {counter.get_sense()} "
          "('both' counts outbound traffic too on a two-way road)")

    # Annotated video writer
    writer = None
    if annotated_video:
        annotated_video = Path(annotated_video)
        annotated_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely compatible
        writer = cv2.VideoWriter(
            str(annotated_video), fourcc, fps, (w, h)
        )

    # Annotators for the video
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=60)
    line_annotators = {
        d: sv.LineZoneAnnotator(
            thickness=4,
            text_thickness=2,
            text_scale=0.6,
            color=sv.Color.from_hex(DIRECTION_COLORS[d]),
        )
        for d in counter.get_observed()
    }

    # For heatmap: accumulate track positions
    track_positions: dict[int, list[tuple[float, float]]] = {}

    # Main loop
    frame_idx = 0
    first_frame = None
    pbar = tqdm(total=total_frames, desc="[analyze]", unit="fr")

    while frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if first_frame is None:
            first_frame = frame.copy()

        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        counter.update(tracked, frame_idx)

        # Accumulate positions for heatmap
        if len(tracked) and tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                x1, y1, x2, y2 = tracked.xyxy[i]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                track_positions.setdefault(tid, []).append((cx, cy))

        # Annotate frame
        if writer:
            annotated = frame.copy()
            annotated = trace_annotator.annotate(annotated, tracked)
            annotated = box_annotator.annotate(annotated, tracked)

            labels = []
            if len(tracked) and tracked.class_id is not None:
                for class_id in tracked.class_id:
                    vtype = detector.class_name(class_id)
                    labels.append(vtype)
            if labels:
                annotated = label_annotator.annotate(annotated, tracked, labels=labels)

            for direction in counter.get_observed():
                zone = counter.zones[direction]
                annotated = line_annotators[direction].annotate(annotated, zone)

            writer.write(annotated)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()
        print(f"[analyze] annotated video: {annotated_video}")

    duration_sec = frame_idx / fps

    # Build TrafficState
    counts = counter.get_counts()
    class_counts = counter.get_class_counts()
    profile_counts, profile_spans = counter.bin_flow_profile(
        bin_sec=5.0,  # 5s bins for the 21.5s demo video
        total_duration_sec=duration_sec,
    )

    state = build_traffic_state(
        scenario_id=scenario_id,
        observed=counter.get_observed(),
        counts=counts,
        class_counts=class_counts,
        duration_sec=duration_sec,
        video=str(video_path),
        fps=fps,
        frames=frame_idx,
        queue_est=None,  # vision cannot estimate queue length reliably
        profile_counts=profile_counts,
        profile_spans=profile_spans,
        profile_bins_sec=5,
    )

    # Validate before writing: a state that violates the contract would fail
    # downstream in route generation, where the cause is much harder to see.
    problems = validate_traffic_state(state)
    if problems:
        print("[analyze] TrafficState validation FAILED:")
        for p in problems:
            print(f"  - {p}")
        if not allow_invalid:
            raise ValueError(
                f"{len(problems)} TrafficState contract violation(s); refusing to write "
                f"{state_out}. Pass allow_invalid=True / --allow-invalid to override.")
        print("[analyze] WARNING: writing anyway (--allow-invalid)")

    state_path = write_traffic_state(state, state_out)
    print(f"[analyze] TrafficState: {state_path}")

    # Generate figures
    if figures_dir:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        _generate_figures(
            state, first_frame, track_positions, (w, h), figures_dir, scenario_id
        )

    return state_path


def _generate_figures(
    state: dict,
    first_frame: np.ndarray | None,
    track_positions: dict[int, list[tuple[float, float]]],
    frame_size: tuple[int, int],
    figures_dir: Path,
    scenario_id: str,
) -> None:
    """Generate four PNG figures per contract."""
    apply_style()

    # 1. Flow timeline (flow_profile over time)
    fig, ax = plt.subplots(figsize=(8, 4))
    profile = state.get("flow_profile") or {}
    bins_sec = state.get("profile_bins_sec", 5)
    for d in APPROACHES:
        ys = profile.get(d) or []
        if not ys:
            continue
        xs = [i * bins_sec for i in range(len(ys))]
        ax.plot(xs, ys, marker="o", label=DIRECTION_LABELS[d],
                color=DIRECTION_COLORS[d], linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flow (vph)")
    ax.set_title("Traffic Flow Timeline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, figures_dir / f"fig_vision_flow_timeline.png")

    # 2. Vehicle mix (stacked bar)
    fig, ax = plt.subplots(figsize=(7, 4))
    approaches_data = state.get("approaches") or {}
    vtypes = ["car", "bus", "truck", "motorcycle"]
    vtype_labels = {"car": "Car", "bus": "Bus", "truck": "Truck", "motorcycle": "Motorcycle"}
    directions_plot = [d for d in APPROACHES if approaches_data.get(d, {}).get("observed")]
    if directions_plot:
        bottom = np.zeros(len(directions_plot))
        for vt in vtypes:
            vals = [
                (approaches_data.get(d, {}).get("vehicle_mix") or {}).get(vt, 0) * 100
                for d in directions_plot
            ]
            ax.bar([DIRECTION_LABELS[d] for d in directions_plot], vals,
                   bottom=bottom, label=vtype_labels[vt], width=0.6)
            bottom += np.array(vals)
    ax.set_ylabel("Vehicle Mix (%)")
    ax.set_title("Vehicle Type Distribution by Approach")
    ax.legend()
    ax.set_ylim(0, 100)
    save_fig(fig, figures_dir / f"fig_vision_vehicle_mix.png")

    # 3. Annotated first frame with count lines
    if first_frame is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title("First Frame with Count Lines")
        save_fig(fig, figures_dir / f"fig_vision_annotated_frame.png")

    # 4. Track heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    w, h = frame_size
    heatmap = np.zeros((h, w), dtype=np.float32)
    for positions in track_positions.values():
        for cx, cy in positions:
            x, y = int(cx), int(cy)
            if 0 <= x < w and 0 <= y < h:
                heatmap[y, x] += 1
    if heatmap.max() > 0:
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    im = ax.imshow(heatmap, cmap="hot", interpolation="bilinear", aspect="auto")
    ax.set_title("Vehicle Track Heatmap")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Track Density")
    save_fig(fig, figures_dir / f"fig_vision_track_heatmap.png")

    print(f"[analyze] figures saved to {figures_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Vision analysis: video → TrafficState")
    ap.add_argument("--video", required=True, help="Input video file")
    ap.add_argument("--roi-config", required=True, help="ROI config JSON with count_lines")
    ap.add_argument("--state-out", required=True, help="Output TrafficState JSON path")
    ap.add_argument("--annotated-video", default=None, help="Optional annotated video output")
    ap.add_argument("--figures-dir", default=None, help="Optional figures output directory")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    ap.add_argument("--allow-invalid", action="store_true",
                    help="write the TrafficState even if it violates schema 1.1")
    args = ap.parse_args()

    roi_config = json.loads(Path(args.roi_config).read_text(encoding="utf-8"))
    scenario_id = Path(args.state_out).stem

    analyze_video(
        video_path=args.video,
        roi_config=roi_config,
        scenario_id=scenario_id,
        state_out=args.state_out,
        annotated_video=args.annotated_video,
        figures_dir=args.figures_dir,
        max_frames=args.max_frames,
        allow_invalid=args.allow_invalid,
    )


if __name__ == "__main__":
    main()
