"""YOLO vehicle detector wrapper (ultralytics -> supervision.Detections).

Only the four COCO vehicle classes that TrafficState's ``vehicle_mix`` knows about
are kept; everything else (person, traffic light, ...) is dropped at the source so
downstream tracking/counting never sees a non-vehicle.

Usage:
    det = VehicleDetector()                 # models/yolo11s.pt, cuda if available
    detections = det.detect(frame_bgr)      # -> sv.Detections
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "yolo11s.pt"

# COCO class id -> TrafficState vehicle_mix category (schema 1.1)
VEHICLE_CLASSES: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASS_IDS: tuple[int, ...] = tuple(sorted(VEHICLE_CLASSES))

DEFAULT_CONF = 0.3
DEFAULT_IOU = 0.5
DEFAULT_IMGSZ = 1280

_ASSET_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"


def ensure_weights(weights: str | Path = DEFAULT_WEIGHTS) -> Path:
    """Return a local weights path, downloading the ultralytics asset if missing."""
    weights = Path(weights)
    if weights.exists():
        return weights
    weights.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        got = Path(attempt_download_asset(weights.name))
    except Exception:
        import urllib.request

        got = weights.parent / weights.name
        urllib.request.urlretrieve(_ASSET_URL.format(name=weights.name), got)
    if got.exists() and got.resolve() != weights.resolve():
        shutil.move(str(got), str(weights))
    if not weights.exists():
        raise FileNotFoundError(f"could not obtain YOLO weights: {weights}")
    return weights


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class VehicleDetector:
    """Single-frame YOLO inference returning ``sv.Detections`` of vehicles only."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        device: str | None = None,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> None:
        from ultralytics import YOLO

        self.weights = ensure_weights(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = pick_device(device)
        self.model = YOLO(str(self.weights))
        self.model.to(self.device)
        # fp16 roughly halves 4K inference time and is lossless enough for detection
        self.half = self.device.startswith("cuda")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=list(VEHICLE_CLASS_IDS),
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        if len(detections) and detections.class_id is not None:
            detections = detections[np.isin(detections.class_id, VEHICLE_CLASS_IDS)]
        return detections

    @staticmethod
    def class_name(class_id) -> str:
        """COCO id -> vehicle_mix key; unknown ids fold into 'car' per contract."""
        try:
            return VEHICLE_CLASSES.get(int(class_id), "car")
        except (TypeError, ValueError):
            return "car"

    def describe(self) -> dict:
        return {
            "weights": str(self.weights),
            "device": self.device,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "classes": {str(k): v for k, v in VEHICLE_CLASSES.items()},
        }
