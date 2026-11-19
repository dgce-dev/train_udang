#!/usr/bin/env python3
"""Train YOLO instance segmentation on combined_shrimp_udang2."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    # ponytail: default ke yolov8n-seg.pt; ganti path checkpoint kalau punya pre-trained khusus
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=str(ROOT / "combined_shrimp_udang2" / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(ROOT / "runs" / "segment"),
        name="combined_shrimp_udang2",
    )
