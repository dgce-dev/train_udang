from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

model = YOLO(str(ROOT / "runs" / "segment" / "dataset-2" / "weights" / "best.pt"))
model.export(
    format="hailo",
    name="hailo8l",  # atau "hailo8" untuk modul 26 TOPS
    imgsz=640,
    data=str(ROOT / "dataset" / "data.yaml"),
)
