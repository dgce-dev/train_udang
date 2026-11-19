from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="hailo",
    name="hailo8",
    imgsz=640,
    data="data.yaml"
)