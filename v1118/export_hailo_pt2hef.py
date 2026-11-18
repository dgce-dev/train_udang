from ultralytics import YOLO

model = YOLO(r"D:\C\Download\train_udang\v1118\runs\segment\dataset-2\weights\best.pt")
model.export(
    format="hailo",
    name="hailo8l",  # atau "hailo8" untuk modul 26 TOPS
    imgsz=640,
    data="v1118/dataset/data.yaml",
)
