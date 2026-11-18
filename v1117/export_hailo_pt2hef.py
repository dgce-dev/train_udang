from ultralytics import YOLO

model = YOLO("runs/segment/combined_shrimp_udang2-4/weights/best.pt")
model.export(
    format="hailo",
    name="hailo8",  # atau "hailo8l" jika pakai Raspberry Pi AI Kit standar (13 TOPS)
    imgsz=640,
    data="combined_shrimp_udang2/data.yaml",
)
