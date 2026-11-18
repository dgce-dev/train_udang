# Tech Stack & Environment

Project train_udang: fine-tuning & training model YOLO Instance Segmentation (deteksi & segmentasi udang).

## 1. System & Hardware
- **Operating System**: Windows 11 (build x86_64)
- **GPU**: NVIDIA GeForce GTX 1660 (6 GB VRAM)
- **NVIDIA Driver**: 610.88
- **CUDA Runtime / Driver Version**: CUDA 13.3 (Driver) / CUDA 12.4 (PyTorch build)

## 2. Runtime & Core Interpreter
- **Python**: 3.13 (C:\laragon\bin\python\python-3.13\python.exe)
- **Pip**: 24.2

## 3. Deep Learning & Computer Vision
- **PyTorch (`torch`)**: 2.6.0+cu124 (CUDA GPU support aktif)
- **TorchVision (`torchvision`)**: 0.21.0+cu124
- **Ultralytics (`ultralytics`)**: 8.4.215
- **OpenCV (`opencv-python`)**: 5.1.0.95
- **Pillow (`pillow`)**: 13.0.0
- **NumPy (`numpy`)**: 2.5.4
- **Matplotlib (`matplotlib`)**: 3.11.2

## 4. Model & Dataset
- **Base Architecture**: YOLOv8 Nano Segment (`yolov8n-seg.pt`)
- **Dataset Format**: YOLO Instance Segmentation (Polygon text labels)
- **Config Parser**: PyYAML 6.0.3
- **Classes**: 4 kelas (`box`, `round_base`, `side_shrimp`, `top_shrimp`)

## 5. Supporting Tools & Utilities
- **tqdm**: 4.70.0
- **psutil**: 7.2.2
- **ninja**: 1.13.2
- **requests**: 2.32.5
