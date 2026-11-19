# Sistem Pengukuran Panjang Udang Bawah Air (Underwater Shrimp Measurement)

Proyek sistem visi komputer (*Computer Vision*) semi-otomatis untuk mendeteksi dan mengestimasi panjang tubuh udang secara presisi di dalam air (*submerged chamber*).

## Ringkasan Proyek
- **Tujuan**: Deteksi objek udang dan estimasi panjang fisik (mm) berbasis segmentasi instans (*Instance Segmentation*) dan analisis kurva sumbu tubuh (*centerline/skeleton*).
- **Target Perangkat**: Raspberry Pi 5 (RAM 8 GB) dengan display touchscreen 7" dan akselerator NPU Hailo-8 / Hailo-8L AI Kit (atau CPU via NCNN).
- **Sub-sistem Hardware**: Chamber silinder bawah air, kamera analog/digital, pencahayaan LED adaptif (PWM), sensor kekeruhan air (*turbidity*) via ESP32.
- **Model AI**: YOLO instance segmentation (YOLOv8-seg / YOLO11-seg) yang diekspor ke format Hailo (`.hef`) atau NCNN.

## Struktur Direktori Utama
- `v1117/`: Dataset gabungan, skrip training YOLO, dan artefak model `combined_shrimp_udang2`.
- `v1118/`: Dataset terbaru, konfigurasi training YOLO11, dan skrip export Hailo.
- `_docs/`: Panduan training, spesifikasi teknis, dokumen 3D model, dan firmware driver ESP32.
- `raspy-v1118/`: Aplikasi runtime GUI dan implementasi inferensi di Raspberry Pi 5.

## Referensi Dokumen Lengkap
Detail arsitektur perangkat keras, kebutuhan fungsional, dan alur sistem selengkapnya mengacu pada dokumen berikut:
1. Dokumen PRD Utama (Docs): [`_docs/PRD_Sistem_Udang_Python.md`](_docs/PRD_Sistem_Udang_Python.md)
2. Dokumen PRD Implementasi Raspberry Pi: [`raspy-v1118/PRD.md`](raspy-v1118/PRD.md)
3. Panduan Training & Deploy Raspberry Pi 5: [`_docs/getting-started-yolo-raspy5.md`](_docs/getting-started-yolo-raspy5.md)
