# Panduan Lengkap: Training YOLO di PC & Deploy ke Raspberry Pi 5 (Hailo NPU & CPU)

Panduan end-to-end melatih model YOLO Instance Segmentation (deteksi & segmentasi udang) di PC Windows, mengekspor model, dan menjalankannya di **Raspberry Pi 5 Model B (8 GB)** menggunakan **Raspberry Pi AI Kit (Hailo-8 / Hailo-8L NPU)** maupun CPU (NCNN).

---

## 1. Spesifikasi Perangkat

- **PC Training**:
  - OS: Windows 11
  - GPU: NVIDIA GeForce GTX 1660 (6 GB VRAM)
  - Python: 3.13 / PyTorch 2.6.0+cu124
- **Edge Device**:
  - SBC: Raspberry Pi 5 Model B Rev 1.1 (8 GB RAM)
  - Akselerator (Opsional/Rekomendasi): Raspberry Pi AI Kit (Hailo-8L 13 TOPS M.2 HAT)

---

## 2. Alur Kerja (End-to-End Workflow)

```
[1. Dataset Siap] ──> [2. Training di PC] ──> [3. Export Model (.hef / .ncnn)]
                                                          │
                                                    Transfer File
                                                          ▼
[5. Realtime Cam] <── [4. Inferensi di RPi 5] <── [File Model di RPi 5]
```

---

## 3. Tahap 0: Penyiapan Dataset & Pelabelan (Dataset Preparation)

### Apakah Cukup Mengumpulkan Foto Saja?
**Tidak cukup.** Foto tanpa label membuat AI tidak tahu objek mana yang harus dideteksi.
Setiap 1 file gambar (misal `shrimp_01.jpg`) **wajib memiliki 1 file teks label dengan nama identik** (misal `shrimp_01.txt`). File teks ini berisi daftar ID kelas dan rangkaian koordinat titik polygon yang membungkus objek:
```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
```

### Memberi Label Menggunakan Tools Apa?
Untuk tugas **Instance Segmentation (Polygon Mask)**, gunakan software anotasi:
1. **Roboflow (Rekomendasi / Paling Mudah)**:
   - Berbasis browser web. Upload foto, buat polygon titik-titik keliling objek, tentukan nama kelas.
   - Fitur export otomatis: pilih format **YOLOv8 Segmentation**. Roboflow otomatis membagi gambar ke folder `train/`, `valid/`, `test/`, dan membuatkan file `data.yaml`.
2. **CVAT (Computer Vision Annotation Tool)**:
   - Gratis, open-source (bisa via web online atau self-hosted via Docker). Sangat handal untuk segmentasi polygon presisi tinggi.
3. **Labelme**:
   - Software desktop lokal Python (`pip install labelme`). Menggambar polygon manual offline di komputer, lalu dikonversi ke format koordinat teks YOLO.

### Mengapa Dataset Wajib Dibagi Menjadi 3 Folder?
Pemisahan dataset (umumnya rasio **70% : 20% : 10%**) sangat penting dalam machine learning:

| Folder | Porsi | Fungsi & Analoginya |
|---|---|---|
| **`train/`** | ~70% | **Buku Pelajaran Utama**. Kumpulan gambar dan label yang dipelajari model secara intensif berulang-ulang di setiap epoch untuk memperbarui bobot neural network. |
| **`valid/`** | ~20% | **Kuis / Tryout Berkala**. Gambar yang tidak pernah dipakai langsung untuk melatih bobot, melainkan diuji setiap akhir epoch untuk mengukur akurasi (`mAP`) dan mendeteksi apakah model mengalami *overfitting* (hanya hafal data latih). |
| **`test/`** | ~10% | **Ujian Akhir Mandiri**. Kumpulan data yang sama sekali belum pernah dilihat oleh model selama proses training atau penyetelan parameter. Digunakan untuk evaluasi final independen sebelum dikirim ke Raspberry Pi. |

---

## 4. Tahap 1: Training Model di PC

Lakukan training di PC menggunakan GPU. Dataset sudah dikonfigurasi di `combined_shrimp_udang2/data.yaml`.

### Memilih Versi YOLO di `train.py`
Versi dan ukuran model ditentukan di baris inisialisasi `train.py`:
```python
model = YOLO("yolov8n-seg.pt")
```
- **`yolov8n-seg.pt`**: YOLOv8 ukuran nano (`n`) segmentasi. Paling ringan, rekomendasi utama untuk Raspberry Pi.
- **`yolo11n-seg.pt`**: YOLOv11 ukuran nano segmentasi (arsitektur lebih baru).
- **Varian ukuran lain**: `yolov8s-seg.pt` (small), `yolov8m-seg.pt` (medium) — lebih akurat tetapi lebih berat.

### Base Model (`.pt` awal) vs Custom Model (`best.pt`)
- **`yolov8n-seg.pt`**: Model dasar pre-trained COCO (80 objek umum). **Belum bisa** mendeteksi udang; hanya dipakai sebagai modal dasar latihan.
- **`best.pt`** (hasil training): Model yang sudah mempelajari dataset udang Anda. **File inilah yang wajib diexport dan dipakai di Raspberry Pi 5**.

Jalankan training di terminal PC:
```cmd
python train.py
```
Atau via CLI Ultralytics langsung:
```cmd
yolo segment train data=combined_shrimp_udang2/data.yaml model=yolov8n-seg.pt epochs=50 imgsz=640 batch=16
```
File bobot terbaik tersimpan di:
`runs/segment/combined_shrimp_udang2/weights/best.pt`

### Parameter & Indikator Hasil Training Bagus

Nilai metrik dapat dilihat di terminal selama training atau di file `runs/segment/.../results.csv` dan grafik `results.png`:

#### 1. Metrik Akurasi (0.0 s/d 1.0 atau 0% s/d 100%)
- **`mAP50(M)`** *(Mask)*: Akurasi segmentasi polygon pada IoU threshold 0.50. Target bagus: **> 0.80 – 0.90+** (80%–90%+).
- **`mAP50-95(M)`**: Rata-rata akurasi polygon pada ambang batas ketat (0.50–0.95). Target bagus: **> 0.50 – 0.70+**.
- **`mAP50(B)`** *(Box)*: Akurasi bounding box deteksi. Target bagus: **> 0.85 – 0.95+**.
- **`Precision (P)`**: Ketepatan deteksi. Nilai tinggi (> 0.80) berarti sangat sedikit *false alarm* (bukan udang tapi ditebak udang).
- **`Recall (R)`**: Kelengkapan deteksi. Nilai tinggi (> 0.80) berarti sangat sedikit udang yang terlewat dari deteksi.

#### 2. Pola Penurunan Loss (Kesalahan Model)
- **`train/box_loss`, `train/seg_loss`, `train/cls_loss`**: Harus turun konsisten dari epoch pertama hingga akhir.
- **`val/box_loss`, `val/seg_loss`, `val/cls_loss`**: Ikut turun lalu mendatar stabil.
- **Waspada Overfitting**: Jika `train loss` terus turun mendekati 0 tetapi `val loss` berbalik naik tajam, model mengalami *overfitting* (hanya hafal data latih).

#### 3. Verifikasi Visual di Folder Run
- **`val_batch0_pred.jpg`**: Cek langsung apakah polygon pas membungkus tubuh udang dan kelasnya tidak tertukar.
- **`confusion_matrix.png`**: Kotak diagonal harus dominan terang (kelas diprediksi tepat sesuai kenyataan).
- **`results.png`**: Grafik garis mAP harus menanjak tinggi lalu stabil di epoch akhir.

---

## 5. Tahap 2: Export Model untuk Raspberry Pi 5

### Opsi A: Menggunakan Raspberry Pi AI Kit (NPU Hailo-8 / Hailo-8L) — *Paling Cepat (30-60+ FPS)*

> [!WARNING]
> **PENTING (Batasan OS Hailo DFC):**
> Hailo Dataflow Compiler (DFC) **hanya mendukung Linux x86_64**.
> Menjalankan export Hailo langsung di Windows 11 akan memunculkan error:
> `AssertionError: Hailo export is only supported on Linux x86_64.`

Gunakan skrip [`export_hailo_pt2hef.py`](export_hailo_pt2hef.py:1) atau snippet:

```python
from ultralytics import YOLO

# Muat model PyTorch hasil training
model = YOLO("runs/segment/combined_shrimp_udang2-4/weights/best.pt")

# Export ke format Hailo Executable Format (.hef)
# - name="hailo8l" untuk Raspberry Pi AI Kit resmi (13 TOPS)
# - name="hailo8" untuk modul M.2 Hailo-8 penuh (26 TOPS)
model.export(
    format="hailo",
    name="hailo8l",
    imgsz=640,
    data="combined_shrimp_udang2/data.yaml"  # Kalibrasi kuantisasi INT8
)
```

#### Cara Menjalankan Export Hailo dari Windows:

**Solusi 1: Gunakan WSL2 (Ubuntu) di Windows 11 (Rekomendasi)**
1. Pasang WSL Ubuntu via PowerShell/Terminal Admin:
   ```cmd
   wsl --install -d Ubuntu
   ```
2. Buka terminal Ubuntu WSL, arahkan ke folder project Windows:
   ```bash
   cd /mnt/d/C/Download/train_udang
   ```
3. Buat virtual environment & install dependensi:
   ```bash
   python3 -m venv .venv_wsl
   source .venv_wsl/bin/activate
   pip install ultralytics hailo-dataflow-compiler
   ```
4. Jalankan export:
   ```bash
   python export_hailo_pt2hef.py
   ```

**Solusi 2: Export Cloud via Ultralytics HUB**
- Upload file `best.pt` ke [Ultralytics HUB](https://hub.ultralytics.com/).
- Pilih opsi export format `Hailo (HEF)` dari dashboard cloud.

Hasil export: `runs/segment/combined_shrimp_udang2-4/weights/best_hailo8l.hef`.

---

### Opsi B: Menggunakan CPU Murni Raspberry Pi 5 (Format NCNN) — *Tanpa NPU Tambahan (~20-30 FPS)*

Jika belum memasang hardware Hailo AI Kit, export ke format NCNN (sangat dioptimalkan untuk CPU ARM Cortex-A76 Raspberry Pi 5):

```cmd
yolo export model=runs/segment/combined_shrimp_udang2/weights/best.pt format=ncnn imgsz=640
```

Hasil export: Folder `best_ncnn_model` berisi file `model.bin` dan `model.param`.

---

## 6. Tahap 3: Pindahkan Model ke Raspberry Pi 5

Salin file model ke Raspberry Pi 5 menggunakan SCP via terminal:

```bash
# Contoh pengiriman file .hef
scp runs/segment/combined_shrimp_udang2/weights/best_hailo8l.hef pi@<IP_RASPBERRY_PI>:~/train_udang/

# Atau contoh pengiriman folder NCNN
scp -r runs/segment/combined_shrimp_udang2/weights/best_ncnn_model pi@<IP_RASPBERRY_PI>:~/train_udang/
```

---

## 7. Tahap 4: Setup & Inferensi di Raspberry Pi 5

Buka terminal di Raspberry Pi 5 (SSH atau Desktop Terminal).

### Setup Driver Hailo (Hanya jika pakai Raspberry Pi AI Kit)
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y hailo-all rpicam-apps

# Cek apakah chip Hailo terdeteksi
hailortcli fw-control identify
```

### Install Python Packages di RPi 5
```bash
pip install ultralytics opencv-python
```

### Skrip Python Deteksi Realtime dari Kamera di RPi 5 (`detect_camera.py`)

```python
import cv2
from ultralytics import YOLO

# 1. Muat model (pilih sesuai opsi yang Anda gunakan):
# Untuk Hailo AI Kit:
model = YOLO("best_hailo8l.hef")

# Untuk NCNN (CPU):
# model = YOLO("best_ncnn_model")

# 2. Buka kamera (Kamera USB = 0, atau gunakan rpicam/Picamera2)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

print("Memulai deteksi... Tekan 'q' untuk keluar.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inferensi deteksi & segmentasi udang
    results = model(frame, conf=0.5)

    # Visualisasikan bounding box & segmentasi polygon
    annotated_frame = results[0].plot()

    cv2.imshow("Deteksi Udang - Raspberry Pi 5", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Jalankan di Raspberry Pi:
```bash
python detect_camera.py
```
