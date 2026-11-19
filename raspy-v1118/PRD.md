

# PRODUCT REQUIREMENTS DOCUMENT (PRD)
**Proyek:** Sistem Pengukuran Panjang Udang Bawah Air
**Versi:** 1.9
**Tanggal:** 4 September 2026
**Platform Utama:** Raspberry Pi 5 (Python GUI)

---

## 1. Ringkasan Eksekutif
Proyek ini mengembangkan sistem *Computer Vision* semi-otomatis untuk mengestimasi ukuran panjang udang secara presisi di bawah air. Sistem menggunakan *chamber* berbentuk tabung (diameter luar 30 cm, panjang 40 cm) dengan dasar datar, plat datar Ø28 cm di tengah tabung sebagai bidang skala (*grid*), dilengkapi dengan pencahayaan adaptif yang menyesuaikan tingkat kekeruhan air. Antarmuka pengguna (GUI) dibangun murni menggunakan Python dan dijalankan secara lokal (Kiosk mode) di Raspberry Pi 5. Pengguna mengoperasikan sistem dengan memantau video *live* dan melakukan *capture* secara manual melalui layar sentuh 7 inch. Jika terdeteksi lebih dari satu udang, sistem memprioritaskan udang terbesar; jika udang terindikasi melayang, pengukuran tetap dilakukan dengan keterangan peringatan.

---

## 2. Arsitektur Sistem (Block Diagram)

Berikut adalah diagram blok interaksi antar perangkat keras. (Gunakan *markdown viewer* yang mendukung Mermaid.js untuk melihat diagram ini, atau Anda bisa membacanya sebagai alur logika).

```mermaid
graph TD
    subgraph "Area Bawah Air (Chamber Tabung Ø30×40 cm)"
        Cam[Kamera Fish Finder Analog]
        LED[Lampu LED Dimmable]
        Turb[Sensor Kekeruhan / Turbidity]
        Dasar[Dasar Tabung Datar]
        Plat[Plat Datar Tengah + Grid Referensi]
    end

    subgraph "Pusat Kontrol (Permukaan)"
        EC[EasyCap Video Capture]
        ESP[ESP32 Microcontroller]
        RPi[Raspberry Pi 5 - 8GB RAM]
        LCD[Touchscreen 7 Inch]
    end

    %% Koneksi Fisik & Data
    Plat -.->|Visual Acuan| Cam
    Cam -->|Kabel CVBS / RCA| EC
    EC -->|USB Video Stream| RPi
    
    Turb -->|Data Analog| ESP
    ESP -->|PWM Control| LED
    
    RPi <-->|Kabel USB - Komunikasi Serial| ESP
    RPi <-->|HDMI / DSI & USB Touch| LCD
    
    %% Alur Software di RPi
    RPi -.->|Python OpenCV| ImageProcessing
    RPi -.->|Python GUI| UserInterface
```

---

## 3. Kebutuhan Perangkat Keras (Hardware Requirements)

### 3.1. Mekanik & Lingkungan
*   **Chamber:** Berbentuk **tabung** (silinder) dengan **diameter luar 30 cm** dan **panjang 40 cm** — bukan lagi kotak/balok. Dinding tabung bening (akrilik/polikarbonat, ketebalan ~3–5 mm). **Dasar chamber datar** (penutup bawah rata, kedap air). **Tutup atas** dapat dibuka untuk memasukkan udang dan maintenance. Tabung diisi air hingga **~5 cm di atas plat tengah** (level isi ≈ 25 cm dari dasar).
    *   **Plat datar tengah:** Dipasang **horizontal di tengah tabung** (setinggi ~20 cm dari dasar, atau setengah panjang). Diameter plat ≈ **28 cm** (clearance **~0.5–0.7 cm** ke dinding dalam, mengingat diameter dalam tabung ≈ 29–29.4 cm setelah dikurangi dinding 3–5 mm) agar tidak menggesek dinding melengkung. Plat terbuat dari akrilik/polikarbonat atau papan PVC dengan permukaan matte; dipasang pada **bracket/siku** menempel dinding dalam (bukan tiang tengah) agar tidak menghalangi pandangan kamera. Udang diletakkan di atas plat ini, bukan langsung di dasar tabung.
    *   **Ruang bawah plat (~20 cm):** Kosong berisi air; menjadi zona sensor kekeruhan dan sirkulasi. Dasar tabung datar memudahkan pemasangan sensor turbidity.
    *   **Alasan plat tengah:** memisahkan zona pengukuran dari dasar tabung, memudahkan pencahayaan, dan memberi bidang rata untuk *grid* referensi.
    *   **Refraksi dinding silinder:** dinding tabung melengkung berisi air membiaskan pandangan kamera akibat refraksi optik (hukum Snell), **bukan** barrel distortion lensa; posisi kamera diset agar pandangan dominan ke plat datar (lihat §3.3), dan *software* menerapkan koreksi refraksi/perspektif bila diperlukan setelah uji lapangan.
*   **Grid Referensi:** Permukaan **plat datar tengah** memakai pola kotak-kotak **2 × 2 cm** (bukan 1 × 1 cm) pada area **lingkaran diameter ~28 cm**. Kotak di tepi lingkaran akan terpotong parsial — kalibrasi skala mengutamakan area tengah plat yang utuh. Ukuran ini adalah skala kalibrasi kamera/konversi piksel→mm. Pada resolusi EasyCap 640×480, lebar bidang pandang efektif ~30 cm, dan satu kotak 2 cm ≈ **40–45 piksel** di area tengah — cukup terdeteksi di video analog bawah air. Kotak 1 cm terlalu halus (hanya ~20 piksel, mudah pecah karena blur/kekeruhan). Ukuran fisik diukur ulang setelah pemasangan (toleransi cetak/cat).
    *   **Warna permukaan plat:** abu-abu tua matte atau hitam matte (bukan mengkilap). Udang vaname umumnya pucat/transparan; latar gelap membuat tubuh lebih mudah disegmentasi. Cat/laminasi mengkilap ditolak karena pantulan LED.
    *   **Warna garis grid:** putih atau kuning pucat. Garis tipis **2–3 mm**, lurus, kontras tinggi terhadap permukaan plat. Hindari papan catur hitam–putih penuh: pola bolak-balik memecah siluet udang saat pengukuran.
    *   **Bukan:** plat putih + garis hitam (udang pucat hampir hilang di latar), dan bukan permukaan mengkilap.

### 3.2. Pemrosesan & Komputasi
*   **Main Board:** Raspberry Pi 5 (RAM 8GB) dengan MicroSD 32GB Class 10/A1.
*   **Display:** Layar Sentuh (*Touchscreen*) 7 Inch.

### 3.3. Visi & Sensor
*   **Kamera:** Kamera *Fish Finder* dengan sensor/LCD resolusi 800x480. Dipasang di **dinding tabung** pada ketinggian **setinggi plat tengah** (~20 cm dari dasar), menghadap **menunduk ~30–45°** ke permukaan plat *grid*. Orientasi ini meminimalkan pantulan dinding melengkung dan menjaga plat terlihat hampir frontal di tengah *frame*. Kabel CVBS/RCA keluar melalui tutup atas atau gland kedap air di dinding.
    *   **Catatan output CVBS:** Sinyal CVBS (composite video) selalu berformat **4:3** (NTSC: 720×480 @ 29.97 fps atau PAL: 720×576 @ 25 fps), terlepas dari resolusi sensor kamera. Kamera fish finder yang mengklaim 800×480 (16:9) pada panel LCD-nya dapat melakukan *anamorphic squeeze* (mampat horizontal), *letterbox* (bar hitam atas-bawah), atau *crop* (potong sisi) saat mengeluarkan sinyal CVBS. **Perilaku output ini harus diuji di lapangan** sebelum menentukan koreksi aspek rasio yang benar di software.
*   **Video Capture:** EasyCap (CVBS to USB) mentransfer video dengan resolusi akhir 640×480 (4:3). Standar CVBS harus cocok antara kamera dan EasyCap (**NTSC atau PAL**, tidak boleh campur).
*   **Pencahayaan:** Modul LED dipasang **di sisi berlawanan kamera** pada ketinggian plat, menyinari plat secara menyamping agar bayangan udang tidak menutupi siluet. Intensitas dikontrol PWM.
*   **Sub-Controller:** ESP32 (Terhubung ke RPi via Kabel Data USB).
*   **Sensor Ekstra:** Modul Sensor Kekeruhan Air (Turbidity) dipasang di **ruang bawah plat**, dekat dasar tabung, untuk membaca kekeruhan air di zona bawah.
*   **Aktuator:** Modul Lampu LED dengan dukungan *dimming* via tegangan PWM.

### 3.4. Catu Daya (Power Supply)
*   **Raspberry Pi 5:** Adaptor USB-C **5V / 5A** (resmi RPi 5 PSU) atau *power bank* high-output.
*   **ESP32 + Sensor Turbidity:** Ditenagai via USB dari RPi (5V, <500 mA total).
*   **LED Dimmable:** Sumber daya terpisah sesuai spesifikasi modul LED (mis. 12V/2A); ground common dengan ESP32 untuk PWM.
*   **Kamera Fish Finder + EasyCap:** Kamera fish finder biasanya punya PSU sendiri (12V); EasyCap ditenagai USB dari RPi.
*   **Penting:** Semua ground (RPi, ESP32, LED driver, kamera) harus **common ground** untuk menghindari noise pada sinyal analog turbidity dan CVBS.

---

## 4. Kebutuhan Perangkat Lunak (Software Requirements)

### 4.1. Teknologi Utama
*   **Sistem Operasi:** Raspberry Pi OS (Debian based).
*   **Bahasa Pemrograman:** Python 3 (di RPi) dan C++/Arduino (di ESP32).
*   **GUI Framework:** CustomTkinter atau PyQt5/6 (Aplikasi *Desktop Native* yang ringan dan cepat).
*   **Image Processing:** OpenCV (`cv2`) dan NumPy.
*   **Komunikasi Data:** `pyserial` (membaca data dari ESP32 via USB).
*   **Penyimpanan Data:** File CSV untuk *logging* (disimpan di partisi MicroSD 32GB).

### 4.2. Logika ESP32 (Firmware)
*   **Input:** Membaca nilai ADC dari sensor Turbidity secara *looping*.
*   **Komunikasi:** Mengirim *raw turbidity index* (nilai ADC / indeks sensor, **bukan NTU**) ke USB Serial (misal: `TURB: 45`). Mendengarkan perintah PWM dari USB Serial (misal: `PWM: 200`).
*   **Output:** Menerapkan nilai PWM ke pin kontrol lampu LED.

### 4.3. Pipeline Pengolahan Citra & Metode Pengukuran Panjang (YOLO + Feret)

Sistem menggunakan model deep learning **YOLO Instance Segmentation** (YOLOv8-seg / YOLO11-seg) yang diakselerasi via NPU Hailo-8 / Hailo-8L (atau CPU via NCNN) untuk memisahkan tubuh udang dari latar belakang, kemudian dilanjutkan dengan analisis morfometri menggunakan **Diameter Feret (Caliper Length)** dan kurva skeleton tubuh.

```
[Frame Kamera / EasyCap]
          │
          ▼
[Pra-proses: CLAHE + Koreksi Rasio/Perspektif]
          │
          ▼
[YOLO Instance Segmentation (Hailo / NCNN)] ──> Deteksi Box, Class, & Binary Mask
          │
          ▼
[Ekstraksi Kontur Mask Tubuh Udang]
          │
    ┌─────┴────────────────────────────────────────┐
    ▼                                              ▼
[Metode 1: Feret Diameter (Max Caliper)]   [Metode 2: Centerline Skeletonization]
(Jarak Euclidean terjauh antar 2 titik)    (Penelusuran kurva tulang belakang)
(Ideal: Udang Lurus / Sedikit Miring)      (Ideal: Udang Melengkung C-Shape)
    └─────┬────────────────────────────────────────┘
          │
          ▼
[Konversi Skala: Pixel ──> Milimeter (mm)] ── (via Kalibrasi Grid Plat / Round Base)
          │
          ▼
[Hasil Panjang Akhir + Validasi Toleransi]
```

#### 4.3.1. Deteksi & Segmentasi Tubuh (YOLO Segmentation)
1. **Model Inferensi**: Output model berupa *binary mask* presisi tinggi untuk tiap instans udang (`top_shrimp` atau `side_shrimp`). Mask memisahkan tubuh udang dari kaki tipis dan antena.
2. **Post-processing Mask**: Mask biner dibersihkan menggunakan operasi morfologi (`cv2.morphologyEx` dengan kernel ellipse 3x3) untuk menghaluskan tepian kontur luar (*boundary contour*).

#### 4.3.2. Penentuan Panjang Udang: Metode Feret Diameter
**Diameter Feret** (dikenal juga sebagai *Maximum Caliper Diameter*) adalah jarak terpanjang antara dua garis sejajar yang mengapit kontur objek dari berbagai orientasi, setara dengan **jarak Euclidean maksimum antara dua titik pada batas kontur (Convex Hull)**:

$$L_{\text{feret}} = \max_{p_i, p_j \in \text{Hull}} \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

1. **Komputasi Feret Maksimum**:
   - Dihitung dari titik-titik *Convex Hull* kontur mask udang (`cv2.convexHull`).
   - Algoritma *Rotating Calipers* dijalankan pada titik hull untuk menemukan pasangan titik terjauh ($P_{\text{head}}$ pada ujung rostrum/kepala dan $P_{\text{tail}}$ pada ujung telson/ekor) dalam kompleksitas $O(N)$.
   - Jarak ini menghasilkan panjang linier fisik udang dalam satuan piksel.
2. **Kapan Menggunakan Feret**:
   - Sangat akurat dan tahan derau (*noise-resistant*) saat udang berada dalam posisi relatif lurus atau hanya membengkok ringan ($< 15^\circ$).
   - Menghindari fluktuasi titik akhir akibat lekukan kaki renang atau antena yang lolos dari mask.

#### 4.3.3. Penanganan Udang Melengkung (Curved / C-Shape)
Jika rasio kelengkungan tinggi (dideteksi dari perbandingan luas mask terhadap luas bounding box atau deviasi bentuk):
1. **Skeletonization / Medial Axis Transform**: Mask biner ditipiskan (*morphological thinning*) menjadi garis kerangka tubuh selebar 1 piksel.
2. **Arc-Length Integral**: Jalur skeleton ditelusuri dari ujung kepala ke ujung ekor sebagai kurva spline/polylines, lalu dihitung total panjang kumulatifnya:

$$L_{\text{skeleton}} = \sum_{k=1}^{n-1} \sqrt{(x_{k+1} - x_k)^2 + (y_{k+1} - y_k)^2}$$

3. **Fallback & Ensembling**: Jika tubuh lurus, sistem memakai **Feret Diameter** karena lebih stabil dari derau percabangan skeleton (*branching artifacts*). Jika tubuh melengkung signifikan, sistem memakai **Skeleton Arc-Length**.

#### 4.3.4. Skala Kalibrasi & Konversi Pixel ke Milimeter (mm)
Panjang piksel dikonversi menjadi satuan fisik milimeter:

$$L_{\text{mm}} = L_{\text{pixel}} \times S_{\text{mm/px}}$$

Faktor skala $S_{\text{mm/px}}$ diperoleh melalui dua opsi kalibrasi:
1. **Grid Referensi Plat (2 × 2 cm)**: Garis kotak-kotak plat tengah dideteksi pada kondisi awal (tanpa udang). Jarak 20 mm dibagi rata-rata jumlah piksel antar garis grid.
2. **Objek Referensi Round Base**: Deteksi mask kelas `round_base` yang diameter fisiknya telah diketahui pasti ($D_{\text{fisik}}$ mm), sehingga:

$$S_{\text{mm/px}} = \frac{D_{\text{fisik}}}{\text{Feret Diameter}(\text{round\_base})_{\text{pixel}}}$$

---

## 5. Kebutuhan Fungsional (Functional Requirements)

1.  **Pengambilan Gambar Manual:** Sistem tidak memotret otomatis. *Frame* video ditahan (di-*capture*) secara seketika saat tombol "Ukur" ditekan di layar.
2.  **Koreksi Distorsi (Pre-processing):** *Software* Python memperbaiki aspek rasio gambar sesuai perilaku output CVBS kamera (lihat §3.3 — koreksi yang benar tergantung apakah kamera melakukan *squeeze*, *letterbox*, atau *crop*). Jika uji lapangan menunjukkan distorsi refraksi dari dinding tabung silinder berisi air atau sudut kamera menunduk, sistem menerapkan koreksi refraksi dan/atau transformasi perspektif (*homography*) agar *grid* 2 × 2 cm tampak mendekati persegi seragam di area pengukuran.
3.  **Pengukuran Udang (Measurement):**
    *   Pipeline inferensi & morfometri mengikuti **§4.3** (YOLO Segmentation + Maximum Feret Caliper Diameter / Skeleton Arc-Length).
    *   *Software* mendeteksi kandidat objek udang di *frame* beserta polygon mask masing-masing.
    *   **Prioritas multi-udang:** jika terdeteksi **lebih dari satu** udang yang terpisah, sistem **hanya mengukur udang terbesar** (kriteria: luas siluet / panjang kandidat terbesar). Overlay GUI menandai udang yang dipilih. Idealnya operator tetap memasukkan satu udang; prioritas terbesar adalah *fallback*, bukan prosedur utama.
    *   **Saling sentuh / bertumpuk:** jika siluet menyatu ambigu (tidak bisa dipisah andal), *capture* di-*reject* (bukan tebak-tebak mana yang terbesar).
    *   Udang **tidak wajib lurus**. Posisi ideal: menempel datar di plat *grid*. Jika terindikasi **melayang / jauh dari plat**, sistem **tetap mengukur**, tetapi menandai hasil dengan keterangan peringatan (lihat poin 4).
    *   Sistem mengukur panjang mengikuti sumbu tubuh (kurva *centerline* / skeleton dari kepala ke ekor), bukan garis lurus ujung-ke-ujung.
    *   Sistem mengkonversi panjang piksel (sepanjang kurva) menjadi ukuran fisik (mm/cm) menggunakan skala *grid* 2 × 2 cm. Skala ini **paling andal** saat udang di bidang plat *grid*; hasil dengan status melayang bersifat estimasi kasar.
4.  **Validasi, Status Hasil & Notifikasi:** Setelah *CAPTURE*, sistem menilai *frame* dan **wajib** menampilkan notifikasi/indikator di GUI untuk **setiap** hasil — termasuk `WARN_*` dan `REJECT`. Tidak ada kegagalan diam-diam tanpa umpan balik.
    *   **`OK`:** satu udang (atau terbesar terpilih), menempel plat, kontras cukup, tidak blur → panjang resmi. Indikator hijau / netral.
    *   **`WARN_MULTI`:** lebih dari satu udang terpisah; panjang dari **udang terbesar**. Indikator kuning + teks (contoh: `Peringatan: diukur udang terbesar dari 3 kandidat`).
    *   **`WARN_FLOATING`:** udang melayang / jauh dari plat; **tetap diukur**. Indikator kuning + teks (contoh: `Peringatan: udang melayang — akurasi skala menurun`).
    *   **`WARN_MULTI_FLOATING`:** multi-udang (terbesar) + melayang. Indikator kuning + kedua keterangan.
    *   **`REJECT_*` (tidak ada panjang resmi) — tetap ada notifikasi jelas + saran singkat:**
        *   `REJECT_OVERLAP` — siluet saling sentuh/bertumpuk ambigu → *“Udang saling menempel. Pisahkan lalu ukur ulang.”*
        *   `REJECT_BLUR` — *motion blur* / buram → *“Gambar buram. Tunggu udang diam lalu capture lagi.”*
        *   `REJECT_LIGHT` — kontras/pencahayaan buruk → *“Pencahayaan kurang. Sesuaikan lampu (Auto/Manual) lalu ulang.”*
        *   `REJECT_NO_OBJECT` — tidak ada objek → *“Udang tidak terdeteksi. Pastikan ada di frame dan kontras.”*
    *   **Indikator visual wajib:** badge/status berwarna di panel hasil + pesan teks yang terbaca di layar 7"; untuk `REJECT_*` boleh banner singkat di atas/bawah video. Overlay *snapshot* (jika ada) ikut menandai masalah bila relevan.
5.  **Kontrol Lampu Adaptif:**
    *   **Mode Otomatis:** GUI menghitung persentase lampu berdasarkan *raw turbidity index*. Semakin keruh, RPi memerintahkan ESP32 menaikkan intensitas PWM LED. Mode Auto **mengurangi** risiko gagal deteksi, tetapi **tidak menjamin** akurasi pengukuran.
    *   **Mode Manual:** Pengguna dapat menimpa (*override*) kecerahan lampu melalui *slider* pada layar sentuh.
6.  **Penyimpanan Data (Data Logging):** Setiap *capture* yang menghasilkan panjang (`OK` atau `WARN_*`) menulis satu baris CSV plus *file* foto. Kolom CSV minimal: *timestamp*, estimasi panjang, status, catatan/pesan notifikasi, jumlah kandidat udang, persentase kecerahan lampu, *raw turbidity index*, nama *file* foto. Setiap `REJECT_*` **juga** di-log (satu baris) dengan status reject + alasan/pesan, tanpa angka panjang sebagai hasil resmi. **Penyimpanan otomatis:** data dan foto langsung tersimpan setelah setiap CAPTURE (tidak perlu konfirmasi user). Hasil *capture* berikutnya ditambahkan sebagai baris baru (tidak menimpa yang sebelumnya).

---

## 6. Batasan Operasional & Sumber Error (Constraints)

Hal berikut bersifat **syarat pakai** dan risiko yang harus dipahami operator:

1.  **Lebih dari satu udang.** Sistem memilih udang terbesar yang terpisah dan memberi notifikasi `WARN_MULTI`. Udang lain diabaikan. Jika saling sentuh/bertumpuk → `REJECT_OVERLAP` + notifikasi saran. Prosedur terbaik: satu udang per *capture*.
2.  **Udang melayang / jauh dari plat.** Pengukuran **tetap dijalankan** dengan `WARN_FLOATING` + indikator. Error skala bisa signifikan; angka panjang **bukan** setara `OK` di plat. Ulangi saat menempel plat bila butuh hasil resmi.
3.  **Pencahayaan & kekeruhan.** Cahaya kurang, silau, atau air sangat keruh → sering `REJECT_LIGHT` dengan notifikasi minta sesuaikan lampu.
4.  **Gerakan saat *capture*.** Udang bergerak → `REJECT_BLUR` + notifikasi tunggu diam.
5.  **Kalibrasi *grid* di air.** Skala 2 × 2 cm diasumsikan setelah kalibrasi kamera/*chamber* terisi air; posisi kamera bergeser, sudut kamera berubah, atau *grid* aus → perlu kalibrasi ulang.
6.  **Plat lingkaran & perspektif.** Kotak *grid* di tepi plat bulat terpotong dan terdistorsi perspektif di pinggir *frame*; pengukuran panjang paling andal di area tengah plat. Pantulan dari dinding tabung bening dapat menambah noise di tepi citra.
7.  **Ambigu anatomi.** Antena, kaki, atau ekor terbuka/tertutup dapat menggeser titik ujung kepala–ekor; operator memastikan tubuh terlihat utuh di *frame*.
8.  **Di luar cakupan v1:** mengukur *semua* udang sekaligus ke banyak baris CSV, konversi turbidity ke NTU, dan target error numerik (±mm) — ditetapkan terpisah bila ada uji lapangan.

---

## 7. Antarmuka Pengguna (GUI Layout)

Sistem menggunakan desain *Single-Page Dashboard* **orientasi landscape** layar penuh agar mudah ditekan:
*   **Area Tengah (Dominan):** Panel *Live Video Feed* secara *real-time*. Setelah *capture*: *snapshot* + overlay (kurva tubuh / penanda masalah) **plus** banner/badge status yang selalu terlihat.
*   **Area Bawah (Action Bar):** Tombol **"CAPTURE / UKUR"** yang besar dan mudah dijangkau jari.
*   **Sisi Kanan (Informasi & Kontrol):**
    *   **Hasil Ukur + Indikator Status** (wajib selalu terisi setelah *capture*):
        *   hijau `OK` + angka panjang (contoh: `120.5 mm`)
        *   kuning `WARN_MULTI` / `WARN_FLOATING` / `WARN_MULTI_FLOATING` + angka + teks peringatan
        *   merah `REJECT_OVERLAP` / `REJECT_BLUR` / `REJECT_LIGHT` / `REJECT_NO_OBJECT` + pesan saran (tanpa angka panjang resmi)
    *   **Status Kekeruhan:** *Raw turbidity index* (contoh: `TURB 45`), bukan satuan NTU.
    *   **Kontrol Lampu:** 
        *   *Toggle Switch* (Auto / Manual).
        *   *Slider* (0% - 100%) untuk menyesuaikan daya lampu.

---

## 8. Alur Kerja Operasional (User Flow)
1.  **Booting:** Alat dinyalakan. Raspberry Pi memuat OS dan otomatis menjalankan script Python GUI layar penuh (Kiosk, orientasi landscape).
2.  **Kalibrasi Cahaya:** User mengecek *live feed*. Jika air keruh, user menyalakan mode `Auto` atau menggeser *slider* manual hingga plat *grid* dan tubuh udang terlihat kontras.
3.  **Posisi Udang:** Sampel dimasukkan ke *chamber* tabung. Ideal: **satu** udang, menempel datar di plat *grid* tengah, relatif diam. Jika ada beberapa udang, sistem nanti memilih yang terbesar (+ notif). Jika masih melayang, tetap bisa diukur (+ notif peringatan).
4.  **Eksekusi:** User menekan "CAPTURE". Program mendeteksi kandidat, memilih udang terbesar bila multi, menilai posisi/blur/kontras, lalu menampilkan hasil **beserta indikator/notifikasi**. Hasil langsung tersimpan ke CSV + foto (otomatis, tanpa konfirmasi).
5.  **Hasil:**
    *   **`OK` / `WARN_*`:** panjang + indikator/teks; data + foto **otomatis tersimpan** ke CSV.
    *   **`REJECT_*`:** indikator merah + pesan alasan & saran; **otomatis di-log** sebagai reject; user koreksi lalu ulangi dari langkah 3.
6.  **Siap ulang:** Sistem kembali ke mode *live feed* untuk pengukuran berikutnya. Tidak ada batasan jumlah *capture*; setiap hasil ditambahkan sebagai baris baru di CSV.

