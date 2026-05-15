# Pendeteksi Penyakit Paru

Aplikasi Flask + YOLO untuk membantu observasi anomali pneumonia pada citra X-Ray paru.

## Menjalankan Aplikasi

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka:

```text
http://127.0.0.1:5000
```

Endpoint API:

```text
POST http://127.0.0.1:5000/api/predict
```

Form-data harus memakai field `file`. Format yang didukung: JPG, PNG, BMP, WEBP, DICOM (`.dcm`/`.dicom`).

## Konfigurasi Prediksi

Threshold YOLO default dibuat lebih rendah untuk prototype medis karena model saat ini masih lemah.

```powershell
$env:PULMO_YOLO_CONF="0.05"
$env:PULMO_YOLO_IOU="0.45"
$env:PULMO_YOLO_MODEL="models/yolo_lung.pt"
python app.py
```

Jika terlalu banyak false positive, naikkan `PULMO_YOLO_CONF` bertahap, misalnya `0.10`, `0.15`, lalu `0.25`.

## Training YOLO

Dataset mentah dari `prepare_dataset.py` masuk ke `dataset_yolo/images/train` dan `dataset_yolo/labels/train`. Sebelum training, data perlu split train/val/test:

```powershell
python split_yolo_dataset.py
```

Training ulang:

```powershell
python train_yolo.py --epochs 50 --batch 8
```

Script akan menyalin `best.pt` ke `models/yolo_lung.pt` agar langsung dipakai aplikasi.

## Catatan Penting

Model severity CNN lama tidak dipakai karena masih dummy dan belum punya label `Ringan/Sedang/Parah` yang valid. Untuk sementara, aplikasi memakai estimasi severity berbasis luas bounding box dan confidence. Untuk hasil medis yang serius, severity perlu dataset berlabel dokter.

Hasil aplikasi ini adalah bantuan observasi AI, bukan diagnosis final.
