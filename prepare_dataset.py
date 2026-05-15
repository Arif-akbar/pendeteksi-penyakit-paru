import os
import cv2
import pandas as pd
import pydicom
import numpy as np

# Konfigurasi Path
DATASET_DIR = "dataset_rsna/stage_2_train_images"
CSV_PATH = "dataset_rsna/stage_2_train_labels.csv"

# Direktori Output untuk YOLO
YOLO_IMG_DIR = "dataset_yolo/images/train"
YOLO_LBL_DIR = "dataset_yolo/labels/train"

# Direktori Output untuk CNN (MobileNetV2 dkk)
CNN_DIR = "dataset_cnn/train"
CNN_CLASSES = ["Normal", "Pneumonia"]

# Buat semua direktori yang dibutuhkan
for path in [YOLO_IMG_DIR, YOLO_LBL_DIR, f"{CNN_DIR}/{CNN_CLASSES[0]}", f"{CNN_DIR}/{CNN_CLASSES[1]}"]:
    os.makedirs(path, exist_ok=True)

print("Membaca anotasi CSV...")
df = pd.read_csv(CSV_PATH)
df.fillna(0, inplace=True) # Tangani nilai kosong (pasien sehat)

def process_data():
    # Kelompokkan berdasarkan ID pasien karena satu pasien bisa punya >1 bounding box
    grouped = df.groupby('patientId')
    total = len(grouped)
    
    for i, (patient_id, group) in enumerate(grouped):
        
        # --- REM OTOMATIS 10.000 DATA ---
        if i >= 10000:
            print(f"\n[!] Batas 10.000 data tercapai (dari total {total}). Menghentikan proses untuk menghemat memori.")
            break
        # --------------------------------
        
        dicom_path = os.path.join(DATASET_DIR, f"{patient_id}.dcm")
        
        if not os.path.exists(dicom_path):
            continue
            
        # 1. Baca gambar medis (DICOM) dan ubah ke format pixel (0-255)
        ds = pydicom.dcmread(dicom_path)
        img_array = ds.pixel_array
        img_height, img_width = img_array.shape
        
        # Simpan gambar utuh ke JPG untuk YOLO
        jpg_path = os.path.join(YOLO_IMG_DIR, f"{patient_id}.jpg")
        cv2.imwrite(jpg_path, img_array)
        
        # Buka file TXT untuk label YOLO
        yolo_label_path = os.path.join(YOLO_LBL_DIR, f"{patient_id}.txt")
        with open(yolo_label_path, 'w') as f:
            
            for _, row in group.iterrows():
                target = int(row['Target']) # 0 = Sehat, 1 = Pneumonia
                
                if target == 1:
                    # Ambil koordinat asli
                    x, y, w, h = float(row['x']), float(row['y']), float(row['width']), float(row['height'])
                    
                    # --- PROSES UNTUK YOLO (Normalisasi Koordinat 0-1) ---
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    # Tulis ke file txt (Kelas 0 untuk Pneumonia)
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    
                    # --- PROSES UNTUK CNN (Crop Bounding Box) ---
                    crop_img = img_array[int(y):int(y+h), int(x):int(x+w)]
                    crop_path = os.path.join(CNN_DIR, CNN_CLASSES[1], f"{patient_id}_{int(x)}.jpg")
                    if crop_img.size > 0:
                        cv2.imwrite(crop_path, crop_img)
                        
                else:
                    # Pasien sehat untuk CNN
                    crop_path = os.path.join(CNN_DIR, CNN_CLASSES[0], f"{patient_id}_normal.jpg")
                    crop_img = img_array[200:424, 200:424] 
                    cv2.imwrite(crop_path, crop_img)

        # Tampilkan progress setiap kelipatan 500 agar lebih sering update
        if i % 500 == 0:
            print(f"Proses ekstraksi: {i}/10000 pasien berjalan...")
            
    print("Pre-processing Selesai! Data siap dilatih.")

if __name__ == "__main__":
    process_data()