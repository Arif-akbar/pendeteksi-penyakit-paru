import os
import shutil
from ultralytics import YOLO
import tensorflow as tf

# 1. Buat direktori models jika belum ada
os.makedirs("models", exist_ok=True)

# 2. Download YOLOv8 Nano (model deteksi paling ringan) dan pindahkan
print("Mengunduh model YOLOv8 dummy...")
# Perintah ini akan otomatis mengunduh file yolov8n.pt dari internet
YOLO("yolov8n.pt") 

# Pindahkan dan ganti namanya agar sesuai dengan kode ai_pipeline.py kita
if os.path.exists("yolov8n.pt"):
    shutil.move("yolov8n.pt", "models/yolo_lung.pt")
    print("Berhasil menyiapkan models/yolo_lung.pt")

# 3. Buat model CNN MobileNetV2 kosong (hanya arsitekturnya saja)
print("Membuat model CNN MobileNetV2 dummy...")
# Weights=None agar tidak perlu download bobot imagenet (biar cepat)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(3, activation='softmax')(x) # 3 output untuk Ringan, Sedang, Parah
model_cnn = tf.keras.Model(inputs=base_model.input, outputs=output)

# Simpan ke dalam folder models
model_cnn.save("models/mobilenetv2_severity.h5")
print("Berhasil menyiapkan models/mobilenetv2_severity.h5")

print("\nSELESAI! Semua model dummy sudah siap. Silakan jalankan 'python app.py'")