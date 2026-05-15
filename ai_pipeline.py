import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from transformers import pipeline

class AIProcessor:
    def __init__(self):
        # 1. Load semua model saat server pertama kali menyala
        # Gunakan path relatif ke folder 'models'
        self.yolo = YOLO("models/yolo_lung.pt")
        self.cnn = load_model("models/mobilenetv2_severity.h5")
        self.nlp = pipeline("text-generation", model="distilgpt2")
        self.classes = ['Ringan', 'Sedang', 'Parah']

    def analyze(self, image_path):
        # 2. Baca gambar asli untuk di-crop nanti
        img = cv2.imread(image_path)
        
        # 3. Jalankan deteksi YOLO
        results = self.yolo(image_path)
        
        detections = []
        for r in results:
            for box in r.boxes:
                # Ambil koordinat [x_min, y_min, x_max, y_max]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.yolo.names[int(box.cls[0])]
                conf = float(box.conf[0])

                # 4. Crop area anomali untuk MobileNetV2
                cropped_roi = img[y1:y2, x1:x2]
                roi_resized = cv2.resize(cropped_roi, (224, 224)) # Sesuai input standar MobileNetV2
                roi_array = np.expand_dims(roi_resized, axis=0) / 255.0
                
                # 5. Prediksi keparahan dengan CNN
                cnn_pred = self.cnn.predict(roi_array)
                severity = self.classes[np.argmax(cnn_pred)]

                detections.append({
                    "temuan": label,
                    "keparahan": severity,
                    "box": [x1, y1, x2, y2],
                    "confidence": round(conf, 2)
                })

        # 6. Generate Teks Laporan
        prompt = f"Berdasarkan X-Ray, terdapat {len(detections)} anomali: {detections}. Buat kesimpulan medis."
        # Untuk pengujian awal, kita bisa bypass NLP berat dengan teks statis/format string 
        # sampai model NLP benar-benar di-finetune
        report = f"Terdeteksi {len(detections)} area anomali pada citra paru-paru. Observasi lebih lanjut disarankan."

        return {"deteksi": detections, "laporan": report}