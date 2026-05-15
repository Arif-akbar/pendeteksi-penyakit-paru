import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pydicom
except ImportError:  # DICOM tetap opsional sampai dependency dipasang.
    pydicom = None

class AIProcessor:
    def __init__(self):
        self.model_path = Path(os.getenv("PULMO_YOLO_MODEL", "models/yolo_lung.pt"))
        self.conf_threshold = float(os.getenv("PULMO_YOLO_CONF", "0.05"))
        self.iou_threshold = float(os.getenv("PULMO_YOLO_IOU", "0.45"))

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model YOLO tidak ditemukan: {self.model_path}")

        self.yolo = YOLO(str(self.model_path))

    def analyze(self, image_path):
        img = self._read_image(image_path)
        image_h, image_w = img.shape[:2]

        results = self.yolo(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = self._clip_box(box.xyxy[0], image_w, image_h)
                if x2 <= x1 or y2 <= y1:
                    continue

                class_id = int(box.cls[0])
                label = self.yolo.names.get(class_id, f"Class {class_id}")
                conf = float(box.conf[0])
                severity, severity_score = self._estimate_severity(
                    box_coords=(x1, y1, x2, y2),
                    image_size=(image_w, image_h),
                    confidence=conf,
                )

                detections.append({
                    "temuan": label,
                    "keparahan": severity,
                    "skor_keparahan": round(severity_score, 2),
                    "box": [x1, y1, x2, y2],
                    "confidence": round(conf, 2)
                })

        report = self._build_report(detections)

        return {
            "deteksi": detections,
            "laporan": report,
            "meta": {
                "model": str(self.model_path),
                "confidence_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "image_width": image_w,
                "image_height": image_h,
            }
        }

    def _read_image(self, image_path):
        path = Path(image_path)
        suffix = path.suffix.lower()

        if suffix in {".dcm", ".dicom"}:
            return self._read_dicom(path)

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("File tidak dapat dibaca sebagai gambar. Gunakan JPG, PNG, atau DICOM yang valid.")
        return img

    def _read_dicom(self, path):
        if pydicom is None:
            raise RuntimeError("DICOM membutuhkan dependency pydicom. Jalankan: pip install pydicom")

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        arr = np.nan_to_num(arr)

        min_value = float(np.min(arr))
        max_value = float(np.max(arr))
        if max_value > min_value:
            arr = (arr - min_value) / (max_value - min_value)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)

        if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
            arr = 255 - arr

        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    def _clip_box(self, coords, image_w, image_h):
        x1, y1, x2, y2 = map(int, coords)
        return (
            max(0, min(x1, image_w - 1)),
            max(0, min(y1, image_h - 1)),
            max(0, min(x2, image_w)),
            max(0, min(y2, image_h)),
        )

    def _estimate_severity(self, box_coords, image_size, confidence):
        x1, y1, x2, y2 = box_coords
        image_w, image_h = image_size
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, image_w * image_h)

        # Estimasi awal sampai tersedia dataset severity berlabel dokter.
        severity_score = min(1.0, (area_ratio * 4.0) + (confidence * 0.35))

        if severity_score >= 0.65:
            return "Parah", severity_score
        if severity_score >= 0.35:
            return "Sedang", severity_score
        return "Ringan", severity_score

    def _build_report(self, detections):
        if not detections:
            return (
                "Tidak ada area anomali yang melewati ambang deteksi saat ini. "
                "Jika citra tetap dicurigai bermasalah, coba turunkan PULMO_YOLO_CONF "
                "atau latih ulang model dengan split data validasi yang benar."
            )

        max_conf = max(item["confidence"] for item in detections)
        severe_count = sum(1 for item in detections if item["keparahan"] == "Parah")
        medium_count = sum(1 for item in detections if item["keparahan"] == "Sedang")

        return (
            f"Terdeteksi {len(detections)} area anomali pada citra paru-paru. "
            f"Confidence tertinggi {round(max_conf * 100)}%. "
            f"Ringkasan keparahan: {severe_count} parah, {medium_count} sedang, "
            f"{len(detections) - severe_count - medium_count} ringan. "
            "Hasil ini adalah bantuan observasi AI dan tetap perlu ditinjau tenaga medis."
        )
