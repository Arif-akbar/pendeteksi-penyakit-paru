import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Siapkan bobot awal YOLO jika belum ada.")
    parser.add_argument("--force", action="store_true", help="Timpa models/yolo_lung.pt jika sudah ada.")
    args = parser.parse_args()

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    output_model = model_dir / "yolo_lung.pt"

    if output_model.exists() and not args.force:
        print(f"{output_model} sudah ada. Tidak ditimpa.")
        print("Gunakan --force hanya jika benar-benar ingin mengganti model aktif.")
        return

    print("Menyiapkan bobot awal YOLOv8 Nano...")
    YOLO("yolov8n.pt")

    source_model = Path("yolov8n.pt")
    if not source_model.exists():
        raise FileNotFoundError("yolov8n.pt tidak ditemukan setelah proses setup.")

    shutil.copy2(source_model, output_model)
    print(f"Model awal tersimpan di {output_model}")
    print("Lanjutkan dengan: python train_yolo.py --epochs 50 --batch 8")


if __name__ == "__main__":
    main()
