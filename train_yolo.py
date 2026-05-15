import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from split_yolo_dataset import ensure_split

def main():
    parser = argparse.ArgumentParser(description="Training ulang YOLO untuk deteksi pneumonia.")
    parser.add_argument("--data", default="dataset.yaml", help="Path file konfigurasi dataset YOLO.")
    parser.add_argument("--source-dataset", default="dataset_yolo", help="Dataset mentah hasil prepare_dataset.py.")
    parser.add_argument("--prepared-dataset", default="dataset_yolo_split", help="Dataset hasil split.")
    parser.add_argument("--model", default="models/yolo_lung.pt", help="Bobot awal YOLO.")
    parser.add_argument("--epochs", type=int, default=50, help="Jumlah epoch training.")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran input image.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Worker dataloader. 0 lebih aman di Windows.")
    parser.add_argument("--name", default="paru_model_refined", help="Nama run di runs/detect.")
    parser.add_argument("--force-split", action="store_true", help="Buat ulang dataset_yolo_split.")
    parser.add_argument("--no-export", action="store_true", help="Jangan salin best.pt ke models/yolo_lung.pt.")
    args = parser.parse_args()

    print("Memastikan dataset punya split train/val/test...")
    split_summary = ensure_split(
        source_dir=args.source_dataset,
        output_dir=args.prepared_dataset,
        force=args.force_split,
    )
    if split_summary.train_count == -1:
        print(f"Split dataset sudah tersedia di {split_summary.output_dir}")
    else:
        print(
            f"Split dibuat: train={split_summary.train_count}, "
            f"val={split_summary.val_count}, test={split_summary.test_count}"
        )

    print(f"Memuat bobot awal: {args.model}")
    model = YOLO(args.model)

    print("Memulai proses training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        name=args.name,
        exist_ok=True,
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training selesai. Model terbaik: {best_model}")

    if not args.no_export and best_model.exists():
        output_model = Path("models") / "yolo_lung.pt"
        output_model.parent.mkdir(exist_ok=True)
        shutil.copy2(best_model, output_model)
        print(f"Model aktif diperbarui: {output_model}")

if __name__ == '__main__':
    main()
