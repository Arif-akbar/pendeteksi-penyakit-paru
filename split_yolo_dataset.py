from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitSummary:
    output_dir: Path
    train_count: int
    val_count: int
    test_count: int


def collect_pairs(source_dir: Path):
    image_dir = source_dir / "images" / "train"
    label_dir = source_dir / "labels" / "train"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            "Dataset YOLO sumber harus punya images/train dan labels/train."
        )

    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError("Tidak ada pasangan image/label YOLO yang ditemukan.")

    return pairs


def split_pairs(pairs, val_ratio: float, test_ratio: float, seed: int):
    if val_ratio <= 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Rasio split tidak valid. Pastikan train masih lebih dari 0%.")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)

    test_items = shuffled[:test_count]
    val_items = shuffled[test_count:test_count + val_count]
    train_items = shuffled[test_count + val_count:]

    return train_items, val_items, test_items


def copy_items(items, output_dir: Path, split_name: str):
    image_out = output_dir / "images" / split_name
    label_out = output_dir / "labels" / split_name
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in items:
        shutil.copy2(image_path, image_out / image_path.name)
        shutil.copy2(label_path, label_out / label_path.name)


def has_ready_split(output_dir: Path):
    return all(
        (output_dir / kind / split).exists()
        for kind in ("images", "labels")
        for split in ("train", "val", "test")
    )


def ensure_split(
    source_dir: str | Path = "dataset_yolo",
    output_dir: str | Path = "dataset_yolo_split",
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
    force: bool = False,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if has_ready_split(output_dir) and not force:
        return SplitSummary(output_dir, -1, -1, -1)

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    pairs = collect_pairs(source_dir)
    train_items, val_items, test_items = split_pairs(pairs, val_ratio, test_ratio, seed)

    copy_items(train_items, output_dir, "train")
    copy_items(val_items, output_dir, "val")
    if test_items:
        copy_items(test_items, output_dir, "test")

    return SplitSummary(
        output_dir=output_dir,
        train_count=len(train_items),
        val_count=len(val_items),
        test_count=len(test_items),
    )


def main():
    parser = argparse.ArgumentParser(description="Buat split train/val/test untuk dataset YOLO.")
    parser.add_argument("--source", default="dataset_yolo", help="Folder dataset YOLO sumber.")
    parser.add_argument("--output", default="dataset_yolo_split", help="Folder output split.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Porsi validasi.")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Porsi test.")
    parser.add_argument("--seed", type=int, default=42, help="Seed acak agar split konsisten.")
    parser.add_argument("--force", action="store_true", help="Buat ulang folder output.")
    args = parser.parse_args()

    summary = ensure_split(
        source_dir=args.source,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force=args.force,
    )

    if summary.train_count == -1:
        print(f"Split sudah tersedia di {summary.output_dir}. Gunakan --force untuk membuat ulang.")
        return

    print(f"Split selesai di {summary.output_dir}")
    print(f"train: {summary.train_count}")
    print(f"val  : {summary.val_count}")
    print(f"test : {summary.test_count}")


if __name__ == "__main__":
    main()
