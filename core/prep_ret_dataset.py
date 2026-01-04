#!/usr/bin/env python3
"""
準備 Recognition Dataset
1. 載入 data/comic_benchmark/det/annotations.txt
2. 根據 bbox 裁切圖片存到 data/comic_benchmark/recognition/images/
3. 將每個裁切圖 path 與 txt，存成 annotations.txt 放到 data/comic_benchmark/recognition/
4. 切成 train.txt 與 val.txt
"""

import json
import os
from pathlib import Path
from PIL import Image
import random


def crop_text_regions(det_annotation_file, output_dir, det_images_dir):
    """
    根據 detection annotations 裁切文字區域

    Args:
        det_annotation_file: detection annotations 文件路徑
        output_dir: 裁切圖片輸出目錄
        det_images_dir: detection 原始圖片目錄

    Returns:
        list: 裁切圖片的 annotations [(image_path, text), ...]
    """
    output_images_dir = Path(output_dir) / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    annotations = []
    crop_counter = 0

    # 讀取 detection annotations
    with open(det_annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 解析格式: image_path\t[{...}, {...}]
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue

            image_rel_path, json_str = parts
            image_path = Path(det_images_dir) / image_rel_path

            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue

            # 解析 JSON annotations
            try:
                regions = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON for {image_path}: {e}")
                continue

            # 開啟原始圖片
            try:
                img = Image.open(image_path)
            except Exception as e:
                print(f"Warning: Failed to open image {image_path}: {e}")
                continue

            # 處理每個文字區域
            for region in regions:
                text = region.get('transcription', '')
                points = region.get('points', [])

                if not text or not points or len(points) < 4:
                    continue

                # 計算 bounding box
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # 裁切圖片
                try:
                    cropped = img.crop((x_min, y_min, x_max, y_max))

                    # 生成裁切圖片檔名
                    crop_filename = f"crop_{crop_counter:06d}.jpg"
                    crop_path = output_images_dir / crop_filename

                    # 儲存裁切圖片
                    cropped.save(crop_path)

                    # 記錄 annotation (相對路徑)
                    rel_crop_path = f"images/{crop_filename}"
                    annotations.append((rel_crop_path, text))

                    crop_counter += 1

                except Exception as e:
                    print(f"Warning: Failed to crop region from {image_path}: {e}")
                    continue

            img.close()

    print(f"Total cropped images: {crop_counter}")
    return annotations


def save_annotations(annotations, output_file):
    """
    儲存 annotations 到文件

    Args:
        annotations: [(image_path, text), ...]
        output_file: 輸出文件路徑
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_path, text in annotations:
            f.write(f"{image_path}\t{text}\n")
    print(f"Saved annotations to {output_file}")


def split_train_val(annotations, train_ratio=0.8, seed=42):
    """
    切分訓練集和驗證集

    Args:
        annotations: [(image_path, text), ...]
        train_ratio: 訓練集比例
        seed: 隨機種子

    Returns:
        tuple: (train_annotations, val_annotations)
    """
    random.seed(seed)
    shuffled = annotations.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_annotations = shuffled[:split_idx]
    val_annotations = shuffled[split_idx:]

    return train_annotations, val_annotations


def main():
    # 設定路徑
    det_annotation_file = "data/comic_benchmark/det/annotations.txt"
    det_images_dir = "data/comic_benchmark/det"
    output_dir = "data/comic_benchmark/rec"

    # 1. 裁切圖片並生成 annotations
    print("Cropping text regions...")
    annotations = crop_text_regions(det_annotation_file, output_dir, det_images_dir)

    if not annotations:
        print("Error: No annotations generated!")
        return

    # 2. 儲存完整的 annotations.txt
    annotations_file = os.path.join(output_dir, "annotations.txt")
    save_annotations(annotations, annotations_file)

    # 3. 切分訓練集和驗證集
    print("\nSplitting train/val datasets...")
    train_annotations, val_annotations = split_train_val(annotations, train_ratio=0.8)

    # 4. 儲存 train.txt 和 val.txt
    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "val.txt")

    save_annotations(train_annotations, train_file)
    save_annotations(val_annotations, val_file)

    print(f"\nDataset preparation completed!")
    print(f"  Total samples: {len(annotations)}")
    print(f"  Train samples: {len(train_annotations)}")
    print(f"  Val samples: {len(val_annotations)}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
