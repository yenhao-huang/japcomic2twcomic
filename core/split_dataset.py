#!/usr/bin/env python3
"""
Split OCR Detection Dataset into Train and Eval sets

Usage:
    python split_dataset.py --input annotations.txt --output_dir data/ocr_det_dataset --ratio 0.9

This will split the dataset with 90% for training and 10% for evaluation.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def load_annotations(input_file: Path) -> List[Tuple[str, str]]:
    """Load annotations from input file

    Returns:
        List of (image_path, annotations_json) tuples
    """
    annotations = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t', 1)
            if len(parts) == 2:
                img_path, boxes_json = parts
                annotations.append((img_path, boxes_json))

    return annotations


def split_dataset(annotations: List[Tuple[str, str]], ratio: float, seed: int = 42) -> Tuple[List, List]:
    """Split annotations into train and eval sets

    Args:
        annotations: List of annotation tuples
        ratio: Ratio for train set (e.g., 0.9 for 90% train, 10% eval)
        seed: Random seed for reproducibility

    Returns:
        (train_annotations, eval_annotations)
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Shuffle annotations
    shuffled = annotations.copy()
    random.shuffle(shuffled)

    # Calculate split point
    split_idx = int(len(shuffled) * ratio)

    train_set = shuffled[:split_idx]
    eval_set = shuffled[split_idx:]

    return train_set, eval_set


def save_annotations(annotations: List[Tuple[str, str]], output_file: Path):
    """Save annotations to file"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for img_path, boxes_json in annotations:
            f.write(f"{img_path}\t{boxes_json}\n")


def main():
    parser = argparse.ArgumentParser(description='Split OCR Detection Dataset')
    parser.add_argument('--input', type=str, required=True,
                      help='Input annotations file')
    parser.add_argument('--output_dir', type=str, default='data/ocr_det_dataset',
                      help='Output directory for train.txt and val.txt')
    parser.add_argument('--ratio', type=float, default=0.9,
                      help='Train set ratio (default: 0.9 for 90%% train, 10%% eval)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not 0 < args.ratio < 1:
        raise ValueError(f"Ratio must be between 0 and 1, got {args.ratio}")

    # Load annotations
    print(f"Loading annotations from {input_file}...")
    annotations = load_annotations(input_file)
    print(f"Loaded {len(annotations)} annotated images")

    if len(annotations) == 0:
        raise ValueError("No annotations found in input file")

    # Split dataset
    print(f"Splitting dataset with ratio {args.ratio:.1%} train / {1-args.ratio:.1%} eval...")
    train_set, eval_set = split_dataset(annotations, args.ratio, args.seed)

    print(f"Train set: {len(train_set)} images")
    print(f"Eval set: {len(eval_set)} images")

    # Save splits
    train_file = output_dir / 'train.txt'
    eval_file = output_dir / 'val.txt'

    print(f"Saving train set to {train_file}...")
    save_annotations(train_set, train_file)

    print(f"Saving eval set to {eval_file}...")
    save_annotations(eval_set, eval_file)

    print("Done!")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total images: {len(annotations)}")
    print(f"Train images: {len(train_set)} ({len(train_set)/len(annotations)*100:.1f}%)")
    print(f"Eval images: {len(eval_set)} ({len(eval_set)/len(annotations)*100:.1f}%)")


if __name__ == '__main__':
    main()
