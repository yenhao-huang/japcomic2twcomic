#!/usr/bin/env python3
"""
OCR Ground Truth Evaluation

Evaluates OCR predictions against ground truth using:
1. Character Error Rate (CER) for text accuracy
2. Mean Intersection over Union (IoU) for bounding box accuracy

Input:
- Prediction file: results/2/ocr.json (OcrOutputSchema list)
- Ground truth directory: data/benchmark/ocr_groundtruth/ (OcrOutputSchema per image)

Output:
- Evaluation results with CER and IoU metrics
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Add parent directory to path for imports
# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from lib.schema import OcrOutputSchema, BoundingBox
from jiwer import cer as jiwer_cer
from shapely.geometry import Polygon
    

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate between reference and hypothesis.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text

    Returns:
        CER value (lower is better)
    """

    return jiwer_cer(reference, hypothesis)


def calculate_iou(box1: List[List[float]], box2: List[List[float]]) -> float:
    """
    Calculate Intersection over Union (IoU) for two quadrilateral bounding boxes.

    Args:
        box1: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        box2: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        IoU value between 0 and 1
    """
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    if union == 0:
        return 0.0

    return intersection / union


def find_best_matching_boxes(
    pred_boxes: List[BoundingBox],
    gt_boxes: List[BoundingBox],
    iou_threshold: float = 0.5
) -> List[Tuple[Optional[int], Optional[int], float]]:
    """
    Find best matching bounding boxes between prediction and ground truth.

    Args:
        pred_boxes: Predicted bounding boxes
        gt_boxes: Ground truth bounding boxes
        iou_threshold: Minimum IoU to consider as a match

    Returns:
        List of (pred_idx, gt_idx, iou) tuples for matched boxes
    """
    matches = []
    used_gt = set()

    # For each prediction, find best matching ground truth
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue

            iou = calculate_iou(pred_box["box"], gt_box["box"])

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            matches.append((pred_idx, best_gt_idx, best_iou))
            used_gt.add(best_gt_idx)
        else:
            # No match found for this prediction (false positive)
            matches.append((pred_idx, None, 0.0))

    # Add unmatched ground truth boxes (false negatives)
    for gt_idx in range(len(gt_boxes)):
        if gt_idx not in used_gt:
            matches.append((None, gt_idx, 0.0))

    return matches


def evaluate_image(
    pred: OcrOutputSchema,
    gt: OcrOutputSchema,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a single image's OCR results.

    Args:
        pred: Predicted OCR output
        gt: Ground truth OCR output
        iou_threshold: Minimum IoU for box matching

    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        "cer": 0.0,
        "miou": 0.0,
        "precision": 0.0,
        "matched_boxes": 0,
        "pred_boxes": len(pred["bounding_boxes"]),
        "gt_boxes": len(gt["bounding_boxes"])
    }

    # Calculate CER for matched bounding boxes
    matches = find_best_matching_boxes(
        pred["bounding_boxes"],
        gt["bounding_boxes"],
        iou_threshold
    )

    matched_cer_sum = 0.0
    matched_iou_sum = 0.0
    matched_count = 0

    for pred_idx, gt_idx, iou in matches:
        if pred_idx is not None and gt_idx is not None:
            # This is a matched box
            pred_text = pred["bounding_boxes"][pred_idx]["text"]
            gt_text = gt["bounding_boxes"][gt_idx]["text"]

            # Calculate CER for this pair
            if gt_text:  # Avoid division by zero
                box_cer = calculate_cer(gt_text, pred_text)
                matched_cer_sum += box_cer

            matched_iou_sum += iou
            matched_count += 1

    # Calculate metrics
    if matched_count > 0:
        results["cer"] = matched_cer_sum / matched_count
        results["miou"] = matched_iou_sum / matched_count
        results["matched_boxes"] = matched_count

    # Calculate precision
    if results["pred_boxes"] > 0:
        results["precision"] = matched_count / results["pred_boxes"]

    return results


def load_ground_truth(gt_dir: Path, image_filename: str) -> Optional[OcrOutputSchema]:
    """
    Load ground truth for a specific image.

    Args:
        gt_dir: Ground truth directory
        image_filename: Image filename (e.g., "109994537_p0_master1200.jpg")

    Returns:
        Ground truth OcrOutputSchema or None if not found
    """
    # Extract base filename without extension
    base_name = Path(image_filename).stem
    gt_file = gt_dir / f"{base_name}.json"

    if not gt_file.exists():
        return None

    with open(gt_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_ocr(
    pred_file: str,
    gt_dir: str,
    iou_threshold: float = 0.5,
    output_file: Optional[str] = None
) -> Dict:
    """
    Evaluate OCR predictions against ground truth.

    Args:
        pred_file: Path to prediction JSON file (list of OcrOutputSchema)
        gt_dir: Path to ground truth directory
        iou_threshold: Minimum IoU for box matching
        output_file: Optional path to save results

    Returns:
        Dictionary with overall evaluation results
    """
    pred_path = Path(pred_file)
    gt_path = Path(gt_dir)

    # Load predictions
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions: List[OcrOutputSchema] = json.load(f)

    print(f"Loaded {len(predictions)} predictions from {pred_file}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"IoU threshold: {iou_threshold}")
    print("-" * 60)

    # Evaluate each image
    image_results = []
    overall_metrics = {
        "total_images": len(predictions),
        "evaluated_images": 0,
        "cer": 0.0,
        "miou": 0.0,
        "precision": 0.0,
        "total_matched": 0,
        "total_pred_boxes": 0,
        "total_gt_boxes": 0
    }

    for pred in predictions:
        # Get image filename from path
        image_filename = Path(pred["image_path"]).name

        # Load corresponding ground truth
        gt = load_ground_truth(gt_path, image_filename)

        if gt is None:
            print(f"Warning: No ground truth found for {image_filename}")
            continue

        # Evaluate this image
        result = evaluate_image(pred, gt, iou_threshold)
        result["image"] = image_filename
        image_results.append(result)

        # Update overall metrics
        overall_metrics["evaluated_images"] += 1
        overall_metrics["cer"] += result["cer"]
        overall_metrics["miou"] += result["miou"]
        overall_metrics["precision"] += result["precision"]
        overall_metrics["total_matched"] += result["matched_boxes"]
        overall_metrics["total_pred_boxes"] += result["pred_boxes"]
        overall_metrics["total_gt_boxes"] += result["gt_boxes"]

        print(f"[{overall_metrics['evaluated_images']}/{len(predictions)}] {image_filename}")
        print(f"  CER: {result['cer']:.4f}, mIoU: {result['miou']:.4f}, "
              f"Precision: {result['precision']:.4f}, Matched: {result['matched_boxes']}/{result['gt_boxes']}")

    # Calculate averages
    if overall_metrics["evaluated_images"] > 0:
        n = overall_metrics["evaluated_images"]
        overall_metrics["cer"] /= n
        overall_metrics["miou"] /= n
        overall_metrics["precision"] /= n

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Images evaluated: {overall_metrics['evaluated_images']}/{overall_metrics['total_images']}")
    print(f"CER: {overall_metrics['cer']:.4f}")
    print(f"mIoU: {overall_metrics['miou']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Total matched boxes: {overall_metrics['total_matched']}/{overall_metrics['total_gt_boxes']}")

    # Prepare output
    output = {
        "overall": overall_metrics,
        "per_image": image_results
    }

    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {output_file}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR predictions against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  %(prog)s

  # Specify custom paths
  %(prog)s -p results/2/ocr.json -g data/benchmark/ocr_groundtruth

  # Save results to file
  %(prog)s -o results/2/eval_results.json

  # Adjust IoU threshold
  %(prog)s --iou-threshold 0.7
        """
    )

    parser.add_argument(
        "-p", "--pred-file",
        default="results/2/ocr.json",
        help="Prediction JSON file (default: results/2/ocr.json)"
    )

    parser.add_argument(
        "-g", "--gt-dir",
        default="data/benchmark/ocr_groundtruth",
        help="Ground truth directory (default: data/benchmark/ocr_groundtruth)"
    )

    parser.add_argument(
        "-o", "--output",
        default="results/eval/ocr.json",
        help="Output file to save evaluation results (optional)"
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for box matching (default: 0.5)"
    )

    args = parser.parse_args()

    # Handle relative and absolute paths
    pred_path = Path(args.pred_file)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / pred_path

    gt_path = Path(args.gt_dir)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path

    output_path = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path = str(output_path)

    # Check if files exist
    if not pred_path.exists():
        print(f"Error: Prediction file not found: {pred_path}")
        sys.exit(1)

    if not gt_path.exists():
        print(f"Error: Ground truth directory not found: {gt_path}")
        sys.exit(1)

    # Run evaluation
    evaluate_ocr(
        str(pred_path),
        str(gt_path),
        args.iou_threshold,
        output_path
    )


if __name__ == "__main__":
    main()
