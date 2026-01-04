#!/usr/bin/env python3
"""
Translation Ground Truth Evaluation

Evaluates translation predictions against ground truth using CER and BLEU metrics.

Metrics:
- CER (Character Error Rate): Measures character-level edit distance (lower is better)
- BLEU (Bilingual Evaluation Understudy): Measures n-gram precision (higher is better)
  - BLEU-1: Unigram precision
  - BLEU-2: Bigram precision
  - BLEU-3: Trigram precision
  - BLEU-4: 4-gram precision (standard BLEU score)

Input:
- Prediction file: results/2/translated.json (list of translation outputs)
- Ground truth directory: data/benchmark/translation_groundtruth/ (one JSON file per image)

Output:
- Evaluation results with CER and BLEU metrics for each bounding box and overall
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import sys
import unicodedata
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from jiwer import cer as jiwer_cer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def normalize_text(text: str) -> str:
    
    if not text:
        return ""
    
    # 1. 將 Unicode 標準化（處理全形半形、重音符號等）
    # NFKC 會將全形英數字轉為半形，這在處理 OCR 結果時非常有用
    text = unicodedata.normalize('NFKC', text)
    
    # 2. 轉小寫（如果是英文的話）
    text = text.lower()
    
    # 3. 去除或統一標點符號 (根據需求，有時會選擇移除所有非文字字元)
    # 範例：移除常見標點符號
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. 去除多餘空格與換行 (把多餘「空格、\n、\t」 壓縮成單個)
    text = " ".join(text.split())
    
    return text


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate between reference and hypothesis.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text

    Returns:
        CER value (lower is better)
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    return jiwer_cer(reference, hypothesis)


def calculate_bleu(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate BLEU scores between reference and hypothesis.

    Args:
        reference: Ground truth text
        hypothesis: Predicted text

    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 and overall BLEU scores
    """
    if not reference or not hypothesis:
        return {
            "bleu": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0
        }

    # Tokenize texts (character-level for Chinese/Japanese)
    # For CJK languages, each character is a token
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)

    # Use smoothing to handle zero n-gram matches
    smooth = SmoothingFunction().method1

    # Calculate BLEU scores with different n-gram weights
    bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # Overall BLEU score (BLEU-4 is standard)
    bleu = bleu4

    return {
        "bleu": bleu,
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4
    }


def match_bounding_boxes(
    pred_boxes: List[Dict],
    gt_boxes: List[Dict]
) -> List[tuple]:
    """
    Match prediction bounding boxes to ground truth based on spatial proximity.

    Args:
        pred_boxes: List of predicted bounding boxes with translated_text
        gt_boxes: List of ground truth bounding boxes with translated_text

    Returns:
        List of (pred_idx, gt_idx) tuples for matched boxes
    """
    def box_center(box):
        """Calculate center point of bounding box"""
        coords = box["box"]
        x = sum(point[0] for point in coords) / len(coords)
        y = sum(point[1] for point in coords) / len(coords)
        return (x, y)

    def distance(p1, p2):
        """Euclidean distance between two points"""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    matches = []
    used_gt = set()

    # For each prediction, find closest ground truth box
    for pred_idx, pred_box in enumerate(pred_boxes):
        pred_center = box_center(pred_box)
        min_dist = float('inf')
        best_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue

            gt_center = box_center(gt_box)
            dist = distance(pred_center, gt_center)

            if dist < min_dist:
                min_dist = dist
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            matches.append((pred_idx, best_gt_idx))
            used_gt.add(best_gt_idx)

    return matches


def evaluate_image(
    pred: Dict,
    gt: Dict
) -> Dict[str, float]:
    """
    Evaluate a single image's translation results.

    Args:
        pred: Predicted translation output
        gt: Ground truth translation output

    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        "cer": 0.0,
        "bleu": 0.0,
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
        "matched_boxes": 0,
        "pred_boxes": len(pred["bounding_boxes"]),
        "gt_boxes": len(gt["bounding_boxes"]),
        "per_box_cer": [],
        "per_box_bleu": []
    }

    # Match bounding boxes
    matches = match_bounding_boxes(
        pred["bounding_boxes"],
        gt["bounding_boxes"]
    )

    total_cer = 0.0
    total_bleu = 0.0
    total_bleu1 = 0.0
    total_bleu2 = 0.0
    total_bleu3 = 0.0
    total_bleu4 = 0.0
    matched_count = 0

    for pred_idx, gt_idx in matches:
        pred_text = pred["bounding_boxes"][pred_idx].get("translated_text", "")
        gt_text = gt["bounding_boxes"][gt_idx].get("translated_text", "")

        # Normalize texts before calculating metrics
        gt_text_normalized = normalize_text(gt_text)
        pred_text_normalized = normalize_text(pred_text)

        # Calculate CER for this pair
        box_cer = calculate_cer(gt_text_normalized, pred_text_normalized)
        total_cer += box_cer

        # Calculate BLEU scores for this pair
        bleu_scores = calculate_bleu(gt_text_normalized, pred_text_normalized)
        total_bleu += bleu_scores["bleu"]
        total_bleu1 += bleu_scores["bleu1"]
        total_bleu2 += bleu_scores["bleu2"]
        total_bleu3 += bleu_scores["bleu3"]
        total_bleu4 += bleu_scores["bleu4"]

        matched_count += 1

        results["per_box_cer"].append({
            "pred_idx": pred_idx,
            "gt_idx": gt_idx,
            "cer": box_cer,
            "pred_text": pred_text,
            "gt_text": gt_text
        })

        results["per_box_bleu"].append({
            "pred_idx": pred_idx,
            "gt_idx": gt_idx,
            "bleu": bleu_scores["bleu"],
            "bleu1": bleu_scores["bleu1"],
            "bleu2": bleu_scores["bleu2"],
            "bleu3": bleu_scores["bleu3"],
            "bleu4": bleu_scores["bleu4"],
            "pred_text": pred_text,
            "gt_text": gt_text
        })

    # Calculate averages
    if matched_count > 0:
        results["cer"] = total_cer / matched_count
        results["bleu"] = total_bleu / matched_count
        results["bleu1"] = total_bleu1 / matched_count
        results["bleu2"] = total_bleu2 / matched_count
        results["bleu3"] = total_bleu3 / matched_count
        results["bleu4"] = total_bleu4 / matched_count
        results["matched_boxes"] = matched_count

    return results


def load_ground_truth(gt_dir: Path, image_filename: str) -> Optional[Dict]:
    """
    Load ground truth for a specific image.

    Args:
        gt_dir: Ground truth directory
        image_filename: Image filename (e.g., "109994537_p0_master1200.jpg")

    Returns:
        Ground truth dictionary or None if not found
    """
    # Extract base filename without extension
    base_name = Path(image_filename).stem
    gt_file = gt_dir / f"{base_name}.json"

    if not gt_file.exists():
        return None

    with open(gt_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_translation(
    pred_file: str,
    gt_dir: str,
    output_file: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate translation predictions against ground truth.

    Args:
        pred_file: Path to prediction JSON file
        gt_dir: Path to ground truth directory
        output_file: Optional path to save results
        verbose: Show per-box CER details

    Returns:
        Dictionary with overall evaluation results
    """
    pred_path = Path(pred_file)
    gt_path = Path(gt_dir)

    # Load predictions
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions: List[Dict] = json.load(f)

    print(f"Loaded {len(predictions)} predictions from {pred_file}")
    print(f"Ground truth directory: {gt_dir}")
    print("-" * 60)

    # Evaluate each image
    image_results = []
    overall_metrics = {
        "total_images": len(predictions),
        "evaluated_images": 0,
        "cer": 0.0,
        "bleu": 0.0,
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
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
        result = evaluate_image(pred, gt)
        result["image"] = image_filename
        image_results.append(result)

        # Update overall metrics
        overall_metrics["evaluated_images"] += 1
        overall_metrics["cer"] += result["cer"]
        overall_metrics["bleu"] += result["bleu"]
        overall_metrics["bleu1"] += result["bleu1"]
        overall_metrics["bleu2"] += result["bleu2"]
        overall_metrics["bleu3"] += result["bleu3"]
        overall_metrics["bleu4"] += result["bleu4"]
        overall_metrics["total_matched"] += result["matched_boxes"]
        overall_metrics["total_pred_boxes"] += result["pred_boxes"]
        overall_metrics["total_gt_boxes"] += result["gt_boxes"]

        print(f"[{overall_metrics['evaluated_images']}/{len(predictions)}] {image_filename}")
        print(f"  CER: {result['cer']:.4f}, BLEU: {result['bleu']:.4f}, Matched: {result['matched_boxes']}/{result['gt_boxes']}")

        if verbose and result["per_box_cer"]:
            print("  Per-box details:")
            for i, box_result in enumerate(result["per_box_cer"]):
                bleu_result = result["per_box_bleu"][i]
                print(f"    Box {box_result['pred_idx']} -> {box_result['gt_idx']}: CER={box_result['cer']:.4f}, BLEU={bleu_result['bleu']:.4f}")
                print(f"      Pred: {box_result['pred_text']}")
                print(f"      GT:   {box_result['gt_text']}")

    # Calculate averages
    if overall_metrics["evaluated_images"] > 0:
        n = overall_metrics["evaluated_images"]
        overall_metrics["cer"] /= n
        overall_metrics["bleu"] /= n
        overall_metrics["bleu1"] /= n
        overall_metrics["bleu2"] /= n
        overall_metrics["bleu3"] /= n
        overall_metrics["bleu4"] /= n

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Images evaluated: {overall_metrics['evaluated_images']}/{overall_metrics['total_images']}")
    print(f"Average CER: {overall_metrics['cer']:.4f}")
    print(f"Average BLEU: {overall_metrics['bleu']:.4f}")
    print(f"  BLEU-1: {overall_metrics['bleu1']:.4f}")
    print(f"  BLEU-2: {overall_metrics['bleu2']:.4f}")
    print(f"  BLEU-3: {overall_metrics['bleu3']:.4f}")
    print(f"  BLEU-4: {overall_metrics['bleu4']:.4f}")
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
        description="Evaluate translation predictions against ground truth using CER and BLEU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  %(prog)s

  # Specify custom paths
  %(prog)s -p results/2/translated.json -g data/benchmark/translation_groundtruth

  # Save results to file
  %(prog)s -o results/2/translation_eval_results.json

  # Show verbose per-box details
  %(prog)s -v
        """
    )

    parser.add_argument(
        "-p", "--pred-file",
        default="results/2/translated.json",
        help="Prediction JSON file (default: results/2/translated.json)"
    )

    parser.add_argument(
        "-g", "--gt-dir",
        default="data/benchmark/translation_groundtruth",
        help="Ground truth directory (default: data/benchmark/translation_groundtruth)"
    )

    parser.add_argument(
        "-o", "--output",
        default="results/eval/translation.json",
        help="Output file to save evaluation results (optional)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-box CER details"
    )

    args = parser.parse_args()

    # Get project root directory
    project_root = Path(__file__).parent.parent.parent

    # Handle relative and absolute paths
    pred_path = Path(args.pred_file)
    if not pred_path.is_absolute():
        pred_path = project_root / pred_path

    gt_path = Path(args.gt_dir)
    if not gt_path.is_absolute():
        gt_path = project_root / gt_path

    output_path = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path = str(output_path)

    # Check if files exist
    if not pred_path.exists():
        print(f"Error: Prediction file not found: {pred_path}")
        sys.exit(1)

    if not gt_path.exists():
        print(f"Error: Ground truth directory not found: {gt_path}")
        sys.exit(1)

    # Run evaluation
    evaluate_translation(
        str(pred_path),
        str(gt_path),
        output_path,
        args.verbose
    )


if __name__ == "__main__":
    main()
