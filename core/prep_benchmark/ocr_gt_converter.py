#!/usr/bin/env python3
"""
OCR Ground Truth Converter

Converts OCR detection annotations from the comic benchmark format to OcrOutputSchema format.

Input: data/comic_benchmark/det/annotations.txt
Output: data/benchmark/ocr_groundtruth/*.json
"""

import json
import os
from pathlib import Path
from typing import List
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from lib.schema import OcrOutputSchema, BoundingBox


def parse_annotation_line(line: str) -> OcrOutputSchema:
    """
    Parse a single line from annotations.txt and convert to OcrOutputSchema.

    Input format: image_path\t[{"transcription": "text", "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}, ...]
    Output format: OcrOutputSchema with source, text, image_path, and bounding_boxes
    """
    parts = line.strip().split('\t')
    if len(parts) != 2:
        raise ValueError(f"Invalid line format: {line}")

    image_path = parts[0]
    annotations = json.loads(parts[1])

    # Convert annotations to BoundingBox format
    bounding_boxes: List[BoundingBox] = []
    all_text_lines: List[str] = []

    for ann in annotations:
        bounding_box: BoundingBox = {
            "box": ann["points"],
            "text": ann["transcription"],
            "score": None  # Ground truth doesn't have confidence scores
        }
        bounding_boxes.append(bounding_box)
        all_text_lines.append(ann["transcription"])

    # Create OcrOutputSchema
    ocr_output: OcrOutputSchema = {
        "source": image_path,
        "text": "\n".join(all_text_lines),  # All recognized text separated by newlines
        "image_path": image_path,
        "bounding_boxes": bounding_boxes
    }

    return ocr_output


def convert_annotations(input_file: str, output_dir: str) -> None:
    """
    Convert all annotations from input file to OcrOutputSchema JSON files.

    Args:
        input_file: Path to annotations.txt
        output_dir: Directory to save output JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read and process each line
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} annotations...")

    for i, line in enumerate(lines):
        if not line.strip():
            continue

        try:
            ocr_output = parse_annotation_line(line)

            # Extract filename from image_path (e.g., "images/109994537_p0_master1200.jpg" -> "109994537_p0_master1200")
            image_filename = Path(ocr_output["image_path"]).stem

            # Save as JSON file
            output_file = os.path.join(output_dir, f"{image_filename}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ocr_output, f, ensure_ascii=False, indent=2)

            print(f"[{i+1}/{len(lines)}] Converted: {image_filename}.json")

        except Exception as e:
            print(f"Error processing line {i+1}: {e}")
            continue

    print(f"\nConversion complete! Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert OCR detection annotations to OcrOutputSchema format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  %(prog)s

  # Specify custom input and output
  %(prog)s -i data/custom/annotations.txt -o data/output/groundtruth

  # Use absolute paths
  %(prog)s -i /path/to/annotations.txt -o /path/to/output
        """
    )

    parser.add_argument(
        "-i", "--input",
        default="data/comic_benchmark/det/annotations.txt",
        help="Input annotations file (default: data/comic_benchmark/det/annotations.txt)"
    )

    parser.add_argument(
        "-o", "--output",
        default="data/benchmark/ocr_groundtruth",
        help="Output directory for ground truth JSON files (default: data/benchmark/ocr_groundtruth)"
    )

    args = parser.parse_args()

    # Get project root directory
    project_root = Path(__file__).parent.parent

    # Handle relative and absolute paths
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Input file: {input_path}")
    print(f"Output directory: {output_path}")
    print("-" * 60)

    convert_annotations(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
