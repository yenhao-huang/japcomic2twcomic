"""
Docling Layout Detection Parser Script

This script processes images from core/layout/data/, uses the Docling model
from Hugging Face transformers to perform layout detection, and outputs the results.

Usage:
    # Basic usage - process all images in data directory
    python core/layout/docling_parse.py

    # Process specific image
    python core/layout/docling_parse.py --image-path core/layout/data/table.png

    # Save visualization with bounding boxes
    python core/layout/docling_parse.py --save-vis

    # Custom output directory
    python core/layout/docling_parse.py --output-dir results/docling --save-vis
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForObjectDetection, AutoImageProcessor


class DoclingParser:
    """Parse layout detection using Docling models"""

    def __init__(
        self,
        model_name: str,
        output_dir: str = "results/layout/docling",
        save_visualization: bool = False,
        device: str = None
    ):
        """
        Initialize Docling Parser

        Args:
            model_name: Hugging Face model name
            output_dir: Directory to save results
            save_visualization: Whether to save bbox visualization images
            device: Device to run inference on (cpu/cuda)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_vis = save_visualization
        if self.save_vis:
            self.vis_dir = self.output_dir / "visualizations"
            self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing Docling model: {model_name}")
        print(f"Using device: {self.device}")

        # Load model and processor
        self._load_model()

    def _load_model(self):
        """Load Docling model and processor from Hugging Face"""
        try:
            print("Loading model from Hugging Face...")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Get label mapping
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

            print(f"Model loaded successfully!")
            print(f"Detected {len(self.id2label)} categories: {list(self.id2label.values())}")

        except Exception as e2:
            print(f"Failed to load model with alternative method: {e2}")
            raise

    def detect_layout(self, image_path: Path, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform layout detection on an image

        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score to keep detections

        Returns:
            List of detection results with bounding boxes and labels
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=confidence_threshold
        )[0]

        # Convert to detection format
        detections = []
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.cpu().numpy()
            label = self.id2label[label_id.item()]

            detection = {
                "bbox": box.tolist(),  # [x_min, y_min, x_max, y_max]
                "label": label,
                "score": score.item(),
                "category_id": label_id.item()
            }
            detections.append(detection)

        return detections

    def _draw_bbox_on_image(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            image: Input image (numpy array)
            detections: List of detection results
            thickness: Thickness of bbox lines

        Returns:
            Image with drawn bboxes
        """
        img_draw = image.copy()

        # Generate colors for each category
        np.random.seed(42)
        colors = {}
        for label in self.id2label.values():
            colors[label] = tuple(np.random.randint(50, 255, 3).tolist())

        for det in detections:
            bbox = det["bbox"]  # [x_min, y_min, x_max, y_max]
            label = det["label"]
            score = det["score"]

            # Convert to integer coordinates
            x_min, y_min, x_max, y_max = map(int, bbox)

            # Get color for this category
            color = colors.get(label, (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(img_draw, (x_min, y_min), (x_max, y_max), color, thickness)

            # Draw label with background
            label_text = f"{label}: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                img_draw,
                (x_min, y_min - text_height - 5),
                (x_min + text_width, y_min),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                img_draw,
                label_text,
                (x_min, y_min - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

        return img_draw

    def save_visualization(self, image_path: Path, detections: List[Dict[str, Any]]):
        """
        Save visualization image with bboxes drawn

        Args:
            image_path: Path to the original image
            detections: List of detection results
        """
        if not self.save_vis:
            return

        try:
            # Read image with OpenCV
            img = cv2.imread(str(image_path))

            if img is None:
                print(f"  Warning: Could not read image: {image_path}")
                return

            # Draw bboxes
            img_with_bbox = self._draw_bbox_on_image(img, detections)

            # Save visualization
            output_path = self.vis_dir / f"{image_path.stem}_docling.jpg"
            cv2.imwrite(str(output_path), img_with_bbox)
            print(f"  Saved visualization: {output_path}")

        except Exception as e:
            print(f"  Warning: Failed to save visualization: {e}")

    def process_single_image(
        self,
        image_path: Path,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Process a single image

        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score

        Returns:
            List of detection results
        """
        print(f"Processing: {image_path}")

        try:
            # Perform layout detection
            detections = self.detect_layout(image_path, confidence_threshold)

            print(f"  Found {len(detections)} detections")

            # Print detection summary
            label_counts = {}
            for det in detections:
                label = det["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

            for label, count in sorted(label_counts.items()):
                print(f"    {label}: {count}")

            # Save visualization if enabled
            if self.save_vis:
                self.save_visualization(image_path, detections)

            return detections

        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def process_all_images(
        self,
        data_dir: str = "core/layout/data",
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Process all images in the data directory

        Args:
            data_dir: Directory containing image files
            confidence_threshold: Minimum confidence score

        Returns:
            Dictionary mapping image paths to their detection results
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        # Find all image files recursively
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(data_path.glob(f"**/*{ext}"))
            image_files.extend(data_path.glob(f"**/*{ext.upper()}"))

        image_files = sorted(set(image_files))

        if not image_files:
            print(f"No image files found in {data_dir}")
            return {}

        print(f"Found {len(image_files)} image files to process\n")

        all_results = {}
        success_count = 0
        failed_count = 0

        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] ", end="")

            detections = self.process_single_image(image_file, confidence_threshold)

            if detections:
                all_results[str(image_file)] = detections
                success_count += 1
            else:
                failed_count += 1

            print()  # Empty line between files

        print(f"\nProcessing complete:")
        print(f"  Success: {success_count}/{len(image_files)}")
        print(f"  Failed: {failed_count}/{len(image_files)}")
        print(f"  Total images with detections: {len(all_results)}")

        return all_results

    def save_results(self, results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Save detection results to JSON file

        Args:
            results: Dictionary mapping image paths to detection results

        Returns:
            Path to the saved JSON file
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output filename
        output_file = self.output_dir / f"docling_results_{timestamp}.json"

        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")
        return str(output_file)


def main():
    """Main function to run Docling layout parsing"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Parse layout detection using Docling models"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="/tmp2/share_data/docling-layout-heron",
        help="Hugging Face model name (default: ds4sd/DocLayNet-base)"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Process a specific image (default: process all images in data-dir)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/pdf2png",
        help="Directory containing image files (default: data/pdf2png)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/layout/docling",
        help="Output directory to save results (default: results/layout/docling)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualization images with bboxes drawn"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cpu/cuda, default: auto-detect)"
    )

    args = parser.parse_args()

    print("Docling Layout Detection Parser")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Confidence threshold: {args.confidence}")
    print(f"  Save visualization: {args.save_vis}")
    print(f"  Device: {args.device or 'auto-detect'}")
    print()

    # Initialize parser
    parser_instance = DoclingParser(
        model_name=args.model_name,
        output_dir=args.output_dir,
        save_visualization=args.save_vis,
        device=args.device
    )

    # Process images
    if args.image_path:
        # Process single image
        image_path = Path(args.image_path)
        detections = parser_instance.process_single_image(image_path, args.confidence)
        results = {str(image_path): detections} if detections else {}
    else:
        # Process all images
        results = parser_instance.process_all_images(args.data_dir, args.confidence)

    # Save results
    if results:
        parser_instance.save_results(results)
    else:
        print("\nNo results to save.")

    print("\nDone!")


if __name__ == "__main__":
    main()
