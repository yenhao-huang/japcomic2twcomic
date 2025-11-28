"""
Text Detection Model using YOLO for detecting text boxes in comic images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch


class Text_Detection_Model:
    """
    YOLO-based text detection model for finding text regions in comic images.
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5, device: Optional[str] = None):
        """
        Initialize the text detection model.

        Args:
            model_path: Path to the YOLO model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path or 'yolo11x.pt')
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")


    def detect_text_boxes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text boxes in a comic image.

        Args:
            image: Input comic image as numpy array (BGR format from cv2)

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
            Coordinates are in absolute pixel values
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call _load_model() first.")

        # Run YOLO inference
        results = self.model(image, verbose=False)

        boxes = []
        if len(results) > 0:
            result = results[0]

            # Filter by confidence threshold and extract boxes
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    # Get coordinates in xyxy format
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    boxes.append((x1, y1, x2, y2))

        # Sort boxes in reading order (manga style: right to left, top to bottom)
        boxes = self._sort_boxes_reading_order(boxes)

        return boxes

    def detect_with_confidence(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect text boxes with confidence scores. 越有信心分排在越前面

        Args:
            image: Input comic image as numpy array

        Returns:
            List of tuples [(box, confidence), ...]
            where box is (x1, y1, x2, y2) and confidence is float
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call _load_model() first.")

        # Run YOLO inference
        results = self.model(image, verbose=False)

        boxes_with_conf = []
        if len(results) > 0:
            result = results[0]

            # Filter by confidence threshold and extract boxes with confidence
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    # Get coordinates in xyxy format
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    boxes_with_conf.append(((x1, y1, x2, y2), conf))

        # Sort boxes in reading order (manga style: right to left, top to bottom)
        boxes_with_conf = sorted(boxes_with_conf, key=lambda x: (x[0][1], -x[0][0]))

        return boxes_with_conf

    def batch_detect(self, images: List[np.ndarray]) -> List[List[Tuple[int, int, int, int]]]:
        """
        Detect text boxes in multiple images.

        Args:
            images: List of comic images

        Returns:
            List of text boxes for each image
        """
        results = []
        for image in images:
            boxes = self.detect_text_boxes(image)
            results.append(boxes)
        return results

    def visualize_detections(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                            color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on the image for visualization.

        Args:
            image: Input image
            boxes: List of bounding boxes
            color: Color for drawing boxes (B, G, R)
            thickness: Line thickness

        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            # Add box number
            cv2.putText(vis_image, f"{i+1}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        return vis_image

    def _sort_boxes_reading_order(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Sort boxes in manga reading order (right to left, top to bottom for Japanese manga).

        Args:
            boxes: List of bounding boxes

        Returns:
            Sorted list of bounding boxes
        """
        # Japanese manga reads right to left, top to bottom
        # Sort by y (top), then -x (right to left)
        return sorted(boxes, key=lambda box: (box[1], -box[0]))


def main():
    """
    Main function to demonstrate text detection on comic images.
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Detect text boxes in comic images')
    parser.add_argument('--input_dir', type=str, help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for visualizations')
    parser.add_argument('--model', type=str, default=None, help='Path to YOLO model weights')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0-1)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--show-confidence', action='store_true', help='Show confidence scores')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization output')

    args = parser.parse_args()

    # Create output directory
    if not args.no_visualize:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    print(f"Loading model on device: {args.device or 'auto'}")
    detector = Text_Detection_Model(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )
    print("Model loaded successfully!")

    # Process single image or directory
    image_paths = []
    if os.path.isfile(args.input_dir):
        image_paths = [args.input_dir]
    elif os.path.isdir(args.input_dir):
        # Get all image files
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for filename in sorted(os.listdir(args.input_dir)):
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(args.input_dir, filename))
    else:
        print(f"Error: {args.input_dir} is not a valid file or directory")
        return

    print(f"Processing {len(image_paths)} image(s)...")

    # Process each image
    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not read image {img_path}")
            continue

        # Detect text boxes
        if args.show_confidence:
            results = detector.detect_with_confidence(image)
            boxes = [box for box, conf in results]
            print(f"Detected {len(boxes)} text boxes:")
            for i, (box, conf) in enumerate(results, 1):
                x1, y1, x2, y2 = box
                print(f"  Box {i}: ({x1}, {y1}, {x2}, {y2}) - Confidence: {conf:.3f}")
        else:
            boxes = detector.detect_text_boxes(image)
            print(f"Detected {len(boxes)} text boxes:")
            for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
                print(f"  Box {i}: ({x1}, {y1}, {x2}, {y2})")

        # Visualize detections
        if not args.no_visualize and len(boxes) > 0:
            vis_image = detector.visualize_detections(image, boxes)

            # Save visualization
            output_filename = os.path.basename(img_path)
            output_path = os.path.join(args.output_dir, f"detected_{output_filename}")
            cv2.imwrite(output_path, vis_image)
            print(f"Visualization saved to: {output_path}")

    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
