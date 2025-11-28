"""
OCR Model for Japanese text recognition from comic images.
"""

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import List, Tuple, Optional, Union
import numpy as np
import cv2
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm


class OCR_Model:
    """
    OCR model for extracting Japanese text from comic panels using PaddleOCR-VL.
    """

    def __init__(self, model_path: str = "PaddlePaddle/PaddleOCR-VL"):
        """
        Initialize the OCR model.

        Args:
            model_path: Path or name of the HuggingFace model
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.prompt = "OCR:"

        self._load_model()

    def _load_model(self):
        """Load the PaddleOCR-VL model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

    def recognize_text(self, image_path: str) -> str:
        """
        Recognize Japanese text from an image.

        Args:
            image_path: Path to input image (.png or other formats)

        Returns:
            Recognized Japanese text as a single string
        """
        # Step 1: Load image
        image = Image.open(image_path).convert("RGB")

        # Step 2: Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ]
            }
        ]

        # Step 3: Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        # Step 4: Generate output
        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return text


    def batch_recognize(self, image_paths: List[str]) -> List[str]:
        """
        Batch recognize text from multiple images with progress bar.

        Args:
            image_paths: List of image file paths

        Returns:
            List of recognized texts for each image
        """
        results = []
        for image_path in tqdm(image_paths, desc="Processing images", unit="img"):
            text = self.recognize_text(image_path)
            results.append(text)
        return results




def main():
    """
    Main function to run OCR model with command line arguments
    # 處理單一圖片
    python utils/ocr_model.py --input_path data/1/00001.png --output_path results.json

    # 處理整個資料夾
    python utils/ocr_model.py --input_dir data/1 --output_path results.json

    # 使用自訂模型路徑
    python utils/ocr_model.py --input_dir data/1 --output_path results.json --model_path "PaddlePaddle/PaddleOCR-VL"
    """
    parser = argparse.ArgumentParser(description="Japanese Comic OCR using PaddleOCR-VL")
    parser.add_argument("--input_path", type=str, help="Path to a single input image")
    parser.add_argument("--input_dir", type=str, default="data/1", help="Path to directory containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--model_path", type=str, default="PaddlePaddle/PaddleOCR-VL",
                       help="HuggingFace model path")

    args = parser.parse_args()

    # Validate input arguments
    if not args.input_path and not args.input_dir:
        parser.error("Either --input_path or --input_dir must be specified")

    if args.input_path and args.input_dir:
        parser.error("Cannot specify both --input_path and --input_dir")

    # Initialize OCR model
    print("Loading OCR model...")
    ocr_model = OCR_Model(model_path=args.model_path)
    print("Model loaded successfully!")

    results = {}

    try:
        # Process single image
        if args.input_path:
            print(f"\nProcessing image: {args.input_path}")
            text = ocr_model.recognize_text(args.input_path)
            results[args.input_path] = text
            print(f"Completed: {args.input_path}")

        # Process directory
        elif args.input_dir:
            print(f"\nProcessing directory: {args.input_dir}")
            image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
            image_files = []

            # Find all image files
            for ext in image_extensions:
                image_files.extend(Path(args.input_dir).glob(f"*{ext}"))
                image_files.extend(Path(args.input_dir).glob(f"*{ext.upper()}"))

            image_files = sorted(image_files)
            print(f"Found {len(image_files)} images")

            # Process each image with progress bar
            for image_path in tqdm(image_files, desc="Processing images", unit="img"):
                text = ocr_model.recognize_text(str(image_path))
                results[str(image_path)] = text

        # Save results to JSON
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n" + "="*50)
        print(f"Results saved to: {args.output_path}")
        print(f"Total images processed: {len(results)}")
        print("="*50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
