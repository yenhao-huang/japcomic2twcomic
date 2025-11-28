"""
Test script for OCR Model
Tests the utils/ocr_model.py using test images in tests/testcases/ocrmodel/
"""

import sys
import os
from pathlib import Path

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ocr_model import OCR_Model


def test_ocr_single_image():
    """
    Test OCR on a single image
    """
    print("="*60)
    print("Test 1: Single Image OCR")
    print("="*60)

    # Initialize OCR model
    print("\nLoading OCR model...")
    ocr_model = OCR_Model(model_path="PaddlePaddle/PaddleOCR-VL")
    print("Model loaded successfully!")

    # Test image path
    test_image = "tests/testcases/ocrmodel/00001.webp"

    if not os.path.exists(test_image):
        print(f"Error: Test image not found: {test_image}")
        return

    print(f"\nProcessing image: {test_image}")

    # Run OCR
    text = ocr_model.recognize_text(test_image)

    # Display results
    print("\n" + "-"*60)
    print("OCR Result:")
    print("-"*60)
    print(text)
    print("-"*60)

    return text


def test_ocr_batch():
    """
    Test OCR on multiple images using batch processing
    """
    print("\n\n" + "="*60)
    print("Test 2: Batch OCR")
    print("="*60)

    # Initialize OCR model
    print("\nLoading OCR model...")
    ocr_model = OCR_Model(model_path="PaddlePaddle/PaddleOCR-VL")
    print("Model loaded successfully!")

    # Test directory
    test_dir = "tests/testcases/ocrmodel"

    # Find all test images
    image_extensions = {".webp", ".png", ".jpg", ".jpeg"}
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(test_dir).glob(f"*{ext}"))
        image_files.extend(Path(test_dir).glob(f"*{ext.upper()}"))

    # Filter out the ocr_model.py file
    image_files = [f for f in image_files if f.suffix.lower() in image_extensions]
    image_files = sorted(image_files)

    print(f"\nFound {len(image_files)} test images")

    if len(image_files) == 0:
        print("No test images found!")
        return

    # Run batch OCR
    print("\nProcessing images...")
    image_paths = [str(f) for f in image_files]
    texts = ocr_model.batch_recognize(image_paths)

    # Display results
    print("\n" + "-"*60)
    print("Batch OCR Results:")
    print("-"*60)

    for img_path, text in zip(image_paths, texts):
        print(f"\n{Path(img_path).name}:")
        print(f"  {text[:100]}..." if len(text) > 100 else f"  {text}")

    print("-"*60)

    return dict(zip(image_paths, texts))


def test_ocr_directory():
    """
    Test OCR using directory argument (simulating command line usage)
    """
    print("\n\n" + "="*60)
    print("Test 3: Directory Processing (Command Line Simulation)")
    print("="*60)

    import json

    # Initialize OCR model
    print("\nLoading OCR model...")
    ocr_model = OCR_Model(model_path="PaddlePaddle/PaddleOCR-VL")
    print("Model loaded successfully!")

    # Test directory
    test_dir = "tests/testcases/ocrmodel"
    output_file = "tests/testcases/ocrmodel/test_results.json"

    print(f"\nProcessing directory: {test_dir}")

    # Find all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(test_dir).glob(f"*{ext}"))
        image_files.extend(Path(test_dir).glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images")

    # Process images
    results = {}
    for image_path in image_files:
        text = ocr_model.recognize_text(str(image_path))
        results[str(image_path)] = text

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Total images processed: {len(results)}")

    return results


def main():
    """
    Run all tests
    """
    print("Starting OCR Model Tests")
    print("Testing utils/ocr_model.py with test images from tests/testcases/ocrmodel/")
    print()

    try:
        # Test 1: Single image
        result1 = test_ocr_single_image()

        # Test 2: Batch processing
        result2 = test_ocr_batch()

        # Test 3: Directory processing
        result3 = test_ocr_directory()

        print("\n\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
