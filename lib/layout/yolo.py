import argparse
import json
from pathlib import Path
from typing import List
from ultralytics import YOLO
from lib.schema import LayoutOutputSchema, RegionSchema
import cv2
import numpy as np


def load_images(input_dir: str) -> List[Path]:
    """
    Load all image files from a directory recursively.

    Args:
        input_dir: Directory path to search for images

    Returns:
        List of Path objects for image files
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    # Convert input_dir to Path object
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Directory {input_dir} does not exist")

    # Recursively find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))

    return sorted(image_files)


def crop_and_save_bboxes(image_path: Path, regions: List[RegionSchema], cropped_dir: str):
    """
    Crop each bounding box region and save to cropped_dir.

    Args:
        image_path: Path to the source image
        regions: List of detected regions with bounding boxes
        cropped_dir: Directory to save cropped images
    """
    if not regions:
        return

    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return

    # Create cropped directory
    cropped_path = Path(cropped_dir)
    cropped_path.mkdir(parents=True, exist_ok=True)

    # Get image filename without extension
    image_stem = image_path.stem

    # Crop and save each bbox
    for idx, region in enumerate(regions):
        box = region['box']
        # box format: [[x1,y1], [x2,y2]]
        x1, y1 = int(box[0][0]), int(box[0][1])
        x2, y2 = int(box[1][0]), int(box[1][1])

        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Crop the region
        cropped_img = image[y1:y2, x1:x2]

        # Save cropped image
        crop_filename = f"{image_stem}_bbox_{idx:03d}.png"
        crop_filepath = cropped_path / crop_filename
        cv2.imwrite(str(crop_filepath), cropped_img)

        print(f"  Saved cropped bbox {idx} to {crop_filepath}")


def process_single(model: YOLO, image_path: Path, output_dir: str = None, cropped_dir: str = None) -> LayoutOutputSchema:
    """
    Process a single image with YOLO model.

    Args:
        model: YOLO model instance
        image_path: Path to the image file
        output_dir: Optional directory to save visualization results
        cropped_dir: Optional directory to save cropped bounding boxes

    Returns:
        LayoutOutputSchema containing region results
    """
    # Run inference
    results = model(str(image_path))

    output: LayoutOutputSchema = {
        'source': str(image_path),
        'region_result': []
    }

    # Load image once for cropping (if needed)
    image = None
    if cropped_dir is not None:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
        else:
            # Create cropped directory
            cropped_path = Path(cropped_dir)
            cropped_path.mkdir(parents=True, exist_ok=True)

    # Get image filename without extension
    image_stem = image_path.stem

    # Process results
    region_idx = 0
    for result in results:
        # Get bounding boxes and segmentation masks
        if result.boxes is not None and result.masks is not None:
            boxes = result.boxes
            masks = result.masks

            for box, mask in zip(boxes, masks):
                coords = box.xyxy[0].cpu().numpy()

                # Convert box to [[x1,y1], [x2,y2]] format
                box_coords = [
                    [float(coords[0]), float(coords[1])],  # top-left
                    [float(coords[2]), float(coords[3])]   # bottom-right
                ]

                # Convert mask to list format
                mask_data = mask.data[0].cpu().numpy().astype(int).tolist()

                # Crop and save the region
                cropped_image_path = ""
                if cropped_dir is not None and image is not None:
                    x1, y1 = int(coords[0]), int(coords[1])
                    x2, y2 = int(coords[2]), int(coords[3])

                    # Ensure coordinates are within image bounds
                    h, w = image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # Crop the region
                    cropped_img = image[y1:y2, x1:x2]

                    # Save cropped image
                    crop_filename = f"{image_stem}_bbox_{region_idx:03d}.png"
                    crop_filepath = Path(cropped_dir) / crop_filename
                    cv2.imwrite(str(crop_filepath), cropped_img)
                    cropped_image_path = str(crop_filepath)

                region: RegionSchema = {
                    'box': box_coords,
                    'segmentation_mask': mask_data,
                    'cropped_image_path': cropped_image_path
                }

                output['region_result'].append(region)
                region_idx += 1

        # Save visualization if output_dir is specified
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Construct output filename
            output_file = output_path / image_path.name
            result.save(str(output_file))

    return output


def process_batch(model: YOLO, image_files: List[Path], output_dir: str = None, cropped_dir: str = None, verbose: bool = True) -> List[LayoutOutputSchema]:
    """
    Process a batch of images with YOLO model.

    Args:
        model: YOLO model instance
        image_files: List of image file paths
        output_dir: Optional directory to save visualization results
        cropped_dir: Optional directory to save cropped bounding boxes
        verbose: Whether to print progress information

    Returns:
        List of LayoutOutputSchema containing results for each image
    """
    results = []

    if verbose:
        print(f"Found {len(image_files)} images to process\n")

    for image_path in image_files:
        if verbose:
            print(f"Processing: {image_path}")

        result = process_single(model, image_path, output_dir, cropped_dir)
        results.append(result)

        if verbose:
            # Print summary
            num_regions = len(result['region_result'])

            if num_regions > 0:
                print(f"  Detected {num_regions} region(s)")
                for i, region in enumerate(result['region_result']):
                    box = region['box']
                    mask_shape = (len(region['segmentation_mask']), len(region['segmentation_mask'][0]) if region['segmentation_mask'] else 0)
                    print(f"    Region {i}: box={box[0]} to {box[1]}, mask_shape={mask_shape}")
            else:
                print("  No detections")

            print()

    return results


def process_images_in_dir(input_dir: str, model_path: str = "/tmp2/share_data/manga109-segmentation-bubble/best.pt", output_dir: str = None, cropped_dir: str = None):
    """
    Main function to process all images in a directory.

    Args:
        input_dir: Directory containing images to process
        model_path: Path to YOLO model file
        output_dir: Optional directory to save visualization results
        cropped_dir: Optional directory to save cropped bounding boxes
    """
    # Load the model
    model = YOLO(model_path)

    # Load images
    try:
        image_files = load_images(input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    # Process batch
    results = process_batch(model, image_files, output_dir, cropped_dir, verbose=True)

    # Save results to JSON file
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        json_file = output_path / "layout.json"

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nSaved layout results to {json_file}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images in a directory with YOLO segmentation model"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing images to process"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="/tmp2/share_data/manga109-segmentation-bubble/best.pt",
        help="Path to YOLO model (default: /tmp2/share_data/manga109-segmentation-bubble/best.pt)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="results/layout/yolo",
        help="Output directory to save visualization results (optional)"
    )
    parser.add_argument(
        "-c", "--cropped_dir",
        type=str,
        default="results/layout/yolo/cropped",
        help="Directory to save cropped bounding boxes (default: results/layout/yolo/cropped)"
    )

    args = parser.parse_args()
    process_images_in_dir(args.input_dir, args.model, args.output_dir, args.cropped_dir)