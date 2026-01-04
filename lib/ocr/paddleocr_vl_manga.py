"""
PaddleOCR-VL implementation using HuggingFace Transformers
Supports OCR, table recognition, chart recognition, and formula recognition
"""

from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
import argparse
import json
import os
from typing import List, Dict, Literal, Union
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from lib.schema import OcrOutputSchema, OcrErrorSchema, LayoutOutputSchema

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TaskType = Literal["ocr", "table", "chart", "formula"]

PROMPTS: Dict[TaskType, str] = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
}


class PaddleOCRVLMANGA:
    """PaddleOCR-VL using Transformers for OCR and document understanding tasks"""

    def __init__(
        self,
        model_path: str = "./paddleocr-vl-finetuned",
        processor_path: str = "./paddleocr-vl-finetuned",
        task: TaskType = "ocr",
        device: str = None,
        max_new_tokens: int = 256,
    ):
        """
        Initialize PaddleOCR-VL model with transformers

        Args:
            model_path: Path to the model weights
            processor_path: Path to the processor/tokenizer
            task: Type of task ('ocr', 'table', 'chart', 'formula')
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None
            max_new_tokens: Maximum number of tokens to generate
        """
        self.device = device or DEVICE
        self.task = task
        self.max_new_tokens = max_new_tokens

        print(f"🔧 Loading PaddleOCR-VL model from: {model_path}")
        print(f"   Device: {self.device}")
        print(f"   Task: {task}")

        # Load model with proper dtype handling
        import logging
        # Suppress the config logging that causes JSON serialization error
        logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            .to(self.device)
            .eval()
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=True, use_fast=True
        )

        # Reset logging level
        logging.getLogger("transformers.configuration_utils").setLevel(logging.INFO)

        print(f"✅ Model loaded successfully!")

    def predict(self, image_path: str) -> str:
        """
        Run OCR/recognition on a single image

        Args:
            image_path: Path to the image file

        Returns:
            str: Recognized text/content
        """
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPTS[self.task]},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated[:, input_length:]  # Slice only new tokens
        answer = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[
            0
        ]

        return answer

    def predict_batch(
        self, image_paths: List[str], show_progress: bool = True
    ) -> List[Union[OcrOutputSchema, OcrErrorSchema]]:
        """
        Run OCR/recognition on multiple images

        Args:
            image_paths: List of image file paths
            show_progress: Whether to show progress bar

        Returns:
            List of OcrOutputSchema or OcrErrorSchema results
        """
        results: List[Union[OcrOutputSchema, OcrErrorSchema]] = []

        iterator = (
            tqdm(image_paths, desc="Processing images")
            if show_progress
            else image_paths
        )

        for image_path in iterator:
            try:
                answer = self.predict(image_path)
                result: OcrOutputSchema = {
                    "source": str(image_path),
                    "text": answer,
                    "image_path": str(image_path),
                    "bounding_boxes": []  # PaddleOCR-VL doesn't provide bounding boxes
                }
                results.append(result)
            except Exception as e:
                error_result: OcrErrorSchema = {
                    "source": str(image_path),
                    "error": str(e)
                }
                results.append(error_result)
                print(f"❌ Error processing {image_path}: {e}")

        return results

    def load_directory(
        self, directory_path: str, recursive: bool = True
    ) -> List[str]:
        """
        Load all image files from a directory

        Args:
            directory_path: Path to directory containing images
            recursive: Whether to search subdirectories

        Returns:
            List of image file paths
        """
        path = Path(directory_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Supported image extensions
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".gif"}

        image_files = []

        if recursive:
            for ext in image_extensions:
                image_files.extend(path.rglob(f"*{ext}"))
                image_files.extend(path.rglob(f"*{ext.upper()}"))
        else:
            for ext in image_extensions:
                image_files.extend(path.glob(f"*{ext}"))
                image_files.extend(path.glob(f"*{ext.upper()}"))

        return sorted([str(f) for f in image_files])

    def load_layout_json(self, layout_json_path: str) -> List[LayoutOutputSchema]:
        """
        Load layout data from a layout.json file

        Args:
            layout_json_path: Path to layout.json file

        Returns:
            List of LayoutOutputSchema objects
        """
        if not os.path.exists(layout_json_path):
            raise FileNotFoundError(f"Layout JSON not found: {layout_json_path}")

        with open(layout_json_path, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)

        return layout_data

    def _box_2points_to_4points(self, box: List[List[float]]) -> List[List[float]]:
        """
        將兩點座標（對角線）轉換成四點座標（矩形四個角）

        Args:
            box: [[x1,y1], [x2,y2]] - 左上角和右下角

        Returns:
            [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] - 左上、右上、右下、左下
        """
        if len(box) == 4:
            # 已經是四個點，直接返回
            return box

        if len(box) != 2:
            raise ValueError(f"Expected 2 or 4 points, got {len(box)}")

        x1, y1 = box[0]
        x2, y2 = box[1]

        # 返回四個角點：左上、右上、右下、左下
        return [
            [x1, y1],  # 左上
            [x2, y1],  # 右上
            [x2, y2],  # 右下
            [x1, y2]   # 左下
        ]

    def process_layout_regions(
        self, layout_data: List[LayoutOutputSchema], show_progress: bool = True
    ) -> List[Union[OcrOutputSchema, OcrErrorSchema]]:
        """
        Process all regions from layout data and run OCR on each cropped region
        將 OCR 結果與 layout_data 的 bbox 進行匹配和補全

        Args:
            layout_data: List of LayoutOutputSchema objects
            show_progress: Whether to show progress bar

        Returns:
            List of OcrOutputSchema with bounding_boxes filled from layout data
        """
        results: List[Union[OcrOutputSchema, OcrErrorSchema]] = []

        # Collect all cropped image paths and their corresponding boxes
        region_info = []  # [(cropped_path, box, source_image)]
        for layout_entry in layout_data:
            source = layout_entry.get("source", "")
            for region in layout_entry.get("region_result", []):
                cropped_path = region.get("cropped_image_path")
                box = region.get("box")
                if cropped_path and box:
                    region_info.append((cropped_path, box, source))

        if not region_info:
            print("⚠️  No cropped regions found in layout data")
            return results

        print(f"📊 Found {len(region_info)} regions to process from {len(layout_data)} images")

        # Extract just the paths for OCR processing
        image_paths = [info[0] for info in region_info]

        # Process all cropped regions
        ocr_results = self.predict_batch(image_paths, show_progress=show_progress)

        # Merge OCR results with bbox information
        for idx, ocr_result in enumerate(ocr_results):
            if "error" in ocr_result:
                results.append(ocr_result)
                continue

            # Get corresponding box from layout data
            _, box_2points, source = region_info[idx]

            # Convert 2-point box to 4-point box
            box_4points = self._box_2points_to_4points(box_2points)

            # Add bounding box information to OCR result
            ocr_result["bounding_boxes"] = [{
                "box": box_4points,
                "text": ocr_result["text"],
                "score": None  # PaddleOCR-VL doesn't provide confidence scores
            }]

            results.append(ocr_result)

        return results


def process_single_image(ocr: PaddleOCRVLMANGA, args):
    """Process a single image and save the result"""
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found - {args.image}")
        return

    print(f"\n🖼️  Processing image: {args.image}")
    result = ocr.predict(args.image)

    print("\n" + "=" * 60)
    print(f"📝 {args.task.upper()} Result:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # Save output if specified
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n💾 Result saved to: {args.output}")


def process_directory_batch(ocr: PaddleOCRVLMANGA, args):
    """Process all images in a directory in batch mode"""
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Directory not found - {args.input_dir}")
        return

    print(f"\n📁 Loading directory: {args.input_dir}")
    recursive = not args.no_recursive
    print(f"   Recursive: {recursive}")

    image_paths = ocr.load_directory(args.input_dir, recursive=recursive)

    if not image_paths:
        print("⚠️  No images found in directory")
        return

    print(f"✅ Found {len(image_paths)} images\n")

    # Batch processing
    results = ocr.predict_batch(image_paths)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Batch Processing Summary:")
    print("=" * 60)

    success_count = sum(1 for r in results if "error" not in r)
    error_count = len(results) - success_count

    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for result in results:
        print(f"\n📄 {Path(result['source']).name}:")
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            # Truncate long text
            text = result["text"]
            display_text = text[:100] + "..." if len(text) > 100 else text
            print(f"   {display_text}")

    # Save results
    if args.output_dir:
        print(results)
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"paddleocrvl.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Results saved to: {output_path}")


def process_layout_json(ocr: PaddleOCRVLMANGA, args):
    """Process regions from a layout JSON file"""
    if not os.path.exists(args.layout_json):
        print(f"❌ Error: Layout JSON file not found - {args.layout_json}")
        return

    print(f"\n📁 Loading layout data from: {args.layout_json}")

    layout_data = ocr.load_layout_json(args.layout_json)
    print(f"✅ Loaded {len(layout_data)} layout entries\n")

    # Process all regions from layout
    results = ocr.process_layout_regions(layout_data)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Layout Region OCR Summary:")
    print("=" * 60)

    success_count = sum(1 for r in results if "error" not in r)
    error_count = len(results) - success_count

    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for result in results:
        print(f"\n📄 {Path(result['source']).name}:")
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            # Truncate long text
            text = result["text"]
            display_text = text[:100] + "..." if len(text) > 100 else text
            print(f"   {display_text}")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"paddleocrvl_layout.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL Transformers - OCR and Document Understanding"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/tmp2/share_data/PaddleOCR-VL-For-Manga",
        help="Path to model weights",
    )
    parser.add_argument(
        "--processor-path",
        type=str,
        default="/tmp2/share_data/PaddleOCR-VL-For-Manga",
        help="Path to processor/tokenizer",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["ocr", "table", "chart", "formula"],
        default="ocr",
        help="Task type: ocr, table, chart, or formula",
    )
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument(
        "--input-dir", type=str, help="Directory path for batch processing"
    )
    parser.add_argument(
        "--layout-json", type=str, help="Path to layout.json file for region-based OCR"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories (only for --input-dir)",
    )
    parser.add_argument(
        "--output", type=str, help="Output text file path (for single image)"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for batch results"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Maximum tokens to generate"
    )

    args = parser.parse_args()

    # Validate input
    if not args.image and not args.input_dir and not args.layout_json:
        print("❌ Error: Please provide --image, --input-dir, or --layout-json")
        parser.print_help()
        return

    # Check for mutually exclusive options
    input_options = sum([bool(args.image), bool(args.input_dir), bool(args.layout_json)])
    if input_options > 1:
        print("❌ Error: --image, --input-dir, and --layout-json cannot be used together")
        return

    # Initialize model
    ocr = PaddleOCRVLMANGA(
        model_path=args.model_path,
        processor_path=args.processor_path,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
    )

    # Single image processing
    if args.image:
        process_single_image(ocr, args)

    # Batch directory processing
    elif args.input_dir:
        process_directory_batch(ocr, args)

    # Layout JSON processing
    elif args.layout_json:
        process_layout_json(ocr, args)




if __name__ == "__main__":
    main()
