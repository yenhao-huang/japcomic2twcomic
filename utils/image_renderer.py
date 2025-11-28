"""
Image renderer for rendering translated text onto comic images.
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


class TranslatedImageRenderer:
    """Renders translated Chinese text onto original comic images."""

    def __init__(self, font_path: Optional[str] = None, default_font_size: int = 20):
        """
        Initialize the renderer.

        Args:
            font_path: Path to a font file supporting Traditional Chinese
            default_font_size: Default font size
        """
        self.font_path = font_path
        self.default_font_size = default_font_size
        self.font_cache = {}

    def polygon_to_bbox(self, polygon: List[List[int]]) -> Tuple[int, int, int, int]:
        """
        Convert a polygon to a bounding box.

        Args:
            polygon: List of [x, y] coordinates

        Returns:
            (x1, y1, x2, y2) bounding box
        """
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    def render_text_on_image(self, image: np.ndarray, bounding_boxes: List[dict]) -> np.ndarray:
        """
        Render all translated text on the image.

        Args:
            image: Original image (BGR format from cv2)
            bounding_boxes: List of bounding box dictionaries with 'box' and 'translated_text'

        Returns:
            Image with rendered translated text
        """
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        for bbox_data in bounding_boxes:
            polygon = bbox_data['box']
            translated_text = bbox_data.get('translated_text', '')

            # Skip empty text
            if not translated_text or not translated_text.strip():
                continue

            # Convert polygon to bounding box
            x1, y1, x2, y2 = self.polygon_to_bbox(polygon)

            # Fill the polygon area with white background
            polygon_points = [(p[0], p[1]) for p in polygon]
            draw.polygon(polygon_points, fill=(255, 255, 255))

            # Render the text
            self._render_text_in_box(draw, translated_text, (x1, y1, x2, y2))

        # Convert back to cv2 format (BGR)
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result

    def _render_text_in_box(self, draw: ImageDraw.Draw, text: str,
                           box: Tuple[int, int, int, int]):
        """
        Render text within a bounding box.

        Args:
            draw: PIL ImageDraw object
            text: Text to render
            box: (x1, y1, x2, y2) bounding box
        """
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        # Determine if text should be vertical or horizontal
        is_vertical = box_height > box_width * 1.5

        # Calculate font size
        font_size = self._calculate_font_size(text, box_width, box_height, is_vertical)
        font = self._get_font(font_size)

        if is_vertical:
            self._draw_vertical_text(draw, text, box, font)
        else:
            self._draw_horizontal_text(draw, text, box, font)

    def _draw_vertical_text(self, draw: ImageDraw.Draw, text: str,
                           box: Tuple[int, int, int, int], font: ImageFont.ImageFont):
        """
        Draw text vertically (top to bottom, right to left for manga).

        Args:
            draw: PIL ImageDraw object
            text: Text to draw
            box: Bounding box (x1, y1, x2, y2)
            font: Font to use
        """
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        # Start from right side with some padding
        current_x = x2 - 5
        current_y = y1 + 5

        # Estimate character height
        char_height = font.size + 2
        chars_per_column = max(1, int(box_height / char_height))

        char_index = 0
        for char in text:
            if char == '\n' or char_index >= chars_per_column:
                # Move to next column (left)
                current_x -= font.size + 5
                current_y = y1 + 5
                char_index = 0
                if char == '\n':
                    continue

            # Don't draw if we're outside the box
            if current_x < x1:
                break

            draw.text((current_x - font.size, current_y), char, font=font, fill=(0, 0, 0))
            current_y += char_height
            char_index += 1

    def _draw_horizontal_text(self, draw: ImageDraw.Draw, text: str,
                             box: Tuple[int, int, int, int], font: ImageFont.ImageFont):
        """
        Draw text horizontally (left to right).

        Args:
            draw: PIL ImageDraw object
            text: Text to draw
            box: Bounding box (x1, y1, x2, y2)
            font: Font to use
        """
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        # Simple word wrapping
        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = current_line + word if not current_line else current_line + " " + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]

            if line_width <= box_width - 10:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # If text still doesn't fit, just use the whole text
        if not lines:
            lines = [text]

        # Draw lines centered in box
        line_height = font.size + 4
        total_height = len(lines) * line_height
        current_y = y1 + (box_height - total_height) // 2

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            text_x = x1 + (box_width - line_width) // 2

            draw.text((text_x, current_y), line, font=font, fill=(0, 0, 0))
            current_y += line_height

    def _calculate_font_size(self, text: str, box_width: int, box_height: int,
                           is_vertical: bool) -> int:
        """
        Calculate appropriate font size.

        Args:
            text: Text to render
            box_width: Width of bounding box
            box_height: Height of bounding box
            is_vertical: Whether text is vertical

        Returns:
            Font size
        """
        if is_vertical:
            # For vertical text, base on width and number of characters
            size = min(int(box_width * 0.7), int(box_height / max(len(text), 1) * 1.2), 30)
        else:
            # For horizontal text, base on height
            size = min(int(box_height * 0.6), int(box_width / max(len(text), 1) * 1.5), 30)

        return max(size, 12)  # Minimum font size

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """
        Get a font with caching.

        Args:
            size: Font size

        Returns:
            PIL ImageFont object
        """
        if size in self.font_cache:
            return self.font_cache[size]

        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, size)
            else:
                # Try common Chinese font paths
                font_paths = [
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/arphic/uming.ttc",
                    "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
                    "C:\\Windows\\Fonts\\msyh.ttc",  # Windows
                ]

                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, size)
                        break
                    except:
                        continue
                else:
                    # Fallback
                    font = ImageFont.load_default()
        except Exception as e:
            print(f"Warning: Could not load font, using default: {e}")
            font = ImageFont.load_default()

        self.font_cache[size] = font
        return font


def process_translated_json(json_path: str, output_dir: Optional[str] = None,
                           font_path: Optional[str] = None):
    """
    Process the translated JSON file and generate output images.

    Args:
        json_path: Path to translated_sakura.json
        output_dir: Output directory for rendered images (default: same dir as json)
        font_path: Path to custom font file
    """
    json_path = Path(json_path)

    if output_dir is None:
        output_dir = json_path.parent / "rendered"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load JSON
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Found {len(data)} images to process")

    # Initialize renderer
    renderer = TranslatedImageRenderer(font_path=font_path)

    # Process each image
    for idx, item in enumerate(data):
        source_path = item['source']
        image_path = item.get('image_path', '')
        bounding_boxes = item['bounding_boxes']

        print(f"\n[{idx + 1}/{len(data)}] Processing: {source_path}")

        # Load original image from source path
        full_source_path = Path(source_path)
        if not full_source_path.exists():
            print(f"  Warning: Source image not found: {full_source_path}")
            continue

        image = cv2.imread(str(full_source_path))
        if image is None:
            print(f"  Error: Could not read image: {full_source_path}")
            continue

        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
        print(f"  Text boxes: {len(bounding_boxes)}")

        # Render translated text
        result_image = renderer.render_text_on_image(image, bounding_boxes)

        # Generate output filename
        output_filename = full_source_path.stem + "_translated.jpg"
        output_path = output_dir / output_filename

        # Save result
        cv2.imwrite(str(output_path), result_image)
        print(f"  Saved to: {output_path}")

    print(f"\nAll done! Output saved to: {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Render translated text onto comic images from JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (outputs to results/2/rendered/)
  python -m utils.image_renderer results/2/translated_sakura.json

  # Specify custom output directory
  python -m utils.image_renderer results/2/translated_sakura.json -o output/

  # Use custom font
  python -m utils.image_renderer results/2/translated_sakura.json -f /path/to/font.ttf
        """
    )

    parser.add_argument(
        'json_path',
        type=str,
        help='Path to the translated JSON file (e.g., translated_sakura.json)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for rendered images (default: <json_dir>/rendered/)'
    )

    parser.add_argument(
        '-f', '--font',
        type=str,
        default=None,
        help='Path to custom font file supporting Traditional Chinese'
    )

    args = parser.parse_args()

    process_translated_json(
        json_path=args.json_path,
        output_dir=args.output,
        font_path=args.font
    )


if __name__ == "__main__":
    main()
