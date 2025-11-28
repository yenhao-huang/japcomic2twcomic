"""
Text Renderer for filling Chinese text into comic images using cv2.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


class Text_Renderer:
    """
    Renders Traditional Chinese text into comic images, replacing Japanese text.
    """

    def __init__(self, font_path: Optional[str] = None, default_font_size: int = 20):
        """
        Initialize the text renderer.

        Args:
            font_path: Path to a font file supporting Traditional Chinese
            default_font_size: Default font size to use
        """
        self.font_path = font_path
        self.default_font_size = default_font_size
        self.font_cache = {}

    def render_text(self, image: np.ndarray, text_boxes: List[Tuple[int, int, int, int]],
                   texts: List[str], background_color: str = "white") -> np.ndarray:
        """
        Render Chinese text into text boxes on the image.

        Args:
            image: Input comic image (BGR format from cv2)
            text_boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            texts: List of Chinese texts to render (same length as text_boxes)
            background_color: Background color for text boxes ("white", "inpaint", or RGB tuple)

        Returns:
            Image with Chinese text rendered
        """
        if len(text_boxes) != len(texts):
            raise ValueError("Number of text boxes must match number of texts")

        # Convert to PIL Image for better text rendering
        result_image = image.copy()

        for box, text in zip(text_boxes, texts):
            if not text or not text.strip():
                continue

            x1, y1, x2, y2 = box

            # Clear the text box area first
            result_image = self._clear_text_box(result_image, box, background_color)

            # Render the Chinese text
            result_image = self._render_single_text(result_image, box, text)

        return result_image

    def _clear_text_box(self, image: np.ndarray, box: Tuple[int, int, int, int],
                        background_color: str = "white") -> np.ndarray:
        """
        Clear the text box area before rendering new text.

        Args:
            image: Input image
            box: Bounding box to clear
            background_color: Fill color ("white", "inpaint", or RGB tuple)

        Returns:
            Image with cleared text box
        """
        x1, y1, x2, y2 = box
        result = image.copy()

        if background_color == "inpaint":
            # TODO: Use inpainting to intelligently fill the background
            # For now, just use white
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), -1)
        elif background_color == "white":
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), -1)
        else:
            # Assume it's an RGB tuple
            cv2.rectangle(result, (x1, y1), (x2, y2), background_color, -1)

        return result

    def _render_single_text(self, image: np.ndarray, box: Tuple[int, int, int, int],
                           text: str) -> np.ndarray:
        """
        Render a single text string into a bounding box.

        Args:
            image: Input image
            box: Bounding box for text
            text: Chinese text to render

        Returns:
            Image with rendered text
        """
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Calculate appropriate font size
        font_size = self._calculate_font_size(text, box_width, box_height)
        font = self._get_font(font_size)

        # Handle vertical or horizontal text layout
        is_vertical = box_height > box_width * 1.5

        if is_vertical:
            # Vertical text (common in manga)
            self._draw_vertical_text(draw, text, box, font)
        else:
            # Horizontal text
            self._draw_horizontal_text(draw, text, box, font)

        # Convert back to cv2 format
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result

    def _draw_vertical_text(self, draw: ImageDraw.Draw, text: str,
                           box: Tuple[int, int, int, int], font: ImageFont.ImageFont):
        """
        Draw text vertically (top to bottom, right to left for manga).

        Args:
            draw: PIL ImageDraw object
            text: Text to draw
            box: Bounding box
            font: Font to use
        """
        x1, y1, x2, y2 = box
        # TODO: Implement vertical text rendering
        # Characters should be drawn top to bottom
        # Consider line breaks if text is too long
        current_y = y1 + 5
        current_x = x2 - 15  # Start from right side

        for char in text:
            draw.text((current_x, current_y), char, font=font, fill=(0, 0, 0))
            current_y += font.size + 2
            if current_y > y2 - font.size:
                current_y = y1 + 5
                current_x -= font.size + 5

    def _draw_horizontal_text(self, draw: ImageDraw.Draw, text: str,
                             box: Tuple[int, int, int, int], font: ImageFont.ImageFont):
        """
        Draw text horizontally (left to right).

        Args:
            draw: PIL ImageDraw object
            text: Text to draw
            box: Bounding box
            font: Font to use
        """
        x1, y1, x2, y2 = box
        # TODO: Implement horizontal text rendering with word wrapping
        # Simple centered text for now
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 - text_height) // 2

        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    def _calculate_font_size(self, text: str, box_width: int, box_height: int) -> int:
        """
        Calculate appropriate font size for the given text and box dimensions.

        Args:
            text: Text to render
            box_width: Width of the bounding box
            box_height: Height of the bounding box

        Returns:
            Calculated font size
        """
        # Simple heuristic: base it on box dimensions and text length
        is_vertical = box_height > box_width * 1.5

        if is_vertical:
            # For vertical text, font size based on width
            return min(box_width - 10, box_height // max(len(text), 1), 30)
        else:
            # For horizontal text, font size based on height
            return min(box_height - 10, box_width // max(len(text), 1), 30)

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """
        Get a font object with caching.

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
                # Try to use a default Chinese font
                # This might need to be adjusted based on the system
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            # Fallback to default font
            font = ImageFont.load_default()

        self.font_cache[size] = font
        return font

    def batch_render(self, images: List[np.ndarray],
                    text_boxes_list: List[List[Tuple[int, int, int, int]]],
                    texts_list: List[List[str]]) -> List[np.ndarray]:
        """
        Render text on multiple images.

        Args:
            images: List of images
            text_boxes_list: List of text boxes for each image
            texts_list: List of texts for each image

        Returns:
            List of images with rendered text
        """
        results = []
        for image, text_boxes, texts in zip(images, text_boxes_list, texts_list):
            result = self.render_text(image, text_boxes, texts)
            results.append(result)
        return results
