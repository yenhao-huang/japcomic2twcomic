#!/usr/bin/env python3
"""
Translation Result Visualization Tool

Compare prediction results with ground truth translations side by side.
Shows both text-level and bounding-box-level comparisons using BLEU metric.

Similarity is calculated using BLEU-4 score (0-100%):
- Higher score means better match
- Uses character-level tokenization for CJK languages

Usage:
    python ui/translation_result_visualization.py
"""

import gradio as gr
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ComparisonState:
    def __init__(self):
        self.pred_file = None
        self.gt_dir = None
        self.pred_data = []
        self.gt_data = {}
        self.current_idx = 0

    def load_files(self, pred_path: str, gt_path: str) -> str:
        """Load prediction file and ground truth directory"""
        try:
            # Load prediction file
            pred_file = Path(pred_path)
            if not pred_file.exists():
                return f"Error: Prediction file not found: {pred_path}"

            with open(pred_file, 'r', encoding='utf-8') as f:
                self.pred_data = json.load(f)

            # Load ground truth directory
            gt_dir = Path(gt_path)
            if not gt_dir.exists():
                return f"Error: Ground truth directory not found: {gt_path}"

            self.gt_data = {}
            for gt_file in gt_dir.glob('*.json'):
                with open(gt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Use filename as key (without extension)
                    self.gt_data[gt_file.stem] = data

            if not self.gt_data:
                return f"Error: No ground truth files found in {gt_path}"

            self.current_idx = 0

            return f"✓ Loaded {len(self.pred_data)} predictions and {len(self.gt_data)} ground truth files"

        except Exception as e:
            return f"Error loading files: {str(e)}"

    def get_current_comparison(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Get current prediction and matching ground truth"""
        if not self.pred_data or self.current_idx >= len(self.pred_data):
            return None, None

        pred = self.pred_data[self.current_idx]

        # Find matching ground truth by image filename
        image_path = pred.get('image_path', pred.get('source', ''))
        filename = Path(image_path).stem

        gt = self.gt_data.get(filename)

        return pred, gt


# Global state
state = ComparisonState()


def load_files(pred_path: str, gt_path: str) -> str:
    """Load prediction and ground truth files"""
    return state.load_files(pred_path, gt_path)


def draw_boxes_on_image(image_path: str, bboxes: List[Dict], color_map: Dict[int, str]) -> Optional[Image.Image]:
    """Draw bounding boxes on image with different colors"""
    try:
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img, 'RGBA')

        for idx, bbox in enumerate(bboxes):
            box = bbox.get('box', [])
            if not box or len(box) != 4:
                continue

            # Convert box format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to polygon
            polygon = [(pt[0], pt[1]) for pt in box]

            # Get color
            color = color_map.get(idx, 'blue')

            # Draw filled polygon with transparency
            if color == 'green':
                fill_color = (0, 255, 0, 30)
                outline_color = (0, 255, 0, 255)
            elif color == 'red':
                fill_color = (255, 0, 0, 30)
                outline_color = (255, 0, 0, 255)
            elif color == 'yellow':
                fill_color = (255, 255, 0, 40)
                outline_color = (255, 165, 0, 255)
            else:
                fill_color = (0, 0, 255, 30)
                outline_color = (0, 0, 255, 255)

            draw.polygon(polygon, fill=fill_color, outline=outline_color)

            # Draw box number
            center_x = sum(pt[0] for pt in box) / 4
            center_y = sum(pt[1] for pt in box) / 4
            draw.text((center_x-10, center_y-10), str(idx+1), fill=outline_color,
                     stroke_width=2, stroke_fill=(255, 255, 255))

        return img

    except Exception as e:
        print(f"Error drawing boxes: {e}")
        return None


def calculate_text_similarity(pred_text: str, gt_text: str) -> float:
    """Calculate BLEU-based similarity score (0-100)"""
    if not pred_text and not gt_text:
        return 100.0
    if not pred_text or not gt_text:
        return 0.0

    # Normalize texts
    pred_normalized = pred_text.replace(' ', '').replace('\n', '').lower()
    gt_normalized = gt_text.replace(' ', '').replace('\n', '').lower()

    if not pred_normalized and not gt_normalized:
        return 100.0
    if not pred_normalized or not gt_normalized:
        return 0.0

    # Character-level tokenization for CJK languages
    pred_tokens = list(pred_normalized)
    gt_tokens = list(gt_normalized)

    # Use smoothing to handle zero n-gram matches
    smooth = SmoothingFunction().method1

    # Calculate BLEU-4 score (standard BLEU)
    bleu_score = sentence_bleu(
        [gt_tokens],
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth
    )

    # Convert to percentage (0-100)
    return round(bleu_score * 100, 2)


def compare_boxes(pred_bbox: Dict, gt_bbox: Dict) -> Dict:
    """Compare a single bounding box"""
    pred_text = pred_bbox.get('translated_text', '')
    gt_text = gt_bbox.get('translated_text', '')

    similarity = calculate_text_similarity(pred_text, gt_text)

    return {
        'original': pred_bbox.get('text', ''),
        'pred_translation': pred_text,
        'gt_translation': gt_text,
        'similarity': similarity,
        'match': similarity > 80
    }


def display_current() -> Tuple[str, str, str, Optional[Image.Image]]:
    """Display current comparison"""
    pred, gt = state.get_current_comparison()

    if pred is None:
        return "No data loaded", "", "", None

    # File info
    image_path = pred.get('image_path', pred.get('source', ''))
    filename = Path(image_path).stem
    file_info = f"📄 File {state.current_idx + 1}/{len(state.pred_data)}: {filename}\n"

    if gt is None:
        file_info += "⚠️ No matching ground truth found"
    else:
        file_info += "✓ Ground truth found"

    # Overall text comparison
    pred_full_text = pred.get('translated_text', '')
    gt_full_text = gt.get('translated_text', '') if gt else 'N/A'

    overall_similarity = calculate_text_similarity(pred_full_text, gt_full_text) if gt else 0.0

    overall_info = f"📊 Overall BLEU Score: {overall_similarity:.1f}%\n"
    overall_info += f"Prediction boxes: {len(pred.get('bounding_boxes', []))}"
    if gt:
        overall_info += f" | Ground truth boxes: {len(gt.get('bounding_boxes', []))}"

    # Box-level comparison
    box_comparison = "🔍 Box-by-Box Comparison:\n\n"

    pred_boxes = pred.get('bounding_boxes', [])
    gt_boxes = gt.get('bounding_boxes', []) if gt else []

    color_map = {}  # idx -> color (green=match, red=mismatch, yellow=partial)

    for idx in range(max(len(pred_boxes), len(gt_boxes))):
        box_comparison += f"━━━ Box {idx + 1} ━━━\n"

        if idx < len(pred_boxes):
            pred_box = pred_boxes[idx]
            box_comparison += f"🇯🇵 Original: {pred_box.get('text', '')}\n"
            box_comparison += f"🤖 Prediction: {pred_box.get('translated_text', '')}\n"
        else:
            box_comparison += f"🤖 Prediction: [MISSING]\n"

        if idx < len(gt_boxes):
            gt_box = gt_boxes[idx]
            box_comparison += f"✅ Ground Truth: {gt_box.get('translated_text', '')}\n"
        else:
            box_comparison += f"✅ Ground Truth: [MISSING]\n"

        # Calculate BLEU score
        if idx < len(pred_boxes) and idx < len(gt_boxes):
            comparison = compare_boxes(pred_boxes[idx], gt_boxes[idx])
            similarity = comparison['similarity']
            box_comparison += f"📈 BLEU Score: {similarity:.1f}%"

            if similarity > 80:
                box_comparison += " ✅ Match\n"
                color_map[idx] = 'green'
            elif similarity > 50:
                box_comparison += " ⚠️ Partial\n"
                color_map[idx] = 'yellow'
            else:
                box_comparison += " ❌ Mismatch\n"
                color_map[idx] = 'red'
        else:
            box_comparison += "❌ Missing\n"
            color_map[idx] = 'red'

        box_comparison += "\n"

    # Draw image with boxes
    image = None
    if Path(image_path).exists():
        image = draw_boxes_on_image(image_path, pred_boxes, color_map)

    return (
        file_info,
        overall_info,
        box_comparison,
        image
    )


def next_item() -> Tuple[str, str, str, Optional[Image.Image]]:
    """Go to next item"""
    if state.current_idx < len(state.pred_data) - 1:
        state.current_idx += 1
    return display_current()


def prev_item() -> Tuple[str, str, str, Optional[Image.Image]]:
    """Go to previous item"""
    if state.current_idx > 0:
        state.current_idx -= 1
    return display_current()


def jump_to(index: int) -> Tuple[str, str, str, Optional[Image.Image]]:
    """Jump to specific index"""
    if 0 <= index < len(state.pred_data):
        state.current_idx = index
    return display_current()


def create_ui():
    """Create Gradio UI"""

    with gr.Blocks(title="Translation Result Visualization") as demo:
        gr.Markdown("# 🔍 Translation Result Visualization")
        gr.Markdown("**Compare prediction results with ground truth translations using BLEU metric**")

        # Load files
        with gr.Group():
            gr.Markdown("### 1️⃣ Load Files")
            with gr.Row():
                pred_file = gr.Textbox(
                    label="Prediction File (translated.json)",
                    value="results/2/translated_hunyuan.json",
                    placeholder="path/to/translated.json",
                    scale=2
                )
                gt_dir = gr.Textbox(
                    label="Ground Truth Directory",
                    value="data/benchmark/translation_groundtruth",
                    placeholder="path/to/translation_groundtruth",
                    scale=2
                )
            load_btn = gr.Button("📂 Load Files", variant="primary", size="lg")
            load_status = gr.Textbox(label="Status", interactive=False)

        # Navigation
        with gr.Group():
            gr.Markdown("### 2️⃣ Navigate")
            file_info = gr.Textbox(label="Current File", interactive=False, lines=2)
            overall_info = gr.Textbox(label="Overall Statistics", interactive=False, lines=2)

            with gr.Row():
                prev_btn = gr.Button("⬅️ Previous", scale=1)
                jump_input = gr.Number(label="Jump to index (0-based)", value=0, scale=1)
                jump_btn = gr.Button("🎯 Jump", scale=1)
                next_btn = gr.Button("➡️ Next", scale=1)

        # Image visualization
        with gr.Group():
            gr.Markdown("### 3️⃣ Visual Comparison")
            gr.Markdown("**Color Legend:** 🟢 Green = Match (>80%) | 🟡 Yellow = Partial (50-80%) | 🔴 Red = Mismatch (<50%)")
            image_display = gr.Image(label="Image with Bounding Boxes", type="pil")

        # Box-level details
        with gr.Group():
            gr.Markdown("### 4️⃣ Detailed Box Comparison")
            box_details = gr.Textbox(
                label="Box-by-Box Analysis",
                interactive=False,
                lines=20,
                max_lines=30
            )

        # Event handlers
        load_btn.click(
            fn=load_files,
            inputs=[pred_file, gt_dir],
            outputs=[load_status]
        ).then(
            fn=display_current,
            inputs=None,
            outputs=[file_info, overall_info, box_details, image_display]
        )

        next_btn.click(
            fn=next_item,
            inputs=None,
            outputs=[file_info, overall_info, box_details, image_display]
        )

        prev_btn.click(
            fn=prev_item,
            inputs=None,
            outputs=[file_info, overall_info, box_details, image_display]
        )

        jump_btn.click(
            fn=jump_to,
            inputs=[jump_input],
            outputs=[file_info, overall_info, box_details, image_display]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=3862,
        share=False
    )
