#!/usr/bin/env python3
"""
OCR Detection Dataset Annotation Tool - Gradio UI

Usage:
    python ui/dataset_annotator.py
    or
    gradio ui/dataset_annotator.py
"""

import gradio as gr
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class AnnotationState:
    def __init__(self):
        self.image_dir = None
        self.output_file = None
        self.image_files = []
        self.current_idx = 0
        self.annotations = {}  # image_name -> list of boxes
        self.current_points = []  # Points being annotated
        self.current_image_name = None

    def reset_current_points(self):
        self.current_points = []


# Global state
state = AnnotationState()


def load_images(image_dir: str, output_file: str) -> Tuple[str, str, Image.Image, str]:
    """Load images from directory and existing annotations"""
    try:
        state.image_dir = Path(image_dir)
        state.output_file = Path(output_file)

        if not state.image_dir.exists():
            return "Error: Image directory not found", "", None, ""

        # Load all images
        state.image_files = sorted([
            f for f in state.image_dir.glob('*')
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])

        if not state.image_files:
            return "Error: No images found in directory", "", None, ""

        # Load existing annotations
        state.annotations = {}
        if state.output_file.exists():
            with open(state.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        img_path, boxes_json = parts
                        img_name = Path(img_path).name
                        state.annotations[img_name] = json.loads(boxes_json)

        state.current_idx = 0
        state.current_points = []

        # Display first image
        img, info = get_current_image()

        status = f"✓ Loaded {len(state.image_files)} images from {image_dir}"
        if state.annotations:
            status += f" | Found {len(state.annotations)} existing annotations"

        return status, info, img, get_annotations_text()

    except Exception as e:
        return f"Error: {str(e)}", "", None, ""


def get_current_image() -> Tuple[Optional[Image.Image], str]:
    """Get current image with annotations drawn"""
    if not state.image_files or state.current_idx >= len(state.image_files):
        return None, ""

    img_file = state.image_files[state.current_idx]
    state.current_image_name = img_file.name

    # Load image
    img = cv2.imread(str(img_file))
    if img is None:
        return None, f"Error loading {img_file.name}"

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw existing annotations
    img_name = img_file.name
    if img_name in state.annotations:
        for box in state.annotations[img_name]:
            points = np.array(box['points'], dtype=np.int32)
            cv2.polylines(img, [points], True, (0, 255, 0), 3)
            # Draw transcription
            text = box['transcription']
            cv2.putText(img, text, tuple(points[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Draw current points being annotated
    for i, point in enumerate(state.current_points):
        cv2.circle(img, point, 8, (255, 0, 0), -1)
        cv2.putText(img, str(i+1), (point[0]+15, point[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        if i > 0:
            cv2.line(img, state.current_points[i-1], point, (255, 0, 0), 3)

    # Close polygon if 4 points
    if len(state.current_points) == 4:
        cv2.line(img, state.current_points[3], state.current_points[0], (255, 0, 0), 3)

    pil_img = Image.fromarray(img)

    # Info text
    num_boxes = len(state.annotations.get(img_name, []))
    info = f"Image {state.current_idx + 1}/{len(state.image_files)}: {img_file.name} | Boxes: {num_boxes} | Current points: {len(state.current_points)}/4"

    return pil_img, info


def handle_image_click(evt: gr.SelectData) -> Tuple[Image.Image, str, str]:
    """Handle click on image to add point"""
    if not state.image_files:
        return None, "", "Please load images first"

    if len(state.current_points) >= 4:
        return get_current_image()[0], get_current_image()[1], "Already marked 4 points. Enter transcription or reset."

    # evt.index returns (x, y) coordinates
    x, y = int(evt.index[0]), int(evt.index[1])
    state.current_points.append((x, y))

    img, info = get_current_image()

    if len(state.current_points) == 4:
        msg = f"✓ 4 points marked! Enter transcription text below and click 'Add Box'"
    else:
        msg = f"✓ Point {len(state.current_points)}/4 added at ({x}, {y})"

    return img, info, msg


def add_box(transcription: str) -> Tuple[Image.Image, str, str, str]:
    """Add current box with transcription to annotations"""
    if len(state.current_points) != 4:
        return get_current_image()[0], get_current_image()[1], "Error: Need exactly 4 points marked", ""

    if not transcription.strip():
        return get_current_image()[0], get_current_image()[1], "Error: Transcription cannot be empty", ""

    img_name = state.current_image_name

    if img_name not in state.annotations:
        state.annotations[img_name] = []

    box = {
        "transcription": transcription.strip(),
        "points": state.current_points.copy()
    }
    state.annotations[img_name].append(box)

    # Reset points
    state.current_points = []

    img, info = get_current_image()
    msg = f"✓ Box added with transcription: '{transcription}'"

    return img, info, msg, ""  # Clear transcription input


def reset_points() -> Tuple[Image.Image, str, str]:
    """Reset current points"""
    state.current_points = []
    img, info = get_current_image()
    return img, info, "Current points reset"


def delete_last_box() -> Tuple[Image.Image, str, str]:
    """Delete last annotation box for current image"""
    img_name = state.current_image_name
    if img_name in state.annotations and state.annotations[img_name]:
        deleted_box = state.annotations[img_name].pop()
        img, info = get_current_image()
        return img, info, f"✓ Deleted box: '{deleted_box['transcription']}'"
    else:
        return get_current_image()[0], get_current_image()[1], "No boxes to delete"


def next_image() -> Tuple[Image.Image, str, str, str]:
    """Go to next image"""
    if state.current_idx < len(state.image_files) - 1:
        state.current_idx += 1
        state.current_points = []
        img, info = get_current_image()
        return img, info, "→ Next image", get_annotations_text()
    else:
        return get_current_image()[0], get_current_image()[1], "Already at last image", get_annotations_text()


def prev_image() -> Tuple[Image.Image, str, str, str]:
    """Go to previous image"""
    if state.current_idx > 0:
        state.current_idx -= 1
        state.current_points = []
        img, info = get_current_image()
        return img, info, "← Previous image", get_annotations_text()
    else:
        return get_current_image()[0], get_current_image()[1], "Already at first image", get_annotations_text()


def save_annotations() -> str:
    """Save all annotations to file"""
    try:
        state.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(state.output_file, 'w', encoding='utf-8') as f:
            for img_file in state.image_files:
                img_name = img_file.name
                if img_name in state.annotations:
                    boxes = state.annotations[img_name]
                    if boxes:
                        relative_path = f"images/{img_name}"
                        f.write(f"{relative_path}\t{json.dumps(boxes, ensure_ascii=False)}\n")

        total_boxes = sum(len(boxes) for boxes in state.annotations.values())
        return f"✓ Saved {len(state.annotations)} images with {total_boxes} total boxes to {state.output_file}"

    except Exception as e:
        return f"Error saving: {str(e)}"


def get_annotations_text() -> str:
    """Get current annotations as formatted text"""
    if not state.annotations:
        return "No annotations yet"

    lines = []
    total_boxes = 0
    for img_name in sorted(state.annotations.keys()):
        boxes = state.annotations[img_name]
        if boxes:
            lines.append(f"{img_name}: {len(boxes)} boxes")
            total_boxes += len(boxes)

    summary = f"Total: {len(state.annotations)} images, {total_boxes} boxes\n" + "="*50 + "\n"
    return summary + "\n".join(lines)


def create_ui():
    """Create Gradio UI"""

    with gr.Blocks(title="OCR Dataset Annotator") as demo:
        gr.Markdown("# 📝 OCR Detection Dataset Annotation Tool")
        gr.Markdown("Click 4 points on the image (clockwise from top-left) to mark text regions, then enter the transcription.")

        with gr.Row():
            with gr.Column(scale=2):
                # Configuration
                with gr.Group():
                    gr.Markdown("### 1️⃣ Configuration")
                    image_dir_input = gr.Textbox(
                        label="Image Directory",
                        value="data/comic_benchmark/det/images",
                        placeholder="path/to/images"
                    )
                    output_file_input = gr.Textbox(
                        label="Output Annotations File",
                        value="data/comic_benchmark/det/annotations.txt",
                        placeholder="path/to/annotations.txt"
                    )
                    load_btn = gr.Button("📂 Load Images", variant="primary")
                    load_status = gr.Textbox(label="Status", interactive=False)

                # Image display
                with gr.Group():
                    gr.Markdown("### 2️⃣ Annotate (Click on image to mark points)")
                    with gr.Row():
                        image_display = gr.Image(
                            label="Click to mark 4 points (clockwise from top-left)",
                            type="pil",
                            show_label=True,
                            height=600,
                            elem_id="annotate-image"
                        )
                    image_info = gr.Textbox(label="Image Info", interactive=False)

                # Annotation controls
                with gr.Group():
                    gr.Markdown("### 3️⃣ Add Box")
                    transcription_input = gr.Textbox(
                        label="Transcription (enter '###' for illegible text)",
                        placeholder="Enter text here...",
                        lines=2
                    )
                    with gr.Row():
                        add_box_btn = gr.Button("✅ Add Box", variant="primary")
                        reset_btn = gr.Button("🔄 Reset Points")
                        delete_btn = gr.Button("🗑️ Delete Last Box", variant="stop")

                    action_status = gr.Textbox(label="Action Status", interactive=False)

                # Navigation
                with gr.Group():
                    gr.Markdown("### 4️⃣ Navigate")
                    with gr.Row():
                        prev_btn = gr.Button("⬅️ Previous Image")
                        next_btn = gr.Button("➡️ Next Image")

                # Save
                with gr.Group():
                    gr.Markdown("### 5️⃣ Save")
                    save_btn = gr.Button("💾 Save Annotations", variant="primary", size="lg")
                    save_status = gr.Textbox(label="Save Status", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Annotations Summary")
                annotations_display = gr.Textbox(
                    label="Current Annotations",
                    lines=30,
                    interactive=False
                )

                gr.Markdown("### 💡 Instructions")
                gr.Markdown("""
                1. **Load** images from directory
                2. **Click** 4 points on image (clockwise from top-left)
                3. **Enter** transcription text
                4. **Add Box** to save the annotation
                5. **Navigate** between images
                6. **Save** when done

                **Tips:**
                - Use `###` for illegible text
                - Delete last box if you make a mistake
                - Reset points to start over
                - Annotations auto-load on restart
                """)

        # Event handlers
        load_btn.click(
            fn=load_images,
            inputs=[image_dir_input, output_file_input],
            outputs=[load_status, image_info, image_display, annotations_display]
        )

        # Note: .select() event triggers when clicking on the image
        image_display.select(
            fn=handle_image_click,
            outputs=[image_display, image_info, action_status]
        )

        add_box_btn.click(
            fn=add_box,
            inputs=[transcription_input],
            outputs=[image_display, image_info, action_status, transcription_input]
        )

        reset_btn.click(
            fn=reset_points,
            inputs=None,
            outputs=[image_display, image_info, action_status]
        )

        delete_btn.click(
            fn=delete_last_box,
            inputs=None,
            outputs=[image_display, image_info, action_status]
        )

        next_btn.click(
            fn=next_image,
            inputs=None,
            outputs=[image_display, image_info, action_status, annotations_display]
        )

        prev_btn.click(
            fn=prev_image,
            inputs=None,
            outputs=[image_display, image_info, action_status, annotations_display]
        )

        save_btn.click(
            fn=save_annotations,
            inputs=None,
            outputs=[save_status]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
