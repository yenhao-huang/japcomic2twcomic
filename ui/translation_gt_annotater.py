#!/usr/bin/env python3
"""
Translation Groundtruth Annotation Tool - Simplified Gradio UI

Load OCR groundtruth JSON files, present each text, and annotate with correct translations.
Lightweight version without image display.

Usage:
    python ui/translation_gt_annotater.py
    or
    gradio ui/translation_gt_annotater.py
"""

import gradio as gr
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.schema import TranslationOutputSchema, TranslatedBoundingBox


class AnnotationState:
    def __init__(self):
        self.input_dir = None
        self.output_dir = None
        self.json_files = []
        self.current_file_idx = 0
        self.current_box_idx = 0
        self.current_data = None  # Current OCR groundtruth data
        self.translations = {}  # box_idx -> translated_text

    def reset_current_file(self):
        self.current_box_idx = 0
        self.translations = {}
        self.current_data = None


# Global state
state = AnnotationState()


def load_groundtruth_files(input_dir: str, output_dir: str) -> Tuple[str, str]:
    """Load OCR groundtruth JSON files from directory"""
    try:
        state.input_dir = Path(input_dir)
        state.output_dir = Path(output_dir)

        if not state.input_dir.exists():
            return "Error: Input directory not found", ""

        # Load all JSON files
        state.json_files = sorted(list(state.input_dir.glob('*.json')))

        if not state.json_files:
            return "Error: No JSON files found in directory", ""

        state.output_dir.mkdir(parents=True, exist_ok=True)

        # Load first file
        state.current_file_idx = 0
        state.reset_current_file()

        status = f"✓ Loaded {len(state.json_files)} groundtruth files from {input_dir}"

        # Load and display first item
        info = load_current_file()

        return status, info

    except Exception as e:
        return f"Error: {str(e)}", ""


def load_current_file() -> str:
    """Load current JSON file and return info"""
    if not state.json_files or state.current_file_idx >= len(state.json_files):
        return "No files available"

    json_file = state.json_files[state.current_file_idx]

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            state.current_data = json.load(f)

        # Load existing translations if output file exists
        output_file = state.output_dir / json_file.name
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # Load existing translations
                state.translations = {}
                for idx, bbox in enumerate(existing_data.get('bounding_boxes', [])):
                    if 'translated_text' in bbox:
                        state.translations[idx] = bbox['translated_text']

        num_boxes = len(state.current_data.get('bounding_boxes', []))
        num_translated = len(state.translations)

        info = f"File {state.current_file_idx + 1}/{len(state.json_files)}: {json_file.name}\n"
        info += f"Total text boxes: {num_boxes} | Translated: {num_translated}/{num_boxes}"

        return info

    except Exception as e:
        return f"Error loading file: {str(e)}"


def get_current_box_display() -> Tuple[str, str, str, str]:
    """Get current text box to annotate"""
    if not state.current_data:
        return "", "", "", ""

    bboxes = state.current_data.get('bounding_boxes', [])

    if state.current_box_idx >= len(bboxes):
        return "All boxes annotated!", "", "", ""

    current_bbox = bboxes[state.current_box_idx]

    # Box info with progress
    num_translated = len(state.translations)
    box_info = f"📍 Box {state.current_box_idx + 1}/{len(bboxes)} | ✅ Translated: {num_translated}/{len(bboxes)}"

    # Original text
    original_text = current_bbox.get('text', '')

    # Existing translation if any
    existing_translation = state.translations.get(state.current_box_idx, '')

    # Context: show surrounding boxes
    context_lines = []
    for idx in range(max(0, state.current_box_idx - 2), min(len(bboxes), state.current_box_idx + 3)):
        prefix = "➤ " if idx == state.current_box_idx else "  "
        status = "✅" if idx in state.translations else "⬜"
        context_lines.append(f"{prefix}{status} [{idx+1}] {bboxes[idx].get('text', '')[:50]}")

    context = "\n".join(context_lines)

    return context, box_info, original_text, existing_translation


def save_translation(translated_text: str) -> Tuple[str, str, str, str, str]:
    """Save translation for current box and move to next"""
    if not state.current_data:
        return "", "", "", "", "Error: No data loaded"

    bboxes = state.current_data.get('bounding_boxes', [])

    if state.current_box_idx >= len(bboxes):
        return "", "", "", "", "Error: No more boxes to annotate"

    if not translated_text.strip():
        return get_current_box_display() + ("Error: Translation cannot be empty",)

    # Save translation
    state.translations[state.current_box_idx] = translated_text.strip()

    # Move to next box
    state.current_box_idx += 1

    # Get next box display
    context, box_info, original, existing = get_current_box_display()

    # Check if we finished this file
    if state.current_box_idx >= len(bboxes):
        status = f"✓ Completed all {len(bboxes)} boxes! Click 'Save File' to save."
    else:
        status = f"✓ Translation saved. Moving to box {state.current_box_idx + 1}/{len(bboxes)}"

    return context, box_info, original, existing, status


def skip_box() -> Tuple[str, str, str, str, str]:
    """Skip current box without saving translation"""
    if not state.current_data:
        return "", "", "", "", "Error: No data loaded"

    bboxes = state.current_data.get('bounding_boxes', [])

    if state.current_box_idx >= len(bboxes):
        return get_current_box_display() + ("Already at end",)

    # Move to next box without saving
    state.current_box_idx += 1

    context, box_info, original, existing = get_current_box_display()

    status = f"⏭ Skipped to box {state.current_box_idx + 1}/{len(bboxes)}"

    return context, box_info, original, existing, status


def prev_box() -> Tuple[str, str, str, str, str]:
    """Go to previous box"""
    if state.current_box_idx > 0:
        state.current_box_idx -= 1
        context, box_info, original, existing = get_current_box_display()
        return context, box_info, original, existing, f"← Previous box {state.current_box_idx + 1}"
    else:
        return get_current_box_display() + ("Already at first box",)


def save_file() -> str:
    """Save current file with translations to output directory"""
    if not state.current_data:
        return "Error: No data loaded"

    try:
        # Build TranslationOutputSchema
        translated_bboxes: List[TranslatedBoundingBox] = []

        for idx, bbox in enumerate(state.current_data.get('bounding_boxes', [])):
            translated_bbox: TranslatedBoundingBox = {
                'box': bbox.get('box', []),
                'text': bbox.get('text', ''),
                'score': bbox.get('score'),
                'translated_text': state.translations.get(idx, '')
            }
            translated_bboxes.append(translated_bbox)

        # Compile all original and translated texts
        all_original_text = '\n'.join([bbox.get('text', '') for bbox in state.current_data.get('bounding_boxes', [])])
        all_translated_text = '\n'.join([state.translations.get(idx, '') for idx in range(len(state.current_data.get('bounding_boxes', [])))])

        output_data: TranslationOutputSchema = {
            'source': state.current_data.get('source', state.current_data.get('image_path', '')),
            'text': all_original_text,
            'translated_text': all_translated_text,
            'image_path': state.current_data.get('image_path', state.current_data.get('source', '')),
            'bounding_boxes': translated_bboxes
        }

        # Save to output directory
        json_file = state.json_files[state.current_file_idx]
        output_file = state.output_dir / json_file.name

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        num_translated = len([t for t in state.translations.values() if t])
        total = len(state.current_data.get('bounding_boxes', []))

        return f"✓ Saved to {output_file.name} ({num_translated}/{total} boxes translated)"

    except Exception as e:
        return f"Error saving: {str(e)}"


def next_file() -> Tuple[str, str, str, str, str, str]:
    """Go to next file"""
    if state.current_file_idx < len(state.json_files) - 1:
        state.current_file_idx += 1
        state.reset_current_file()
        info = load_current_file()
        context, box_info, original, existing = get_current_box_display()
        return info, context, box_info, original, existing, "→ Next file"
    else:
        info = load_current_file()
        context, box_info, original, existing = get_current_box_display()
        return info, context, box_info, original, existing, "Already at last file"


def prev_file() -> Tuple[str, str, str, str, str, str]:
    """Go to previous file"""
    if state.current_file_idx > 0:
        state.current_file_idx -= 1
        state.reset_current_file()
        info = load_current_file()
        context, box_info, original, existing = get_current_box_display()
        return info, context, box_info, original, existing, "← Previous file"
    else:
        info = load_current_file()
        context, box_info, original, existing = get_current_box_display()
        return info, context, box_info, original, existing, "Already at first file"


def create_ui():
    """Create Gradio UI - Simplified text-only interface"""

    with gr.Blocks(title="Translation Groundtruth Annotator") as demo:
        gr.Markdown("# 🌐 Translation Groundtruth Annotation Tool (Simplified)")
        gr.Markdown("**Lightweight text-only interface** - Annotate translations for OCR groundtruth data.")

        # Configuration
        with gr.Group():
            gr.Markdown("### 1️⃣ Load Files")
            with gr.Row():
                input_dir = gr.Textbox(
                    label="OCR Groundtruth Directory",
                    value="data/benchmark/ocr_groundtruth",
                    placeholder="path/to/ocr_groundtruth",
                    scale=2
                )
                output_dir = gr.Textbox(
                    label="Translation Output Directory",
                    value="data/benchmark/translation_groundtruth",
                    placeholder="path/to/translation_groundtruth",
                    scale=2
                )
            load_btn = gr.Button("📂 Load Files", variant="primary", size="lg")
            load_status = gr.Textbox(label="Status", interactive=False)

        # File info
        file_info = gr.Textbox(label="Current File Progress", interactive=False, lines=2)

        # Context display
        with gr.Group():
            gr.Markdown("### 2️⃣ Text Context")
            context_display = gr.Textbox(
                label="Surrounding Text Boxes (➤ = current, ✅ = done, ⬜ = pending)",
                interactive=False,
                lines=7,
                max_lines=10
            )
            box_info = gr.Textbox(label="Progress", interactive=False)

        # Text display and input
        with gr.Group():
            gr.Markdown("### 3️⃣ Translate Current Box")
            original_text = gr.Textbox(
                label="🇯🇵 Original Japanese Text",
                interactive=False,
                lines=4
            )
            translation_input = gr.Textbox(
                label="🇹🇼 Traditional Chinese Translation",
                placeholder="輸入繁體中文翻譯...",
                lines=4
            )

            with gr.Row():
                save_btn = gr.Button("✅ Save & Next", variant="primary", scale=2)
                skip_btn = gr.Button("⏭ Skip", scale=1)
                prev_box_btn = gr.Button("⬅️ Previous", scale=1)

            action_status = gr.Textbox(label="Action Status", interactive=False)

        # File navigation and save
        with gr.Group():
            gr.Markdown("### 4️⃣ File Operations")
            with gr.Row():
                prev_file_btn = gr.Button("⬅️ Previous File")
                next_file_btn = gr.Button("➡️ Next File")
                save_file_btn = gr.Button("💾 Save File", variant="primary")
            save_status = gr.Textbox(label="Save Status", interactive=False)

        gr.Markdown("""
        ### 💡 Quick Guide
        **Workflow:** Load files → Review Japanese text → Enter Chinese translation → Save & Next → Save File when done

        **Keyboard:** Press Enter in translation field = Save & Next | **Context:** ➤ = current, ✅ = translated, ⬜ = pending
        """)

        # Event handlers
        load_btn.click(
            fn=load_groundtruth_files,
            inputs=[input_dir, output_dir],
            outputs=[load_status, file_info]
        ).then(
            fn=get_current_box_display,
            inputs=None,
            outputs=[context_display, box_info, original_text, translation_input]
        )

        save_btn.click(
            fn=save_translation,
            inputs=[translation_input],
            outputs=[context_display, box_info, original_text, translation_input, action_status]
        )

        # Allow Enter key to trigger save
        translation_input.submit(
            fn=save_translation,
            inputs=[translation_input],
            outputs=[context_display, box_info, original_text, translation_input, action_status]
        )

        skip_btn.click(
            fn=skip_box,
            inputs=None,
            outputs=[context_display, box_info, original_text, translation_input, action_status]
        )

        prev_box_btn.click(
            fn=prev_box,
            inputs=None,
            outputs=[context_display, box_info, original_text, translation_input, action_status]
        )

        save_file_btn.click(
            fn=save_file,
            inputs=None,
            outputs=[save_status]
        )

        next_file_btn.click(
            fn=next_file,
            inputs=None,
            outputs=[file_info, context_display, box_info, original_text, translation_input, action_status]
        )

        prev_file_btn.click(
            fn=prev_file,
            inputs=None,
            outputs=[file_info, context_display, box_info, original_text, translation_input, action_status]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
