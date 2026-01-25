"""
Gradio UI for Batch Japanese Comic to Traditional Chinese Comic Translation

Supports:
- Uploading multiple manga images
- Batch processing with progress tracking
"""

import gradio as gr
from pathlib import Path
from PIL import Image
import sys
import time
from typing import List, Tuple
import shutil
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.manga_translator import MangaTranslator, load_pipeline_config


def create_gradio_interface():
    """Create and return the Gradio batch processing interface"""

    def translate_batch(files, progress=gr.Progress()):
        """
        Batch translation function for Gradio interface

        Args:
            files: List of uploaded files
            progress: Gradio progress tracker

        Returns:
            List of tuples (original_image, translated_image, status_text)
        """
        try:
            # Collect image files
            valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
            image_files = []

            # Process uploaded files
            if files:
                for file_obj in files:
                    if hasattr(file_obj, 'name'):
                        file_path = Path(file_obj.name)
                    else:
                        file_path = Path(file_obj)

                    # Check if it's a file (not directory) and has valid extension
                    if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                        image_files.append(str(file_path))

            # Remove duplicates
            image_files = list(set(image_files))

            if not image_files:
                return [(None, None, "No valid image files found. Please upload images. Supported formats: JPG, PNG, WEBP, BMP, GIF")]

            # Initialize translator
            progress(0, desc="Initializing models...")
            trans = initialize_translator()

            total_images = len(image_files)
            results = []

            for idx, image_path in enumerate(image_files):
                image_progress = idx / total_images
                progress(image_progress, desc=f"Processing image {idx + 1}/{total_images}...")

                try:
                    # Load image
                    image = Image.open(image_path)

                    # Progress callback for individual image
                    def progress_callback(value, desc):
                        overall_progress = (idx + value) / total_images
                        progress(overall_progress, desc=f"[{idx+1}/{total_images}] {desc}")

                    # Run translation pipeline
                    start_time = time.time()
                    translated_image, translation_results = trans.translate_image(
                        image,
                        progress_callback=progress_callback
                    )
                    elapsed_time = time.time() - start_time

                    # Print status to console (backend)
                    print(f"--- Image {idx+1}/{total_images}: {Path(image_path).name} ---")
                    print(f"Processing time: {elapsed_time:.2f}s")

                    # OCR results
                    if 'ocr' in translation_results:
                        ocr_count = len(translation_results['ocr'])
                        print(f"Text regions detected: {ocr_count}")

                    # Translation results
                    if 'translation' in translation_results:
                        trans_count = 0
                        for trans_item in translation_results['translation']:
                            trans_count += len(trans_item.get('bounding_boxes', []))
                        print(f"Texts translated: {trans_count}")

                    print("Status: Success")

                    results.append((image, translated_image))

                except Exception as e:
                    import traceback
                    error_msg = f"Error processing {Path(image_path).name}:\n{str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    results.append((None, None))

            progress(1.0, desc=f"Completed! Processed {total_images} images")
            return results

        except Exception as e:
            import traceback
            error_msg = f"Batch processing error: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return [(None, None)]

    def save_results(results):
        """
        Prepare translated images for client download as a zip file

        Args:
            results: List of (original_image, translated_image) tuples

        Returns:
            Tuple of (zip_file_path, status_message)
        """
        if not results:
            return None, "No results to save."

        try:
            # Create a temporary directory for the images
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir)

            saved_count = 0
            for idx, (original, translated) in enumerate(results):
                if translated is not None:
                    output_file = temp_path / f"translated_{idx+1:03d}.jpg"
                    translated.save(str(output_file), quality=95)
                    saved_count += 1

            if saved_count == 0:
                return None, "No translated images to save."

            # Create zip file
            zip_path = tempfile.mktemp(suffix='.zip')
            shutil.make_archive(zip_path[:-4], 'zip', temp_dir)

            # Clean up temp directory
            shutil.rmtree(temp_dir)

            return zip_path, f"Ready to download: {saved_count} translated images"

        except Exception as e:
            import traceback
            return None, f"Error preparing download: {str(e)}\n{traceback.format_exc()}"

    # Pre-load all models at startup
    print("=" * 60)
    print("Pre-loading all models at startup...")
    print("=" * 60)
    translator = MangaTranslator()
    # Access each model to trigger loading
    _ = translator.yolo_model
    _ = translator.text_det_model
    _ = translator.ocr_model
    _ = translator.translation_model
    _ = translator.renderer
    print("=" * 60)
    print("All models loaded successfully!")
    print("=" * 60)

    def initialize_translator():
        return translator

    # Create Gradio interface
    with gr.Blocks(
        title="Batch Japanese Comic Translator",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# Batch Japanese Comic to Traditional Chinese Comic Translator")
        gr.Markdown("Upload multiple manga images for batch translation")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload Multiple Images")
                file_input = gr.File(
                    label="Upload Images (Multiple Selection)",
                    file_count="multiple",
                    type="filepath",
                    file_types=["image"]
                )

                translate_btn = gr.Button(
                    "Translate Batch",
                    variant="primary",
                    size="lg"
                )

        with gr.Row():
            output_gallery = gr.Gallery(
                label="Translation Results (Original | Translated)",
                show_label=True,
                columns=2,
                rows=2,
                height=800,
                object_fit="contain"
            )

        with gr.Row():
            save_btn = gr.Button(
                "Download Results",
                variant="secondary"
            )

        with gr.Row():
            download_file = gr.File(
                label="Download",
                visible=True
            )
            save_status = gr.Textbox(
                label="Status",
                lines=2
            )

        # State to store results
        results_state = gr.State([])

        def update_gallery(results):
            """Format results for gallery display"""
            gallery_items = []

            for idx, (original, translated) in enumerate(results):
                if translated is not None:
                    gallery_items.append(original)
                    gallery_items.append(translated)

            return gallery_items, results

        # Set up event handlers
        translate_btn.click(
            fn=translate_batch,
            inputs=[file_input],
            outputs=[results_state]
        ).then(
            fn=update_gallery,
            inputs=[results_state],
            outputs=[output_gallery, results_state]
        )

        save_btn.click(
            fn=save_results,
            inputs=[results_state],
            outputs=[download_file, save_status]
        )

        gr.Markdown("""
        ---
        **Notes:**
        - **Supported formats:** JPG, PNG, WEBP, BMP, GIF
        - Upload multiple image files using the file picker (select multiple files with Ctrl/Cmd+Click)
        - Best results with clear, high-resolution manga pages
        - Click "Download Results" to download all translated images as a ZIP file
        """)

    return demo


if __name__ == "__main__":
    # Load server config
    config = load_pipeline_config()
    server_config = config.get('server', {})

    demo = create_gradio_interface()
    demo.launch(
        server_name=server_config.get('server_name', "0.0.0.0"),
        server_port=server_config.get('server_port', 7861),  # Different port from single image UI
        share=server_config.get('share', False)
    )
