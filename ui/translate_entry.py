"""
Gradio UI for Japanese Comic to Traditional Chinese Comic Translation
"""

import gradio as gr
from pathlib import Path
from PIL import Image
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.manga_translator import MangaTranslator, load_pipeline_config


def create_gradio_interface():
    """Create and return the Gradio interface"""

    def translate_manga(image, progress=gr.Progress()):
        """
        Main translation function for Gradio interface

        Args:
            image: Input image (PIL Image or numpy array)
            progress: Gradio progress tracker

        Returns:
            Tuple of (translated_image, status_text)
        """
        if image is None:
            return None, "Please upload an image first."

        try:
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            # Initialize translator
            progress(0, desc="Initializing models...")
            trans = initialize_translator()

            # Define progress callback
            def progress_callback(value, desc):
                progress(value, desc=desc)

            # Run translation pipeline
            translated_image, results = trans.translate_image(
                image,
                progress_callback=progress_callback
            )

            # Generate status text
            status_parts = []

            # OCR results
            if 'ocr' in results:
                ocr_texts = []
                for ocr_item in results['ocr']:
                    if 'text' in ocr_item:
                        ocr_texts.append(ocr_item['text'])
                if ocr_texts:
                    status_parts.append(f"OCR detected text:\n{chr(10).join(ocr_texts)}")

            # Translation results
            if 'translation' in results:
                trans_texts = []
                for trans_item in results['translation']:
                    for bbox in trans_item.get('bounding_boxes', []):
                        if 'translated_text' in bbox and bbox['translated_text']:
                            trans_texts.append(f"{bbox.get('text', '')} -> {bbox['translated_text']}")
                if trans_texts:
                    status_parts.append(f"\nTranslation results:\n{chr(10).join(trans_texts)}")

            status_text = "\n".join(status_parts) if status_parts else "Translation completed successfully!"

            return translated_image, status_text

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg

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
        title="Japanese Comic to Traditional Chinese Translator",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# Japanese Comic to Traditional Chinese Comic Translator")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Manga Image",
                    type="pil",
                    height=700
                )
                translate_btn = gr.Button(
                    "Translate",
                    variant="primary",
                    size="lg"
                )

            with gr.Column():
                output_image = gr.Image(
                    label="Translated Image",
                    type="pil",
                    height=700
                )

        with gr.Row():
            status_output = gr.Textbox(
                label="Status / Translation Details",
                lines=10,
                max_lines=20
            )

        # Set up event handlers
        translate_btn.click(
            fn=translate_manga,
            inputs=[input_image],
            outputs=[output_image, status_output]
        )

        gr.Markdown("""
        ---
        **Notes:**
        - First translation may take longer as models are being loaded
        - Supported formats: JPG, PNG, WEBP
        - Best results with clear, high-resolution manga pages
        """)

    return demo


if __name__ == "__main__":
    # Load server config
    config = load_pipeline_config()
    server_config = config.get('server', {})

    demo = create_gradio_interface()
    demo.launch(
        server_name=server_config.get('server_name', "0.0.0.0"),
        server_port=server_config.get('server_port', 7860),
        share=server_config.get('share', False)
    )
