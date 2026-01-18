"""
Gradio UI for Japanese Comic to Traditional Chinese Comic Translation

Pipeline:
1. Text Mask Detection (YOLO segmentation)
2. Text Detection (PPOCRv5)
3. OCR Recognition (PaddleOCR-VL)
4. Translation (Hunyuan)
5. Text Region Refinement (Text Allocater)
6. Render (Text Det Render)
"""

import gradio as gr
import tempfile
import shutil
import json
import time
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from PIL import Image
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.text_mask.yolo import process_single, YOLO
from lib.text_detection.ppocrv5 import OCR as TextDetOCR
from lib.ocr.paddleocr_vl_manga import PaddleOCRVLMANGA
from lib.translation.hunyuan import Translation_Model_Hunyuan
from lib.text_allocater.text_allocater import TextAllocater
from lib.render.text_det_render import TextDetRender
from lib.schema import LayoutOutputSchema, OcrOutputSchema, TranslationOutputSchema


def load_pipeline_config(config_path: str = "configs/pipeline.yml") -> dict:
    """Load pipeline configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class MangaTranslator:
    """
    Japanese Comic to Traditional Chinese Comic Translation Pipeline
    """

    def __init__(
        self,
        config_path: str = "configs/pipeline.yml",
        **kwargs
    ):
        """
        Initialize all models

        Args:
            config_path: Path to pipeline configuration YAML file
            **kwargs: Override config values (yolo_model_path, text_det_model_dir, etc.)
        """
        # Load config from YAML
        config = load_pipeline_config(config_path)
        models_config = config.get('models', {})
        translation_config = config.get('translation', {})
        renderer_config = config.get('renderer', {})
        text_allocater_config = config.get('text_allocater', {})

        # Model paths (can be overridden by kwargs)
        self.yolo_model_path = kwargs.get('yolo_model_path',
            models_config.get('yolo_model_path', "/tmp2/share_data/manga109-segmentation-bubble/best.pt"))
        self.text_det_model_dir = kwargs.get('text_det_model_dir',
            models_config.get('text_det_model_dir', "PaddleOCR/output/inference/PP-OCRv5_comic_det_infer/"))
        self.ocr_model_path = kwargs.get('ocr_model_path',
            models_config.get('ocr_model_path', "/tmp2/share_data/PaddleOCR-VL-For-Manga"))
        self.translation_model_path = kwargs.get('translation_model_path',
            models_config.get('translation_model_path', "/tmp2/share_data/HY-MT1.5-7B"))
        self.prompt_config = kwargs.get('prompt_config',
            translation_config.get('prompt_config', "configs/translation_prompt/hunyuan_default.yml"))

        # Renderer config
        self.renderer_config = {
            'font_path': kwargs.get('font_path', renderer_config.get('font_path')),
            'default_font_size': kwargs.get('default_font_size', renderer_config.get('default_font_size', 20)),
            'inpaint_shrink_ratio': kwargs.get('inpaint_shrink_ratio', renderer_config.get('inpaint_shrink_ratio', 0.8)),
            'default_is_vertical': kwargs.get('default_is_vertical', renderer_config.get('default_is_vertical', True)),
            'min_font_size': kwargs.get('min_font_size', renderer_config.get('min_font_size', 15)),
        }

        # Text allocater config
        self.split_strategy = kwargs.get('split_strategy',
            text_allocater_config.get('split_strategy', "region_newline"))

        # Lazy loading - models will be loaded on first use
        self._yolo_model = None
        self._text_det_model = None
        self._ocr_model = None
        self._translation_model = None
        self._renderer = None

    @property
    def yolo_model(self):
        if self._yolo_model is None:
            print("Loading YOLO segmentation model...")
            self._yolo_model = YOLO(self.yolo_model_path)
        return self._yolo_model

    @property
    def text_det_model(self):
        if self._text_det_model is None:
            print("Loading text detection model...")
            self._text_det_model = TextDetOCR(
                text_det_model_dir=self.text_det_model_dir
            )
        return self._text_det_model

    @property
    def ocr_model(self):
        if self._ocr_model is None:
            print("Loading OCR model...")
            self._ocr_model = PaddleOCRVLMANGA(
                model_path=self.ocr_model_path,
                processor_path=self.ocr_model_path,
            )
        return self._ocr_model

    @property
    def translation_model(self):
        if self._translation_model is None:
            print("Loading translation model...")
            self._translation_model = Translation_Model_Hunyuan(
                model_path=self.translation_model_path,
                prompt_config=self.prompt_config,
            )
        return self._translation_model

    @property
    def renderer(self):
        if self._renderer is None:
            print("Initializing renderer...")
            self._renderer = TextDetRender(**self.renderer_config)
        return self._renderer

    def _step_text_mask_detection(
        self,
        input_image_path: Path,
        text_mask_output: Path
    ) -> Tuple[dict, Path]:
        """
        Step 1: Text Mask Detection (YOLO Segmentation)

        Args:
            input_image_path: Path to input image
            text_mask_output: Output directory for text mask

        Returns:
            Tuple of (layout_result, text_mask_json_path)
        """
        text_mask_output.mkdir(exist_ok=True)

        layout_result = process_single(
            self.yolo_model,
            input_image_path,
            str(text_mask_output)
        )

        # Save layout.json
        text_mask_json_path = text_mask_output / "text_mask.json"
        with open(text_mask_json_path, 'w', encoding='utf-8') as f:
            json.dump([layout_result], f, ensure_ascii=False, indent=2)

        return layout_result, text_mask_json_path

    def _step_text_detection(
        self,
        input_image_path: Path,
        temp_path: Path
    ) -> Tuple[dict, Path]:
        """
        Step 2: Text Detection (PPOCRv5)

        Args:
            input_image_path: Path to input image
            temp_path: Temporary directory path

        Returns:
            Tuple of (text_det_result, text_region_json_path)
        """
        text_det_result = self.text_det_model.predict(str(input_image_path))

        # Save text_region.json
        text_region_json_path = temp_path / "text_region.json"
        with open(text_region_json_path, 'w', encoding='utf-8') as f:
            json.dump([text_det_result], f, ensure_ascii=False, indent=2)

        return text_det_result, text_region_json_path

    def _step_ocr_recognition(
        self,
        layout_result: dict,
        temp_path: Path
    ) -> Tuple[list, Path]:
        """
        Step 3: OCR Recognition (PaddleOCR-VL)

        Args:
            layout_result: Layout result from step 1
            temp_path: Temporary directory path

        Returns:
            Tuple of (ocr_results, ocr_json_path)
        """
        ocr_results = self.ocr_model.process_layout_regions([layout_result])

        # Save ocr.json
        ocr_json_path = temp_path / "ocr.json"
        with open(ocr_json_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)

        return ocr_results, ocr_json_path

    def _step_translation(
        self,
        ocr_results: list,
        temp_path: Path
    ) -> Tuple[list, Path]:
        """
        Step 4: Translation (Hunyuan)

        Args:
            ocr_results: OCR results from step 3
            temp_path: Temporary directory path

        Returns:
            Tuple of (translation_results, translation_json_path)
        """
        translation_results = self.translation_model.translate_ocr_output_batch(ocr_results)

        # Save translation.json
        translation_json_path = temp_path / "translation.json"
        with open(translation_json_path, 'w', encoding='utf-8') as f:
            json.dump(translation_results, f, ensure_ascii=False, indent=2)

        return translation_results, translation_json_path

    def _step_text_allocation(
        self,
        translation_json_path: Path,
        text_region_json_path: Path,
        temp_path: Path
    ) -> Tuple[list, Path]:
        """
        Step 5: Text Region Refinement (Text Allocater)

        Args:
            translation_json_path: Path to translation JSON
            text_region_json_path: Path to text region JSON
            temp_path: Temporary directory path

        Returns:
            Tuple of (allocate_results, text_allocate_output_path)
        """
        text_allocate_output_path = temp_path / "text_allocate.json"
        allocater = TextAllocater(
            translation_json_path=str(translation_json_path),
            text_region_json_path=str(text_region_json_path),
            output_path=str(text_allocate_output_path),
            split_strategy=self.split_strategy
        )
        allocate_results = allocater.run()

        return allocate_results, text_allocate_output_path

    def _step_render(
        self,
        text_mask_json_path: Path,
        text_allocate_output_path: Path,
        temp_path: Path,
        original_image: Image.Image
    ) -> Image.Image:
        """
        Step 6: Render translated text onto image

        Args:
            text_mask_json_path: Path to text mask JSON
            text_allocate_output_path: Path to allocated text JSON
            temp_path: Temporary directory path
            original_image: Original input image (fallback)

        Returns:
            Rendered output image
        """
        render_output_dir = temp_path / "render"
        render_output_dir.mkdir(exist_ok=True)

        self.renderer.render_from_files(
            text_mask_json=str(text_mask_json_path),
            allocated_text_json=str(text_allocate_output_path),
            output_dir=str(render_output_dir)
        )

        # Load rendered image
        rendered_images = list(render_output_dir.glob("*_rendered.jpg"))
        if rendered_images:
            output_image = Image.open(rendered_images[0])
            output_image = output_image.copy()
        else:
            output_image = original_image.copy()

        return output_image

    def translate_image(
        self,
        image: Image.Image,
        progress_callback: Callable[[float, str], None] = None
    ) -> Tuple[Image.Image, dict]:
        """
        Translate a single manga image

        Args:
            image: Input PIL Image
            progress_callback: Optional callback function for progress updates

        Returns:
            Tuple of (translated_image, intermediate_results)
        """
        total_start_time = time.time()
        step_times = {}

        print("=" * 60)
        print("Starting Manga Translation Pipeline")
        print("=" * 60)

        # Create temporary directory for intermediate files (not auto-deleted for debugging)
        temp_dir = tempfile.mkdtemp(dir="ui/tmp")
        temp_path = Path(temp_dir)

        # Save input image
        input_image_path = temp_path / "input.jpg"
        image.save(str(input_image_path), quality=95)

        results = {}

        # Step 1: Text Mask Detection (YOLO Segmentation)
        if progress_callback:
            progress_callback(0.1, "Step 1/6: Detecting text regions...")

        step_start = time.time()
        text_mask_output = temp_path / "text_mask"
        layout_result, text_mask_json_path = self._step_text_mask_detection(
            input_image_path, text_mask_output
        )
        results['layout'] = layout_result
        step_times['1_text_mask_detection'] = time.time() - step_start
        print(f"[Step 1/6] Text Mask Detection: {step_times['1_text_mask_detection']:.2f}s")

        # Step 2: Text Detection (PPOCRv5)
        if progress_callback:
            progress_callback(0.25, "Step 2/6: Detecting text bounding boxes...")

        step_start = time.time()
        text_det_result, text_region_json_path = self._step_text_detection(
            input_image_path, temp_path
        )
        results['text_detection'] = text_det_result
        step_times['2_text_detection'] = time.time() - step_start
        print(f"[Step 2/6] Text Detection: {step_times['2_text_detection']:.2f}s")

        # Step 3: OCR Recognition
        if progress_callback:
            progress_callback(0.4, "Step 3/6: Recognizing text with OCR...")

        step_start = time.time()
        ocr_results, ocr_json_path = self._step_ocr_recognition(
            layout_result, temp_path
        )
        results['ocr'] = ocr_results
        step_times['3_ocr_recognition'] = time.time() - step_start
        print(f"[Step 3/6] OCR Recognition: {step_times['3_ocr_recognition']:.2f}s")

        # Step 4: Translation
        if progress_callback:
            progress_callback(0.55, "Step 4/6: Translating text...")

        step_start = time.time()
        translation_results, translation_json_path = self._step_translation(
            ocr_results, temp_path
        )
        results['translation'] = translation_results
        step_times['4_translation'] = time.time() - step_start
        print(f"[Step 4/6] Translation: {step_times['4_translation']:.2f}s")

        # Step 5: Text Region Refinement (Text Allocater)
        if progress_callback:
            progress_callback(0.7, "Step 5/6: Refining text regions...")

        step_start = time.time()
        allocate_results, text_allocate_output_path = self._step_text_allocation(
            translation_json_path, text_region_json_path, temp_path
        )
        results['allocate'] = allocate_results
        step_times['5_text_allocation'] = time.time() - step_start
        print(f"[Step 5/6] Text Allocation: {step_times['5_text_allocation']:.2f}s")

        # Step 6: Render
        if progress_callback:
            progress_callback(0.85, "Step 6/6: Rendering translated image...")

        step_start = time.time()
        output_image = self._step_render(
            text_mask_json_path, text_allocate_output_path, temp_path, image
        )
        step_times['6_render'] = time.time() - step_start
        print(f"[Step 6/6] Render: {step_times['6_render']:.2f}s")

        # Calculate total time
        total_time = time.time() - total_start_time
        results['step_times'] = step_times
        results['total_time'] = total_time

        # Print summary
        print("=" * 60)
        print("Pipeline Summary")
        print("-" * 60)
        for step_name, step_time in step_times.items():
            percentage = (step_time / total_time) * 100
            print(f"  {step_name}: {step_time:.2f}s ({percentage:.1f}%)")
        print("-" * 60)
        print(f"  TOTAL TIME: {total_time:.2f}s")
        print("=" * 60)

        if progress_callback:
            progress_callback(1.0, f"Complete! Total time: {total_time:.2f}s")

        return output_image, results


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
