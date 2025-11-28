"""
Translation Model for Japanese to Traditional Chinese translation.
"""

from typing import List, Optional, Union
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import argparse
import os
from tqdm import tqdm


class Translation_Model:
    """
    Translation model for converting Japanese text to Traditional Chinese using NLLB-200.
    """

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: Optional[str] = None):
        """
        Initialize the translation model.

        Args:
            model_name: Name of the translation model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

        # NLLB language codes
        self.src_lang = "jpn_Jpan"  # Japanese
        self.tgt_lang = "zho_Hant"  # Traditional Chinese

        self._load_model()

    def _load_model(self):
        """Load the translation model and tokenizer."""
        print(f"Loading translation model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang
        )

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name
        ).to(self.device)

        print(f"Model loaded on device: {self.device}")

    def translate(self, text: str) -> str:
        """
        Translate a single Japanese text to Traditional Chinese.

        Args:
            text: Japanese text to translate

        Returns:
            Translated Traditional Chinese text
        """
        if not text or not text.strip():
            return ""

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Generate translation
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
            max_length=512
        )

        # Decode output
        translated_text = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]

        return translated_text

    def batch_translate(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Translate multiple Japanese texts to Traditional Chinese.

        Args:
            texts: List of Japanese texts to translate
            batch_size: Batch size for processing

        Returns:
            List of translated Traditional Chinese texts
        """
        if not texts:
            return []

        translated_texts = []

        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_translations = self._translate_batch(batch)
            translated_texts.extend(batch_translations)

        return translated_texts

    def _translate_batch(self, batch: List[str]) -> List[str]:
        """
        Translate a batch of texts.

        Args:
            batch: List of Japanese texts

        Returns:
            List of translated texts
        """
        if not batch:
            return []

        # Filter out empty strings
        non_empty_batch = [text for text in batch if text and text.strip()]
        if not non_empty_batch:
            return ["" for _ in batch]

        # Tokenize batch
        inputs = self.tokenizer(
            non_empty_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate translations
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            max_length=512
        )

        # Decode outputs
        translated_texts = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )

        # Map back to original batch (handling empty strings)
        result = []
        translated_idx = 0
        for text in batch:
            if text and text.strip():
                result.append(translated_texts[translated_idx])
                translated_idx += 1
            else:
                result.append("")

        return result


def main():
    """
    Main function to translate OCR results from JSON

    Usage:
        python utils/translation_model.py --input_path ocr.json --output_path translated.json
    """
    parser = argparse.ArgumentParser(description="Japanese to Traditional Chinese Translation")
    parser.add_argument("--input_path", type=str, default="ocr.json", help="Path to input JSON file with OCR results")
    parser.add_argument("--output_path", type=str, default="translated.json", help="Path to output JSON file with translations")
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-3.3B",
                       help="Translation model name")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        return

    # Load input JSON
    print(f"Loading input JSON: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} entries")

    # Initialize translation model
    print("\nInitializing translation model...")
    translator = Translation_Model(model_name=args.model_name)
    print("Model loaded successfully!")

    # Translate texts with progress bar
    print(f"\nTranslating {len(data)} texts...")
    output_data = {}

    for image_path, text_info in tqdm(data.items(), desc="Translating", unit="text"):
        text = text_info.get("text", "")
        translated_text = translator.translate(text)
        output_data[image_path] = {
            "original": text,
            "translated": translated_text
        }

    # Save output JSON
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print(f"Translation completed!")
    print(f"Results saved to: {args.output_path}")
    print(f"Total entries translated: {len(output_data)}")
    print("="*50)


if __name__ == "__main__":
    main()
