"""
日文到繁體中文的翻譯模型 (使用 Google Translate API)
"""

from typing import List, Optional, Dict, Any
import json
import argparse
import os
from tqdm import tqdm

# 導入 schema
try:
    from utils.schema import (
        OcrOutputSchema,
        BoundingBox,
        TranslationOutputSchema,
        TranslatedBoundingBox,
        TranslationErrorSchema
    )
except ImportError:
    from schema import (
        OcrOutputSchema,
        BoundingBox,
        TranslationOutputSchema,
        TranslatedBoundingBox,
        TranslationErrorSchema
    )

try:
    from googletrans import Translator
except ImportError:
    print("警告: 未安裝 googletrans，請執行: pip install googletrans==4.0.0-rc1")
    raise


class Translation_Model_Google:
    """
    使用 Google Translate API 將日文文字轉換為繁體中文的翻譯模型
    """

    def __init__(self):
        """
        初始化 Google 翻譯模型
        """
        print("正在初始化 Google Translate API...")
        self.translator = Translator()
        print("Google Translate API 初始化成功！")

    def translate(self, text: str) -> str:
        """
        將單一日文文字翻譯為繁體中文

        Args:
            text: 要翻譯的日文文字

        Returns:
            翻譯後的繁體中文文字
        """
        if not text or not text.strip():
            return ""

        try:
            # 使用 Google Translate API 翻譯
            # src='ja' 表示日文，dest='zh-tw' 表示繁體中文
            result = self.translator.translate(text, src='ja', dest='zh-tw')
            return result.text
        except Exception as e:
            print(f"翻譯錯誤: {e}")
            return text  # 如果翻譯失敗，返回原文

    def batch_translate(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[str]:
        """
        批次翻譯多個日文文字為繁體中文

        Args:
            texts: 要翻譯的日文文字列表
            batch_size: 批次大小（Google API 建議不要太大）

        Returns:
            翻譯後的繁體中文文字列表
        """
        if not texts:
            return []

        translated_texts = []

        # 分批處理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                if text and text.strip():
                    translated = self.translate(text)
                    translated_texts.append(translated)
                else:
                    translated_texts.append("")

        return translated_texts

    def translate_ocr_output(
        self,
        ocr_data: OcrOutputSchema
    ) -> TranslationOutputSchema:
        """
        翻譯單個 OCR 輸出並返回標準格式

        Args:
            ocr_data: OCR 輸出的資料

        Returns:
            符合 TranslationOutputSchema 的翻譯結果
        """
        # 翻譯主要文字
        main_translated_text = self.translate(ocr_data["text"]) if ocr_data["text"] else ""

        # 翻譯所有 bounding boxes
        translated_bboxes: List[TranslatedBoundingBox] = []
        for bbox in ocr_data["bounding_boxes"]:
            bbox_text = bbox.get("text", "")
            translated_bbox_text = self.translate(bbox_text) if bbox_text else ""

            translated_bbox: TranslatedBoundingBox = {
                "box": bbox["box"],
                "text": bbox["text"],
                "score": bbox.get("score"),
                "translated_text": translated_bbox_text
            }
            translated_bboxes.append(translated_bbox)

        # 構造輸出
        output: TranslationOutputSchema = {
            "source": ocr_data["source"],
            "text": ocr_data["text"],
            "translated_text": main_translated_text,
            "image_path": ocr_data["image_path"],
            "bounding_boxes": translated_bboxes
        }

        return output


def main():
    """
    從 JSON 翻譯 OCR 結果的主函式

    接受包含 bounding_boxes 的 OCR 輸出格式，並為每個 box 加上 translated_text
    輸出格式符合 TranslationOutputSchema

    使用方式:
        python utils/translation_google.py --input_path results/2/ocr.json --output_path results/2/translated.json
    """
    parser = argparse.ArgumentParser(description="日文到繁體中文翻譯 (使用 Google Translate API)")
    parser.add_argument("--input_path", type=str, default="results/2/ocr.json",
                       help="包含 OCR 結果的輸入 JSON 檔案路徑")
    parser.add_argument("--output_path", type=str, default="results/2/translated.json",
                       help="翻譯結果的輸出 JSON 檔案路徑")

    args = parser.parse_args()

    # 檢查輸入檔案是否存在
    if not os.path.exists(args.input_path):
        print(f"錯誤: 找不到輸入檔案: {args.input_path}")
        return

    # 載入輸入 JSON
    print(f"載入輸入 JSON: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        ocr_data_list: List[OcrOutputSchema] = json.load(f)

    # 初始化翻譯模型
    print("\n初始化 Google 翻譯模型...")
    translator = Translation_Model_Google()
    print("模型載入成功！")

    # 處理 OCR 資料
    print(f"\n處理 OCR 資料...")
    total_boxes = sum(len(entry.get("bounding_boxes", [])) for entry in ocr_data_list)
    print(f"找到 {len(ocr_data_list)} 個條目，共 {total_boxes} 個邊界框")

    # 使用 schema-based 方法翻譯每個 OCR 輸出
    translation_results: List[TranslationOutputSchema] = []

    print(f"\n翻譯文字中...")
    for ocr_entry in tqdm(ocr_data_list, desc="翻譯條目", unit="條目"):
        try:
            # 使用新的 translate_ocr_output 方法
            translated_entry = translator.translate_ocr_output(ocr_entry)
            translation_results.append(translated_entry)
        except Exception as e:
            # 如果翻譯失敗，記錄錯誤
            error_entry: TranslationErrorSchema = {
                "source": ocr_entry.get("source", "unknown"),
                "error": str(e)
            }
            print(f"\n警告: 翻譯失敗 - {error_entry['source']}: {error_entry['error']}")
            # 仍然加入一個包含原始資料的條目，但標記為錯誤
            fallback_entry: TranslationOutputSchema = {
                "source": ocr_entry["source"],
                "text": ocr_entry["text"],
                "translated_text": f"[翻譯錯誤: {str(e)}]",
                "image_path": ocr_entry["image_path"],
                "bounding_boxes": [
                    {
                        "box": bbox["box"],
                        "text": bbox["text"],
                        "score": bbox.get("score"),
                        "translated_text": "[翻譯錯誤]"
                    }
                    for bbox in ocr_entry["bounding_boxes"]
                ]
            }
            translation_results.append(fallback_entry)

    # 儲存輸出 JSON
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(translation_results, f, ensure_ascii=False, indent=2)

    # 統計資訊
    total_translated_texts = sum(len(entry["bounding_boxes"]) + 1 for entry in translation_results)

    print("\n" + "="*50)
    print(f"翻譯完成！")
    print(f"結果已儲存至: {args.output_path}")
    print(f"處理的條目總數: {len(translation_results)}")
    print(f"翻譯的文字總數: {total_translated_texts}")
    print("="*50)


if __name__ == "__main__":
    main()
