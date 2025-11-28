"""
日文到繁體中文的翻譯模型
"""

from typing import List, Optional, Union
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import argparse
import os
from tqdm import tqdm
from schema import OcrOutputSchema, BoundingBox


class Translation_Model:
    """
    使用 NLLB-200 模型將日文文字轉換為繁體中文的翻譯模型
    """

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: Optional[str] = None):
        """
        初始化翻譯模型

        Args:
            model_name: 要使用的翻譯模型名稱
            device: 執行模型的裝置 ('cuda', 'cpu', 或 None 表示自動選擇)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

        # NLLB 語言代碼
        self.src_lang = "jpn_Jpan"  # 日文
        self.tgt_lang = "zho_Hant"  # 繁體中文

        self._load_model()

    def _load_model(self):
        """載入翻譯模型和分詞器"""
        print(f"正在載入翻譯模型: {self.model_name}")

        # 載入分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang
        )

        # 載入模型
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name
        ).to(self.device)

        print(f"模型已載入到裝置: {self.device}")

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

        # 分詞輸入
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # 生成翻譯
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
            max_length=512
        )

        # 解碼輸出
        translated_text = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]

        return translated_text

    def batch_translate(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        批次翻譯多個日文文字為繁體中文

        Args:
            texts: 要翻譯的日文文字列表
            batch_size: 處理的批次大小

        Returns:
            翻譯後的繁體中文文字列表
        """
        if not texts:
            return []

        translated_texts = []

        # 為了效率，批次處理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_translations = self._translate_batch(batch)
            translated_texts.extend(batch_translations)

        return translated_texts

    def _translate_batch(self, batch: List[str]) -> List[str]:
        """
        翻譯一批文字

        Args:
            batch: 日文文字列表

        Returns:
            翻譯後的文字列表
        """
        if not batch:
            return []

        # 過濾空字串
        non_empty_batch = [text for text in batch if text and text.strip()]
        if not non_empty_batch:
            return ["" for _ in batch]

        # 批次分詞
        inputs = self.tokenizer(
            non_empty_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # 生成翻譯
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
            max_length=512
        )

        # 解碼輸出
        translated_texts = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )

        # 對應回原始批次（處理空字串）
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
    從 JSON 翻譯 OCR 結果的主函式

    接受包含 bounding_boxes 的 OCR 輸出格式，並為每個 box 加上 translated_text

    使用方式:
        python utils/translation_model.py --input_path ocr.json --output_path translated.json
    """
    parser = argparse.ArgumentParser(description="日文到繁體中文翻譯")
    parser.add_argument("--input_path", type=str, default="ocr.json", help="包含 OCR 結果的輸入 JSON 檔案路徑")
    parser.add_argument("--output_path", type=str, default="translated.json", help="翻譯結果的輸出 JSON 檔案路徑")
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-3.3B",
                       help="翻譯模型名稱")
    parser.add_argument("--batch_size", type=int, default=8, help="翻譯批次大小")

    args = parser.parse_args()

    # 檢查輸入檔案是否存在
    if not os.path.exists(args.input_path):
        print(f"錯誤: 找不到輸入檔案: {args.input_path}")
        return

    # 載入輸入 JSON
    print(f"載入輸入 JSON: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 初始化翻譯模型
    print("\n初始化翻譯模型...")
    translator = Translation_Model(model_name=args.model_name)
    print("模型載入成功！")

    # 處理 OCR 資料
    print(f"\n處理 OCR 資料...")

    # 收集所有需要翻譯的文字（來自 bounding boxes 和主要文字欄位）
    all_texts_to_translate = []
    text_indices = []  # 追蹤每個文字屬於哪個 entry 和 bbox
    total_boxes = sum(len(entry.get("bounding_boxes", [])) for entry in data)

    for entry_idx, entry in enumerate(data):
        # 加入主要文字欄位
        main_text = entry.get("text", "")
        if main_text:
            all_texts_to_translate.append(main_text)
            text_indices.append(("main", entry_idx, None))

        # 加入所有 bounding box 的文字
        for bbox_idx, bbox in enumerate(entry.get("bounding_boxes", [])):
            bbox_text = bbox.get("text", "")
            if bbox_text:
                all_texts_to_translate.append(bbox_text)
                text_indices.append(("bbox", entry_idx, bbox_idx))

    print(f"找到 {len(data)} 個條目，共 {total_boxes} 個邊界框")
    print(f"總共需要翻譯的文字: {len(all_texts_to_translate)}")

    # 批次翻譯所有文字
    print(f"\n以批次大小 {args.batch_size} 翻譯文字...")
    translated_texts = []
    for i in tqdm(range(0, len(all_texts_to_translate), args.batch_size), desc="翻譯中", unit="批次"):
        batch = all_texts_to_translate[i:i + args.batch_size]
        batch_translations = translator.batch_translate(batch, batch_size=args.batch_size)
        translated_texts.extend(batch_translations)

    # 將翻譯對應回原始結構
    output_data = json.loads(json.dumps(data))  # 深度複製

    for idx, (text_type, entry_idx, bbox_idx) in enumerate(text_indices):
        if text_type == "main":
            output_data[entry_idx]["translated_text"] = translated_texts[idx]
        elif text_type == "bbox":
            output_data[entry_idx]["bounding_boxes"][bbox_idx]["translated_text"] = translated_texts[idx]

    # 儲存輸出 JSON
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print(f"翻譯完成！")
    print(f"結果已儲存至: {args.output_path}")
    print(f"處理的條目總數: {len(output_data)}")
    print(f"翻譯的文字總數: {len(translated_texts)}")
    print("="*50)


if __name__ == "__main__":
    main()
