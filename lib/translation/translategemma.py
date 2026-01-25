"""
日文到繁體中文的翻譯模型 (使用 TranslateGemma-4B)
"""

from typing import List, Dict
import json
import argparse
import os
import opencc
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from lib.schema import (
    OcrOutputSchema,
    TranslationOutputSchema,
    TranslatedBoundingBox,
    TranslationErrorSchema
)


class Translation_Model_Gemma:
    """
    使用 TranslateGemma-4B 模型將日文文字轉換為繁體中文的翻譯模型
    """

    def __init__(
        self,
        model_path: str = "/tmp2/share_data/google-translategemma-4b-it",
        max_new_tokens: int = 200,
        device: str = "cuda",
        dtype=torch.bfloat16,
        load_in_4bit: bool = False
    ):
        """
        初始化 TranslateGemma 翻譯模型

        Args:
            model_path: 模型路徑
            max_new_tokens: 最大生成 token 數
            device: 設備 (cuda 或 cpu)
            dtype: 資料型別
            load_in_4bit: 是否使用 4-bit 量化載入模型
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit

        print(f"正在載入 TranslateGemma 翻譯模型: {model_path}")
        if load_in_4bit:
            print("使用 4-bit 量化載入模型")

        # 載入 tokenizer
        print("載入 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            torch_dtype=dtype,
            device_map="auto"  # 自動分配設備
        )

        self.model.eval()  # 設定為評估模式
        print("模型載入成功！")

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

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "ja",
                        "target_lang_code": "zh",
                        "text": text
                    }
                ],
            }
        ]

        # 使用 tokenizer 的 apply_chat_template 處理訊息
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize 輸入
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True
        )

        # 將輸入移到正確的設備
        if not self.load_in_4bit:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # 4-bit 模型已經在正確的設備上
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 生成翻譯
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False  # 使用貪婪解碼以獲得確定性結果
            )

        # 解碼輸出
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # 提取翻譯結果（移除輸入提示部分）
        # TranslateGemma 的輸出格式通常是在最後
        if input_text in generated_text:
            translated_text = generated_text.replace(input_text, "").strip()
        else:
            translated_text = generated_text.strip()

        return translated_text

    def twcc_polish(self, raw_output: str) -> str:
        """
        將簡體中文轉換為繁體中文並進行格式化

        Args:
            raw_output: 模型的原始輸出文字（簡體中文）

        Returns:
            轉換後的繁體中文文字
        """
        cc = opencc.OpenCC('s2tw')
        translated_text = cc.convert(raw_output)
        return translated_text.strip()

    def translate_with_parse(self, text: str) -> str:
        """
        將單一日文文字翻譯為繁體中文，並解析輸出以獲取純翻譯內容

        Args:
            text: 要翻譯的日文文字

        Returns:
            翻譯後的繁體中文文字
        """
        parsed_output = self.translate(text)
        parsed_output = self.twcc_polish(parsed_output)
        return parsed_output.strip()

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
        # 翻譯所有 bounding boxes
        translated_bboxes: List[TranslatedBoundingBox] = []
        for bbox in ocr_data["bounding_boxes"]:
            bbox_text = bbox.get("text", "")
            translated_bbox_text = self.translate_with_parse(bbox_text) if bbox_text else ""

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
            "translated_text": None,
            "image_path": ocr_data["image_path"],
            "bounding_boxes": translated_bboxes
        }

        return output

    def translate_ocr_output_batch(
        self,
        ocr_data_list: List[OcrOutputSchema]
    ) -> List[TranslationOutputSchema]:
        """
        批次翻譯多個 OCR 輸出並返回標準格式列表

        Args:
            ocr_data_list: OCR 輸出的資料列表

        Returns:
            符合 TranslationOutputSchema 的翻譯結果列表
        """
        translation_results: List[TranslationOutputSchema] = []

        print(f"\n翻譯文字中...")
        for ocr_entry in tqdm(ocr_data_list, desc="翻譯條目", unit="條目"):
            try:
                # 使用 translate_ocr_output 方法
                translated_entry = self.translate_ocr_output(ocr_entry)
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
        return translation_results

    def translate_ocr_output_multi_text(
        self,
        ocr_data_list: List[OcrOutputSchema],
        batch_size: int = 10
    ) -> List[TranslationOutputSchema]:
        """
        使用批次翻譯處理 OCR 輸出並返回標準格式

        input: 多張圖片、每張圖片可能有多個 ocr 結果

        # Format: List[OcrOutputSchema]

        img1
        - {ocr_text1}
        - {ocr_text2}
        img2
        - ...

        output: 讓每個 ocr 結果都加上翻譯

        Steps:
        1. 收集所有 ocr text (bbox_texts: List[str])
        2. 分批翻譯所有 ocr text (all_translated_texts: List[str])
        3. 建立 ocr id 到 translated text 的映射 (ocrid2translatedtext: Dict[str, str])
        4. 重建資料結構並存回原架構:
           - 遍歷每個 ocr_data
           - 為每個 bounding box 建立 TranslatedBoundingBox
           - 組裝成 TranslationOutputSchema
           - 加入 translation_data_list

        Args:
            ocr_data_list: OCR 輸出的資料列表
            batch_size: 每批次處理的 bounding box 數量，預設為 10

        Returns:
            符合 TranslationOutputSchema 的翻譯結果列表
        """
        # Step 1: 收集所有 ocr text
        bbox_texts: List[str] = []
        for ocr_data in ocr_data_list:
            for bbox in ocr_data["bounding_boxes"]:
                bbox_texts.append(bbox.get("text", ""))

        # Step 2: 分批翻譯所有 ocr text
        all_translated_texts: List[str] = []
        print(f"\n翻譯文字中...")
        for i in tqdm(range(0, len(bbox_texts), batch_size), desc="批次翻譯", unit="批次"):
            batch = bbox_texts[i:i + batch_size]

            # 過濾空字串並記錄索引
            non_empty_indices = [j for j, text in enumerate(batch) if text.strip()]
            non_empty_texts = [batch[j] for j in non_empty_indices]

            if non_empty_texts:
                # 批次翻譯非空文字
                translated_batch = self.translate_multi_text(non_empty_texts)

                # 重建完整批次，空字串保持為空
                full_batch_translations = [""] * len(batch)
                for j, translated_text in zip(non_empty_indices, translated_batch):
                    full_batch_translations[j] = translated_text

                all_translated_texts.extend(full_batch_translations)
            else:
                # 整批都是空字串
                all_translated_texts.extend([""] * len(batch))

        # Step 3: 存回原架構
        translation_data_list: List[TranslationOutputSchema] = []

        # 建立 ocr id 到 translated text 的映射
        ocrid2translatedtext: Dict[str, str] = {}
        text_idx = 0
        for i, ocr_data in enumerate(ocr_data_list):
            for j in range(len(ocr_data["bounding_boxes"])):
                ocrid2translatedtext[f"{i}_{j}"] = all_translated_texts[text_idx]
                text_idx += 1

        # Step 3a-3g: 重建資料結構
        for i, ocr_data in enumerate(ocr_data_list):
            # Step 3aa: 建立翻譯資料
            translation_data: List[TranslatedBoundingBox] = []

            # Step 3b: 遍歷每個 bounding box
            for j, ocr_instance in enumerate(ocr_data["bounding_boxes"]):
                # Step 3c: 取得翻譯文字
                translated_text = ocrid2translatedtext[f"{i}_{j}"]

                # Step 3d: 建立翻譯後的 bounding box
                translation_data.append({
                    "box": ocr_instance["box"],
                    "text": ocr_instance["text"],
                    "score": ocr_instance.get("score"),
                    "translated_text": translated_text
                })

            # Step 3e-3f: 建立 TranslationOutputSchema
            tmp: TranslationOutputSchema = {
                "source": ocr_data["source"],
                "text": ocr_data["text"],
                "translated_text": None,  # 若需要可以合併所有翻譯文字
                "image_path": ocr_data["image_path"],
                "bounding_boxes": translation_data
            }

            # Step 3g: 加入結果列表
            translation_data_list.append(tmp)

        return translation_data_list

    def translate_multi_text(self, texts: List[str]) -> List[str]:
        """
        批次翻譯多個日文文字為繁體中文

        Args:
            texts: 要翻譯的日文文字列表

        Returns:
            翻譯後的繁體中文文字列表
        """
        if not texts:
            return []

        # 逐個翻譯每個文字 (TranslateGemma 不支援批次處理，所以逐個處理)
        results = []
        for text in texts:
            translated_text = self.translate_with_parse(text) if text.strip() else ""
            results.append(translated_text)

        return results


def main():
    """
    從 JSON 翻譯 OCR 結果的主函式

    接受包含 bounding_boxes 的 OCR 輸出格式，並為每個 box 加上 translated_text
    輸出格式符合 TranslationOutputSchema

    使用方式:
        python lib/translation/translategemma.py --input_path ocr.json --output_path translated.json --model_path path/to/model
    """
    parser = argparse.ArgumentParser(description="日文到繁體中文翻譯 (使用 TranslateGemma-4B)")
    parser.add_argument("--input_path", type=str, default="results/2/ocr.json",
                       help="包含 OCR 結果的輸入 JSON 檔案路徑")
    parser.add_argument("--output_path", type=str, default="results/2/translated_gemma.json",
                       help="翻譯結果的輸出 JSON 檔案路徑")
    parser.add_argument("--model_path", type=str, default="/tmp2/share_data/google-translategemma-4b-it",
                       help="TranslateGemma 模型路徑")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                       help="最大生成 token 數")
    parser.add_argument("--device", type=str, default="cuda",
                       help="設備 (cuda 或 cpu)")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi"],
                       help="翻譯模式: single (逐個翻譯) 或 multi (批次翻譯)")
    parser.add_argument("--batch_size", type=int, default=3,
                       help="批次翻譯模式下每批次處理的 bounding box 數量")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="使用 4-bit 量化載入模型以節省記憶體")

    args = parser.parse_args()

    # 檢查輸入檔案是否存在
    if not os.path.exists(args.input_path):
        print(f"錯誤: 找不到輸入檔案: {args.input_path}")
        return

    if not os.path.exists(args.model_path):
        print(f"錯誤: 找不到模型路徑: {args.model_path}")
        return

    # 載入輸入 JSON
    print(f"載入輸入 JSON: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        ocr_data_list: List[OcrOutputSchema] = json.load(f)

    # 初始化翻譯模型
    print("\n初始化 TranslateGemma 翻譯模型...")
    translator = Translation_Model_Gemma(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        load_in_4bit=args.load_in_4bit
    )
    print("模型載入成功！")

    # 處理 OCR 資料
    print(f"\n處理 OCR 資料...")
    total_boxes = sum(len(entry.get("bounding_boxes", [])) for entry in ocr_data_list)
    print(f"找到 {len(ocr_data_list)} 個條目，共 {total_boxes} 個邊界框")

    # 翻譯每個 OCR 輸出
    if args.mode == "multi":
        print(f"使用批次翻譯模式 (batch_size={args.batch_size})")
        translation_results = translator.translate_ocr_output_multi_text(ocr_data_list, batch_size=args.batch_size)
    else:
        print("使用逐個翻譯模式")
        translation_results = translator.translate_ocr_output_batch(ocr_data_list)

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
