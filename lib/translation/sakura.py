"""
日文到繁體中文的翻譯模型 (使用 SakuraLLM)
"""

from typing import List, Optional, Dict, Any
import json
import argparse
import os
from tqdm import tqdm
from schema import (
    OcrOutputSchema,
    BoundingBox,
    TranslationOutputSchema,
    TranslatedBoundingBox,
    TranslationErrorSchema
)

try:
    from llama_cpp import Llama
except ImportError:
    print("警告: 未安裝 llama-cpp-python，請執行: pip install llama-cpp-python")
    raise

try:
    from opencc import OpenCC
except ImportError:
    print("警告: 未安裝 opencc-python-reimplemented，請執行: pip install opencc-python-reimplemented")
    raise


class Translation_Model_Sakura:
    """
    使用 SakuraLLM (GGUF) 模型將日文文字轉換為繁體中文的翻譯模型
    專門針對輕小說和視覺小說優化
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.3,
        max_tokens: int = 512,
        frequency_penalty: float = 0.1,
    ):
        """
        初始化 Sakura 翻譯模型

        Args:
            model_path: GGUF 模型檔案的路徑
            n_gpu_layers: 要載入到 GPU 的層數 (-1 表示全部)
            n_ctx: 上下文視窗大小
            temperature: 溫度參數 (建議 0.1)
            top_p: top_p 參數 (建議 0.3)
            max_tokens: 最大生成 token 數
            frequency_penalty: 頻率懲罰 (0.1-0.2)
        """
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty

        print(f"正在載入 Sakura 翻譯模型: {model_path}")

        # 載入 GGUF 模型
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

        # 系統提示詞 (繁體中文版本)
        self.system_prompt = "你是一個輕小說翻譯模型，可以流畅通順地以日本輕小說的風格將日文翻譯成繁體中文，並聯繫上下文正確使用人稱代詞，不擅自添加原文中沒有的代詞。"

        # 初始化 OpenCC 轉換器 (簡體轉繁體)
        self.opencc_converter = OpenCC('s2t')

        print("模型載入成功！")

    def _build_prompt(self, japanese_text: str, gpt_dict: Optional[List[Dict[str, str]]] = None) -> str:
        """
        建構 Sakura 模型的提示詞

        Args:
            japanese_text: 要翻譯的日文文字
            gpt_dict: 術語表 (可選)，格式: [{"src": "原文", "dst": "譯文", "info": "註釋"}]

        Returns:
            完整的提示詞
        """
        # 建構術語表文字
        gpt_dict_text = ""
        if gpt_dict:
            gpt_dict_text_list = []
            for entry in gpt_dict:
                src = entry['src']
                dst = entry['dst']
                info = entry.get('info')
                if info:
                    single = f"{src}->{dst} #{info}"
                else:
                    single = f"{src}->{dst}"
                gpt_dict_text_list.append(single)
            gpt_dict_text = "\n".join(gpt_dict_text_list)

        # 建構用戶提示詞
        if gpt_dict_text:
            user_prompt = f"根據以下術語表（可以為空）：\n{gpt_dict_text}\n將下面的日文文本根據對應關係和備註翻譯成繁體中文：{japanese_text}"
        else:
            user_prompt = f"將下面的日文文本翻譯成繁體中文：{japanese_text}"

        # 使用 ChatML 格式
        prompt = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return prompt

    def translate(self, text: str, gpt_dict: Optional[List[Dict[str, str]]] = None) -> str:
        """
        將單一日文文字翻譯為繁體中文

        Args:
            text: 要翻譯的日文文字
            gpt_dict: 術語表 (可選)

        Returns:
            翻譯後的繁體中文文字
        """
        if not text or not text.strip():
            return ""

        # 建構提示詞
        prompt = self._build_prompt(text, gpt_dict)

        # 生成翻譯
        output = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            stop=["<|im_end|>", "\n\n"],
            echo=False,
        )

        # 提取翻譯文字
        translated_text = output['choices'][0]['text'].strip()

        return translated_text

    def simplify_to_traditional(self, text: str) -> str:
        """
        將簡體中文轉換為繁體中文

        Args:
            text: 要轉換的簡體中文文字

        Returns:
            轉換後的繁體中文文字
        """
        if not text or not text.strip():
            return ""

        return self.opencc_converter.convert(text)

    def batch_translate(
        self,
        texts: List[str],
        batch_size: int = 1,
        gpt_dict: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        批次翻譯多個日文文字為繁體中文

        注意: GGUF 模型通常不支援真正的批次處理，所以這裡是順序處理

        Args:
            texts: 要翻譯的日文文字列表
            batch_size: 此參數保留以相容介面，但不使用
            gpt_dict: 術語表 (可選)

        Returns:
            翻譯後的繁體中文文字列表
        """
        if not texts:
            return []

        translated_texts = []

        for text in texts:
            if text and text.strip():
                translated = self.translate(text, gpt_dict)
                translated_texts.append(translated)
            else:
                translated_texts.append("")

        return translated_texts

    def translate_ocr_output(
        self,
        ocr_data: OcrOutputSchema,
        gpt_dict: Optional[List[Dict[str, str]]] = None
    ) -> TranslationOutputSchema:
        """
        翸譯單個 OCR 輸出並返回標準格式

        Args:
            ocr_data: OCR 輸出的資料
            gpt_dict: 術語表 (可選)

        Returns:
            符合 TranslationOutputSchema 的翻譯結果
        """
        # 翻譯主要文字
        main_translated_text = self.translate(ocr_data["text"], gpt_dict) if ocr_data["text"] else ""

        # 翻譯所有 bounding boxes
        translated_bboxes: List[TranslatedBoundingBox] = []
        for bbox in ocr_data["bounding_boxes"]:
            bbox_text = bbox.get("text", "")
            translated_bbox_text = self.translate(bbox_text, gpt_dict) if bbox_text else ""
            translated_bbox_text = self.simplify_to_traditional(translated_bbox_text)

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
        python utils/translation_model_sakura.py --input_path ocr.json --output_path translated.json --model_path path/to/model.gguf
    """
    parser = argparse.ArgumentParser(description="日文到繁體中文翻譯 (使用 SakuraLLM)")
    parser.add_argument("--input_path", type=str, default="results/2/ocr.json", help="包含 OCR 結果的輸入 JSON 檔案路徑")
    parser.add_argument("--output_path", type=str, default="results/2/translated_sakura.json", help="翻譯結果的輸出 JSON 檔案路徑")
    parser.add_argument("--model_path", type=str, default="model/sakura-1.5b-qwen2.5-v1.0-fp16.gguf",
                       help="GGUF 模型檔案路徑")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                       help="要載入到 GPU 的層數 (-1 表示全部)")
    parser.add_argument("--n_ctx", type=int, default=2048,
                       help="上下文視窗大小")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="溫度參數")
    parser.add_argument("--top_p", type=float, default=0.3,
                       help="top_p 參數")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="最大生成 token 數")
    parser.add_argument("--frequency_penalty", type=float, default=0.1,
                       help="頻率懲罰")
    parser.add_argument("--glossary", type=str, default=None,
                       help="術語表 JSON 檔案路徑 (可選)")

    args = parser.parse_args()

    # 檢查輸入檔案是否存在
    if not os.path.exists(args.input_path):
        print(f"錯誤: 找不到輸入檔案: {args.input_path}")
        return

    if not os.path.exists(args.model_path):
        print(f"錯誤: 找不到模型檔案: {args.model_path}")
        return

    # 載入術語表 (如果提供)
    gpt_dict = None
    if args.glossary and os.path.exists(args.glossary):
        print(f"載入術語表: {args.glossary}")
        with open(args.glossary, "r", encoding="utf-8") as f:
            gpt_dict = json.load(f)

    # 載入輸入 JSON
    print(f"載入輸入 JSON: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        ocr_data_list: List[OcrOutputSchema] = json.load(f)

    # 初始化翻譯模型
    print("\n初始化 Sakura 翻譯模型...")
    translator = Translation_Model_Sakura(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        frequency_penalty=args.frequency_penalty,
    )
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
            translated_entry = translator.translate_ocr_output(ocr_entry, gpt_dict)
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
