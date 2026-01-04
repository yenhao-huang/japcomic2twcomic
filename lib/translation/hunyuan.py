"""
日文到繁體中文的翻譯模型 (使用 Hunyuan-MT-7B)
"""

from typing import List
import json
import argparse
import os
import opencc
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from lib.schema import (
    OcrOutputSchema,
    TranslationOutputSchema,
    TranslatedBoundingBox,
    TranslationErrorSchema
)


class Translation_Model_Hunyuan:
    """
    使用 Hunyuan-MT-7B 模型將日文文字轉換為繁體中文的翻譯模型
    """

    DEFAULT_PROMPT_CONFIG = "configs/translation_prompt/default.yml"

    def __init__(
        self,
        model_path: str = "/tmp2/share-data/Hunyuan-MT-7B",
        max_new_tokens: int = 2048,
        device_map: str = "auto",
        prompt_config: str = None
    ):
        """
        初始化 Hunyuan 翻譯模型

        Args:
            model_path: 模型路徑
            max_new_tokens: 最大生成 token 數
            device_map: 設備映射策略
            prompt_config: 翻譯提示詞配置檔案路徑 (YAML 格式)
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        # 載入 prompt 配置
        prompt_config_path = prompt_config if prompt_config is not None else self.DEFAULT_PROMPT_CONFIG
        self.prompt = self._load_prompt_config(prompt_config_path)

        print(f"正在載入 Hunyuan 翻譯模型: {model_path}")

        # 載入模型和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map
        )

        print("模型載入成功！")

    def _load_prompt_config(self, config_path: str) -> str:
        """
        從 YAML 配置檔案載入 prompt

        Args:
            config_path: 配置檔案路徑

        Returns:
            prompt 字串
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到 prompt 配置檔案: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        print(f"已載入 prompt 配置: {config_path}")
        print(f"Prompt 內容預覽:\n {prompt}")
        return prompt

    def output_parse(self, raw_output: str) -> str:
        """
        解析模型輸出，萃取 [開始翻譯] 之後的翻譯內容

        Args:
            raw_output: 模型的原始輸出文字

        Returns:
            解析後的純翻譯文字
        """
        cc = opencc.OpenCC('s2tw')
        translated_text = cc.convert(raw_output)
        return translated_text.strip()

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

        # 使用 .format() 將 {text} 替換為實際文字
        # 如果 prompt 中沒有 {text}，則直接附加到後面
        if "{text}" in self.prompt:
            formatted_prompt = self.prompt.format(text=text)
        else:
            formatted_prompt = f"{self.prompt}\n\n{text}"

        messages = [
            {
                "role": "user",
                "content": formatted_prompt
            },
        ]

        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        outputs = self.model.generate(
            tokenized_chat.to(self.model.device),
            max_new_tokens=self.max_new_tokens
        )
        generated_tokens = outputs[0][tokenized_chat.shape[1]:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_text

    def translate_with_parse(self, text: str) -> str:
        """
        將單一日文文字翻譯為繁體中文，並解析輸出以獲取純翻譯內容

        Args:
            text: 要翻譯的日文文字

        Returns:
            翻譯後的繁體中文文字
        """
        parsed_output = self.translate(text)
        parsed_output = self.output_parse(parsed_output)
        return parsed_output

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
        # main_translated_text = self.translate_with_parse(ocr_data["text"]) if ocr_data["text"] else ""
        main_translated_text = None

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
        python lib/translation/hunyuan.py --input_path ocr.json --output_path translated.json --model_path path/to/model
    """
    parser = argparse.ArgumentParser(description="日文到繁體中文翻譯 (使用 Hunyuan-MT-7B)")
    parser.add_argument("--input_path", type=str, default="results/2/ocr.json",
                       help="包含 OCR 結果的輸入 JSON 檔案路徑")
    parser.add_argument("--output_path", type=str, default="results/2/translated_hunyuan.json",
                       help="翻譯結果的輸出 JSON 檔案路徑")
    parser.add_argument("--model_path", type=str, default="/tmp2/share_data/HY-MT1.5-7B",
                       help="Hunyuan 模型路徑")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                       help="最大生成 token 數")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="設備映射策略")
    parser.add_argument("--prompt_config", type=str, default="configs/translation_prompt/hunyuan_default.yml",
                       help="翻譯提示詞配置檔案路徑 (YAML 格式)")

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
    print("\n初始化 Hunyuan 翻譯模型...")
    translator = Translation_Model_Hunyuan(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        prompt_config=args.prompt_config
    )
    print("模型載入成功！")

    # 處理 OCR 資料
    print(f"\n處理 OCR 資料...")
    total_boxes = sum(len(entry.get("bounding_boxes", [])) for entry in ocr_data_list)
    print(f"找到 {len(ocr_data_list)} 個條目，共 {total_boxes} 個邊界框")

    # 翻譯每個 OCR 輸出
    translation_results: List[TranslationOutputSchema] = []

    print(f"\n翻譯文字中...")
    for ocr_entry in tqdm(ocr_data_list, desc="翻譯條目", unit="條目"):
        try:
            # 使用 translate_ocr_output 方法
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