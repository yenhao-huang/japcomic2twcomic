import os
import json
import requests
import opencc
import yaml
from typing import List, Dict, Any
from lib.schema import TranslationOutputSchema, TranslationErrorSchema

def load_prompt_from_config(config_path: str) -> str:
    """
    從 YAML 配置檔案載入翻譯提示詞

    Args:
        config_path: YAML 配置檔案路徑

    Returns:
        翻譯提示詞文字
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置檔案不存在：{config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        prompt_content = f.read().strip()

    return prompt_content


def call_vllm(vllm_url: str, model_name: str, text: str) -> str:
    """
    呼叫 VLLM 方法對單張圖片執行任務

    Args:
        vllm_url: VLLM 服務的 URL
        image_url: 圖片的 URL 或路徑
        model_name: 模型名稱
        text: 提示詞文字（例如："執行 OCR 任務：" 或 "翻譯日文"）

    Returns:
        生成的回應文字
    """

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "stop": ["<end_of_turn>"],
    }

    try:
        response = requests.post(
            vllm_url,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        print(f"請求超時，返回空字串")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 錯誤：{e}")
        print(f"請求 URL：{vllm_url}")
        print(f"回應內容：{response.text}")
        return ""
    except Exception as e:
        print(f"未預期的錯誤：{e}")
        return ""

def translate_text_with_vllm(
    text: str,
    vllm_url: str = "http://192.168.1.76:3132/v1/chat/completions",
    model_name: str = "mistralai3",
    translate_prompt: str = "將以下日文翻譯成繁體中文："
) -> str:
    """
    使用 VLLM 翻譯文字

    Args:
        text: 要翻譯的文字
        vllm_url: VLLM 服務 URL
        model_name: 模型名稱
        translate_prompt: 翻譯提示詞

    Returns:
        翻譯後的文字
    """
    # 初始化 OpenCC 轉換器（簡體轉繁體）
    # cc = opencc.OpenCC('s2tw')

    # 呼叫 VLLM 進行翻譯
    full_prompt = f"{translate_prompt}\n\n{text}"
    translated_text = call_vllm(vllm_url, model_name, full_prompt)

    # 使用 OpenCC 確保繁體中文正確
    # translated_text = cc.convert(translated_text)

    return translated_text


def translate_ocr_json(
    ocr_json_path: str,
    vllm_url: str = "http://192.168.1.76:3132/v1/chat/completions",
    model_name: str = "mistralai3",
    output_path: str = None,
    translate_prompt: str = "將以下日文翻譯成繁體中文："
) -> None:
    """
    從 OCR JSON 檔案載入結果並進行翻譯

    Args:
        ocr_json_path: OCR 結果的 JSON 檔案路徑
        vllm_url: VLLM 服務 URL
        model_name: 模型名稱
        output_path: 輸出檔案路徑（若未指定則自動生成）
        translate_prompt: 翻譯提示詞
    """
    # 載入 OCR 結果
    print(f"📁 載入 OCR 結果：{ocr_json_path}")
    with open(ocr_json_path, 'r', encoding='utf-8') as f:
        ocr_results = json.load(f)

    print(f"✅ 找到 {len(ocr_results)} 筆 OCR 結果\n")

    # 處理每筆 OCR 結果
    results: List[Dict[str, Any]] = []
    for idx, ocr_item in enumerate(ocr_results, 1):
        # 跳過錯誤項目
        if 'error' in ocr_item:
            print(f"⏭️  跳過第 {idx} 筆（OCR 失敗）：{ocr_item.get('source', 'unknown')}")
            results.append(ocr_item)
            continue

        source = ocr_item.get('source', ocr_item.get('image_path', 'unknown'))
        ocr_text = ocr_item.get('text', '')

        print(f"🔄 處理進度：{idx}/{len(ocr_results)} - {os.path.basename(source)}")
        print(f"   原文長度：{len(ocr_text)} 字元")

        try:
            # 翻譯 OCR 文字
            print(f"   🌐 執行翻譯...")
            translated_text = translate_text_with_vllm(
                text=ocr_text,
                vllm_url=vllm_url,
                model_name=model_name,
                translate_prompt=translate_prompt
            )
            print(f"   ✓ 翻譯完成，譯文長度：{len(translated_text)} 字元")

            # 翻譯每個邊界框中的文字
            translated_bboxes = []
            if 'bounding_boxes' in ocr_item and ocr_item['bounding_boxes']:
                print(f"   🔄 翻譯 {len(ocr_item['bounding_boxes'])} 個邊界框...")
                for bbox_idx, bbox in enumerate(ocr_item['bounding_boxes'], 1):
                    bbox_text = bbox.get('text', '')
                    if bbox_text:
                        print(f"      翻譯邊界框 {bbox_idx}/{len(ocr_item['bounding_boxes'])}: {bbox_text}")
                        bbox_translated = translate_text_with_vllm(
                            text=bbox_text,
                            vllm_url=vllm_url,
                            model_name=model_name,
                            translate_prompt=translate_prompt
                        )
                        print(f"      結果: {bbox_translated}\\n")
                    else:
                        bbox_translated = ''

                    translated_bboxes.append({
                        'box': bbox.get('box', []),
                        'text': bbox_text,
                        'score': bbox.get('score'),
                        'translated_text': bbox_translated
                    })

            result: TranslationOutputSchema = {
                "source": source,
                "text": ocr_text,
                "translated_text": translated_text,
                "image_path": ocr_item.get('image_path', source),
                "bounding_boxes": translated_bboxes
            }
            results.append(result)

        except Exception as e:
            print(f"   ✗ 翻譯失敗：{e}")
            error_result: TranslationErrorSchema = {
                "source": source,
                "error": str(e)
            }
            results.append(error_result)  # type: ignore

    # 決定輸出路徑
    if output_path is None:
        # 自動生成輸出路徑（與輸入檔案同目錄）
        input_dir = os.path.dirname(ocr_json_path)
        output_path = os.path.join(input_dir, "mistral_translated.json")

    # 儲存結果
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"📊 處理完成！")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if 'error' not in r)
    failed_count = len(results) - success_count

    print(f"✅ 成功：{success_count} 筆")
    print(f"❌ 失敗：{failed_count} 筆")
    print(f"💾 結果已儲存至：{output_path}")


def main():
    """主程式"""
    import argparse

    parser = argparse.ArgumentParser(description='使用 VLLM (Mistral) 批次翻譯')
    parser.add_argument('--data-dir', type=str,
                        help='圖片資料目錄路徑或 OCR JSON 檔案路徑')
    parser.add_argument('--vllm-url', type=str,
                        default='http://192.168.1.76:3132/v1/chat/completions',
                        help='VLLM 服務 URL（預設：http://192.168.1.76:3132/v1/chat/completions）')
    parser.add_argument('--model-name', type=str,
                        default='mistralai3',
                        help='模型名稱（預設：mistralai3）')
    parser.add_argument('--output-dir', type=str,
                        help='輸出目錄路徑（若從 JSON 載入則為輸出檔案路徑）')
    parser.add_argument('--prompt-config', type=str,
                        default='configs/translation_prompt/default.yml',
                        help='翻譯提示詞配置檔案路徑（預設：configs/translation_prompt/default.yml）')

    args = parser.parse_args()

    if not args.data_dir:
        parser.error('需要提供 --data-dir 參數')

    # 載入翻譯提示詞
    translate_prompt = load_prompt_from_config(args.prompt_config)
    print(f"📝 載入翻譯提示詞配置：{args.prompt_config}")
    print(f"提示詞長度：{len(translate_prompt)} 字元\n")

    translate_ocr_json(
        ocr_json_path=args.data_dir,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        output_path=args.output_dir,
        translate_prompt=translate_prompt
    )


if __name__ == "__main__":
    main()