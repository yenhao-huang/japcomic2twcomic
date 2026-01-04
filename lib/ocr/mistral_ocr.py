import os
import json
import glob
import base64
import requests
from pathlib import Path
from typing import List
import opencc


def _encode_image(image_path: str) -> str:
    """
    將圖片編碼為 base64 data URL

    Args:
        image_path: 圖片檔案路徑

    Returns:
        base64 編碼的 data URL
    """
    # 如果已經是 URL（http/https/data/file），直接返回
    if image_path.startswith(('http://', 'https://', 'data:', 'file://')):
        return image_path

    # 讀取圖片並轉為 base64
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # 根據檔案副檔名決定 MIME 類型
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/png')

    return f"data:{mime_type};base64,{image_data}"


def call_vllm(vllm_url: str, image_url: str, model_name: str, text: str) -> str:
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
    encoded_image = _encode_image(image_url)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_image
                        }
                    },
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


def load_images_from_directory(data_dir: str) -> List[str]:
    """
    從目錄載入所有圖片檔案

    Args:
        data_dir: 資料目錄路徑

    Returns:
        圖片路徑列表
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"目錄不存在：{data_dir}")

    # 支援的圖片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']

    image_paths = []
    for ext in image_extensions:
        pattern = os.path.join(data_dir, '**', ext)
        image_paths.extend(glob.glob(pattern, recursive=True))

    # 排序確保處理順序一致
    image_paths.sort()

    return image_paths


def process_images_with_vllm(
    data_dir: str,
    vllm_url: str = "http://192.168.1.76:3132/v1/chat/completions",
    model_name: str = "mistralai3",
    output_dir: str = "results/2_mistralocr",
    ocr_prompt: str = "執行 OCR 任務：",
    translate_prompt: str = "將日文翻譯成繁體中文"
) -> None:
    """
    批次處理圖片並使用 VLLM 進行 OCR 和翻譯

    Args:
        data_dir: 圖片資料目錄
        vllm_url: VLLM 服務 URL
        model_name: 模型名稱
        output_dir: 輸出目錄
        ocr_prompt: OCR 提示詞（預設："執行 OCR 任務："）
        translate_prompt: 翻譯提示詞（預設："翻譯日文"）
    """
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 初始化 OpenCC 轉換器（簡體轉繁體）
    cc = opencc.OpenCC('s2tw')

    # 載入圖片
    print(f"📁 從目錄載入圖片：{data_dir}")
    image_paths = load_images_from_directory(data_dir)

    if not image_paths:
        print(f"⚠️  警告：目錄中沒有找到圖片檔案")
        return

    print(f"✅ 找到 {len(image_paths)} 張圖片\n")

    # 處理每張圖片
    results = []
    for idx, image_path in enumerate(image_paths, 1):
        print(f"🔄 處理進度：{idx}/{len(image_paths)} - {os.path.basename(image_path)}")

        try:
            # 第一步：呼叫 VLLM 進行 OCR
            print(f"   📝 執行 OCR...")
            ocr_text = call_vllm(vllm_url, image_path, model_name, ocr_prompt)
            print(f"   ✓ OCR 完成，文字長度：{len(ocr_text)} 字元")

            # 第二步：呼叫 VLLM 進行翻譯
            print(f"   🌐 執行翻譯...")
            translated_text = call_vllm(vllm_url, image_path, model_name, translate_prompt)

            # 第三步：使用 OpenCC 確保繁體中文正確
            print(f"   🔄 轉換為繁體中文...")
            translated_text = cc.convert(translated_text)
            print(f"   ✓ 翻譯完成，文字長度：{len(translated_text)} 字元")

            result = {
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "ocr_text": ocr_text,
                "translated_text": translated_text,
                "status": "success"
            }

        except Exception as e:
            print(f"   ✗ 處理失敗：{e}")
            result = {
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "error": str(e),
                "status": "failed"
            }

        results.append(result)

    # 儲存結果
    output_file = os.path.join(output_dir, "ocr_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"📊 處理完成！")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count

    print(f"✅ 成功：{success_count} 張")
    print(f"❌ 失敗：{failed_count} 張")
    print(f"💾 結果已儲存至：{output_file}")


def main():
    """主程式"""
    import argparse

    parser = argparse.ArgumentParser(description='使用 VLLM (Mistral OCR) 批次處理圖片並翻譯')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='圖片資料目錄路徑')
    parser.add_argument('--vllm-url', type=str,
                        default='http://192.168.1.76:3132/v1/chat/completions',
                        help='VLLM 服務 URL（預設：http://192.168.1.76:3132/v1/chat/completions）')
    parser.add_argument('--model-name', type=str,
                        default='mistralai3',
                        help='模型名稱（預設：mistralai3）')
    parser.add_argument('--output-dir', type=str,
                        default='results/2_mistralocr',
                        help='輸出目錄路徑（預設：results/2_mistralocr）')
    parser.add_argument('--ocr-prompt', type=str,
                        default='執行 OCR 任務：',
                        help='OCR 提示詞（預設：執行 OCR 任務：）')
    parser.add_argument('--translate-prompt', type=str,
                        default='翻譯日文',
                        help='翻譯提示詞（預設：翻譯日文）')

    args = parser.parse_args()

    process_images_with_vllm(
        data_dir=args.data_dir,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        output_dir=args.output_dir,
        ocr_prompt=args.ocr_prompt,
        translate_prompt=args.translate_prompt
    )


if __name__ == "__main__":
    main()
