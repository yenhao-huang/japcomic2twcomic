"""
測試 PaddleOCR v5 文字偵測並視覺化結果
"""

import sys
import os
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from lib.ocr.ppocr import OCR
import argparse


def test_detection_visualization(image_path, output_dir, text_det_model_dir, text_rec_model_dir=None):
    """
    測試 OCR 偵測並視覺化結果

    Args:
        image_path: 圖片路徑
        output_dir: 輸出目錄
        text_det_model_dir: 文字偵測模型路徑
        text_rec_model_dir: 文字辨識模型路徑（可選）
    """
    print(f"🔧 初始化 OCR 模型...")
    ocr = OCR(
        text_det_model_dir=text_det_model_dir,
        text_recognition_model_dir=text_rec_model_dir,
        output_dir=output_dir
    )

    print(f"\n📸 處理圖片: {image_path}")

    # 執行 OCR
    result = ocr.predict(image_path)

    # 輸出結果摘要
    print(f"\n📊 偵測結果摘要:")
    print(f"   偵測到的文字區域數量: {len(result['bounding_boxes'])}")
    print(f"\n📝 辨識文字:")
    print("=" * 60)
    print(result['text'])
    print("=" * 60)

    # 繪製邊界框
    print(f"\n🎨 繪製邊界框...")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_detection.jpg")

    visualized_path = ocr.draw_boxes(
        image_path=image_path,
        bounding_boxes=result['bounding_boxes'],
        output_path=output_path,
        box_color=(255, 0, 0),  # 紅色
        box_width=3,
        show_text=True,
        font_size=16
    )

    print(f"✅ 視覺化結果已儲存至: {visualized_path}")

    # 輸出詳細的邊界框資訊
    print(f"\n📦 詳細邊界框資訊:")
    for i, bbox in enumerate(result['bounding_boxes'], 1):
        print(f"\n區域 {i}:")
        print(f"  文字: {bbox['text']}")
        print(f"  信心分數: {bbox['score']:.4f}" if bbox['score'] else "  信心分數: N/A")
        print(f"  座標: {bbox['box']}")


def test_batch_detection(input_dir, output_dir, text_det_model_dir, text_rec_model_dir=None):
    """
    批次測試 OCR 偵測並視覺化結果

    Args:
        input_dir: 輸入圖片目錄
        output_dir: 輸出目錄
        text_det_model_dir: 文字偵測模型路徑
        text_rec_model_dir: 文字辨識模型路徑（可選）
    """
    print(f"🔧 初始化 OCR 模型...")
    ocr = OCR(
        text_det_model_dir=text_det_model_dir,
        text_recognition_model_dir=text_rec_model_dir,
        output_dir=output_dir
    )

    print(f"\n📁 載入目錄: {input_dir}")
    image_paths = ocr.load_directory(input_dir, recursive=True)

    if not image_paths:
        print("⚠️  沒有找到圖片")
        return

    print(f"✅ 找到 {len(image_paths)} 張圖片\n")

    # 批次處理
    results = ocr.predict_batch(image_paths)

    # 為每個結果繪製邊界框
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    for i, result in enumerate(results):
        if 'error' in result:
            print(f"❌ 跳過失敗的圖片: {result['source']}")
            continue

        base_name = os.path.splitext(os.path.basename(result['source']))[0]
        output_path = os.path.join(output_dir, f"{base_name}_detection.jpg")

        try:
            visualized_path = ocr.draw_boxes(
                image_path=result['source'],
                bounding_boxes=result['bounding_boxes'],
                output_path=output_path,
                box_color=(0, 255, 0),  # 綠色
                box_width=2,
                show_text=True,
                font_size=14
            )
            success_count += 1
            print(f"✅ [{i+1}/{len(results)}] 已儲存: {visualized_path}")
        except Exception as e:
            print(f"❌ [{i+1}/{len(results)}] 繪製失敗: {result['source']}, 錯誤: {e}")

    # 輸出摘要
    print(f"\n" + "=" * 60)
    print(f"📊 批次處理摘要:")
    print(f"=" * 60)
    print(f"✅ 成功視覺化: {success_count}/{len(results)} 張")
    print(f"💾 輸出目錄: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='測試 PaddleOCR v5 文字偵測並視覺化')
    parser.add_argument('--image', type=str, help='單張圖片路徑')
    parser.add_argument('--input-dir', type=str, help='圖片目錄路徑（批次處理）')
    parser.add_argument('--output-dir', type=str, required=True, help='輸出目錄路徑')
    parser.add_argument('--text-det-model-dir', type=str, required=True, help='文字偵測模型路徑')
    parser.add_argument('--text-rec-model-dir', type=str, default=None, help='文字辨識模型路徑')

    args = parser.parse_args()

    # 檢查必須提供 --image 或 --input-dir 其中之一
    if not args.image and not args.input_dir:
        print(f"❌ 錯誤：請提供 --image 或 --input-dir 參數", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if args.image and args.input_dir:
        print(f"❌ 錯誤：--image 和 --input-dir 不能同時使用", file=sys.stderr)
        sys.exit(1)

    # 單張圖片處理
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ 錯誤：圖片檔案不存在 - {args.image}", file=sys.stderr)
            sys.exit(1)

        test_detection_visualization(
            image_path=args.image,
            output_dir=args.output_dir,
            text_det_model_dir=args.text_det_model_dir,
            text_rec_model_dir=args.text_rec_model_dir
        )

    # 批次處理
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"❌ 錯誤：目錄不存在 - {args.input_dir}", file=sys.stderr)
            sys.exit(1)

        test_batch_detection(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            text_det_model_dir=args.text_det_model_dir,
            text_rec_model_dir=args.text_rec_model_dir
        )


if __name__ == "__main__":
    main()
