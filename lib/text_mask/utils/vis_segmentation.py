"""
載入 text mask 與圖片，將 segmentation mask 應用到圖片並儲存
"""

import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from lib.text_mask.yolo import decompress_mask_from_base64_png


def apply_inpaint_masks(
    input_json_path: str,
    output_dir: str
) -> None:
    """
    載入 text mask JSON 並根據 segmentation mask 將區域填色
    segmentation=1 的區域填黑，segmentation=0 的區域填白

    Args:
        input_json_path: 輸入的 layout.json 路徑
        output_dir: 輸出圖片的目錄
    """
    input_path = Path(input_json_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 載入 JSON
    print(f"載入 JSON 檔案: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"總共 {len(data)} 張圖片")

    # 統計
    processed = 0
    skipped = 0

    for idx, item in enumerate(data):
        source = item['source']
        region_result = item.get('region_result', [])

        # 過濾出有 segmentation_mask 的區域
        regions_with_mask = [r for r in region_result if 'segmentation_mask' in r and r['segmentation_mask']]

        if not regions_with_mask:
            skipped += 1
            continue

        print(f"\n[{idx + 1}/{len(data)}] 處理: {source}")
        print(f"  找到 {len(regions_with_mask)} 個 segmentation 區域")

        # 載入圖片
        image_path = Path(source)
        if not image_path.exists():
            print(f"  警告: 圖片不存在: {image_path}")
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  錯誤: 無法讀取圖片: {image_path}")
            skipped += 1
            continue

        # 建立合併的 mask（所有文字區域的聯集）
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # 合併所有 segmentation mask
        for region in regions_with_mask:
            # 解壓縮 Base64 PNG 編碼的 mask
            mask = decompress_mask_from_base64_png(region['segmentation_mask'])

            # 確保 mask 尺寸與圖片一致
            if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 合併到 combined_mask（使用 OR 運算）
            combined_mask = np.maximum(combined_mask, mask)

        # 建立結果圖片
        result = np.zeros_like(image)

        # mask=1 的區域填黑色，mask=0 的區域填白色
        result[combined_mask == 0] = [255, 255, 255]  # 白色
        result[combined_mask > 0] = [0, 0, 0]         # 黑色

        # 儲存結果
        output_filename = image_path.stem + "_inpainted.jpg"
        output_file_path = output_path / output_filename
        cv2.imwrite(str(output_file_path), result)
        print(f"  已儲存至: {output_file_path}")

        processed += 1

    print(f"\n=== 完成 ===")
    print(f"處理成功: {processed} 張")
    print(f"跳過: {skipped} 張")


def main():
    """主要 CLI 進入點"""
    parser = argparse.ArgumentParser(
        description="載入 text mask JSON 並產生黑白 mask 圖片（文字區域=黑色，背景=白色）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 產生 mask 圖片：segmentation=1 填黑色，segmentation=0 填白色
  python -m lib.text_mask.utils.vis_segmentation \\
      --input exp/exp1/pixel_level_inpaint/benchmark2/layout/yolo/layout.json \\
      --output results/layout/yolo/inpaint
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='輸入的 layout.json 檔案路徑'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='輸出圖片的目錄'
    )

    args = parser.parse_args()

    apply_inpaint_masks(
        input_json_path=args.input,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
