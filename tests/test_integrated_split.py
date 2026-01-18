"""
測試整合到主渲染器的分離功能
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.render.text_mask_render import TextMaskBasedRenderer


def test_real_mask_split():
    """測試真實 mask 的分離效果"""
    print("=" * 70)
    print("測試整合的 Morphological Split 功能（使用真實 mask）")
    print("=" * 70)

    # 載入真實 mask
    mask_path = "tests/testcases/png.png"
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"錯誤: 無法讀取圖像 {mask_path}")
        return

    # 二值化
    _, mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    print(f"\n原始 mask:")
    print(f"  形狀: {mask.shape}")
    print(f"  非零像素: {np.sum(mask > 0)}")

    # 創建渲染器
    renderer = TextMaskBasedRenderer()

    # 測試分離功能
    print("\n執行分離...")
    split_regions = renderer._split_connected_regions(mask)

    print(f"\n分離結果:")
    print(f"  成功分離出 {len(split_regions)} 個區域")

    # 顯示每個區域的資訊
    for i, region in enumerate(split_regions[:10], 1):  # 只顯示前 10 個
        pixel_count = np.sum(region > 0)
        coords = np.column_stack(np.where(region > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            width = x_max - x_min
            height = y_max - y_min
            print(f"  區域 {i}: {pixel_count:5d} 像素, "
                  f"範圍 x=[{x_min:3d},{x_max:3d}], y=[{y_min:3d},{y_max:3d}], "
                  f"大小 {width}x{height}")

    if len(split_regions) > 10:
        print(f"  ... 還有 {len(split_regions) - 10} 個區域")

    # 測試文字分配
    test_text = "這是一段很長的測試文字，用來測試文字分配功能。包含多個句子。"
    print(f"\n測試文字分配:")
    print(f"  原始文字: '{test_text}'")
    print(f"  文字長度: {len(test_text)} 個字元")

    assignments = renderer._assign_text_to_regions(test_text, split_regions)

    print(f"\n文字分配結果 (前 10 個):")
    for i, (region_mask, text) in enumerate(assignments[:10], 1):
        pixel_count = np.sum(region_mask > 0)
        print(f"  區域 {i} ({pixel_count:5d} px): '{text}'")

    if len(assignments) > 10:
        print(f"  ... 還有 {len(assignments) - 10} 個區域")

    # 視覺化結果
    print("\n創建視覺化...")
    visualize_split_result(mask, split_regions)

    print("\n✅ 測試完成！")


def visualize_split_result(original_mask, split_regions):
    """視覺化分離結果"""
    h, w = original_mask.shape

    # 原始 mask
    vis_original = np.zeros((h, w, 3), dtype=np.uint8)
    vis_original[original_mask > 0] = [255, 255, 255]

    # 分離結果（不同區域用不同顏色）
    vis_split = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [
        (255, 100, 100),  # 紅色
        (100, 255, 100),  # 綠色
        (100, 100, 255),  # 藍色
        (255, 255, 100),  # 黃色
        (255, 100, 255),  # 紫色
        (100, 255, 255),  # 青色
        (255, 150, 100),  # 橙色
        (150, 100, 255),  # 紫羅蘭
    ]

    for i, region_mask in enumerate(split_regions):
        color = colors[i % len(colors)]
        vis_split[region_mask > 0] = color

    # 合併圖像
    combined = np.hstack([vis_original, vis_split])

    # 保存
    output_path = Path("tests/testcases/integrated_split_result.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_img = Image.fromarray(combined)
    combined_img.save(str(output_path))

    print(f"視覺化結果已保存至: {output_path}")
    print(f"  左側: 原始 mask")
    print(f"  右側: 分離後的 {len(split_regions)} 個區域（不同顏色）")


if __name__ == "__main__":
    test_real_mask_split()
