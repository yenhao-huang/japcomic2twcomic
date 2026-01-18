"""
測試 Morphological Split 功能
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw
import sys
from pathlib import Path

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.render.text_mask_render import TextMaskBasedRenderer


def create_test_mask_with_two_bubbles():
    """創建一個包含兩個連接對話框的測試 mask"""
    mask = np.zeros((400, 300), dtype=np.uint8)

    # 第一個對話框（上方，較大）
    cv2.ellipse(mask, (150, 100), (80, 60), 0, 0, 360, 255, -1)

    # 第二個對話框（下方，較小）
    cv2.ellipse(mask, (150, 250), (60, 50), 0, 0, 360, 255, -1)

    # 添加一個細長的連接部分（模擬兩個對話框連在一起）
    cv2.rectangle(mask, (140, 160), (160, 200), 255, -1)

    return mask


def test_split_regions():
    """測試區域分離功能"""
    print("測試 Morphological Split 功能...")

    # 創建渲染器
    renderer = TextMaskBasedRenderer()

    # 創建測試 mask
    test_mask = create_test_mask_with_two_bubbles()

    print(f"原始 mask 形狀: {test_mask.shape}")
    print(f"原始 mask 非零像素數: {np.sum(test_mask > 0)}")

    # 執行分離
    split_regions = renderer._split_connected_regions(test_mask)

    print(f"\n分離結果:")
    print(f"  分離出 {len(split_regions)} 個區域")

    for i, region in enumerate(split_regions, 1):
        pixel_count = np.sum(region > 0)
        coords = np.column_stack(np.where(region > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            print(f"  區域 {i}: {pixel_count} 像素, 範圍: ({x_min}, {y_min}) -> ({x_max}, {y_max})")

    # 測試文字分配
    test_text = "這是第一個對話框的文字\n這是第二個對話框的文字"
    assignments = renderer._assign_text_to_regions(test_text, split_regions)

    print(f"\n文字分配結果:")
    for i, (region_mask, text) in enumerate(assignments, 1):
        print(f"  區域 {i}: '{text}' ({np.sum(region_mask > 0)} 像素)")

    # 保存視覺化結果
    visualize_split_result(test_mask, split_regions, assignments)

    print("\n測試完成！")


def visualize_split_result(original_mask, split_regions, assignments):
    """視覺化分離結果"""
    # 創建彩色圖像來顯示不同區域
    h, w = original_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # 為每個區域分配不同顏色
    colors = [
        (255, 100, 100),  # 紅色
        (100, 255, 100),  # 綠色
        (100, 100, 255),  # 藍色
        (255, 255, 100),  # 黃色
        (255, 100, 255),  # 紫色
    ]

    for i, (region_mask, text) in enumerate(assignments):
        color = colors[i % len(colors)]
        vis[region_mask > 0] = color

    # 保存圖像
    output_path = Path("tests/testcases/morphological_split_result.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis_img = Image.fromarray(vis)
    vis_img.save(str(output_path))

    print(f"\n視覺化結果已保存至: {output_path}")


if __name__ == "__main__":
    test_split_regions()
