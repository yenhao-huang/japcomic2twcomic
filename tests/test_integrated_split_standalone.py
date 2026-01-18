"""
獨立測試整合的分離功能（不依賴其他模組）
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Optional


class SimplifiedRenderer:
    """簡化的渲染器，僅用於測試分離功能"""

    def __init__(self):
        self.font_cache = {}

    def _split_connected_regions(self, mask: np.ndarray) -> List[np.ndarray]:
        """使用形態學操作分離連在一起的 speech bubbles"""
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        # 使用連通組件分析找出所有獨立的區域
        num_labels, labels = cv2.connectedComponents(mask_uint8)

        split_regions = []
        for label in range(1, num_labels):
            region_mask = (labels == label).astype(np.uint8)
            split_regions.append(region_mask)

        # 如果只有一個區域，嘗試使用形態學操作進一步分離
        if len(split_regions) == 1:
            split_regions = self._morphological_split(mask_uint8)

        # 按區域大小排序
        split_regions.sort(key=lambda m: np.sum(m > 0), reverse=True)

        return split_regions

    def _morphological_split(self, mask: np.ndarray) -> List[np.ndarray]:
        """優先使用距離變換，失敗時回退到腐蝕方法"""
        # 方法 1: 嘗試距離變換
        try:
            split_regions = self._split_by_distance_transform(mask)
            if len(split_regions) > 1:
                return split_regions
        except Exception as e:
            print(f"    距離變換失敗: {e}")

        # 方法 2: 腐蝕方法
        kernel_size = max(5, int(np.sqrt(np.sum(mask > 0)) / 15))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(mask, kernel, iterations=3)
        num_labels, labels = cv2.connectedComponents(eroded)

        if num_labels > 2:
            split_regions = []
            for label in range(1, num_labels):
                region_mask = np.zeros_like(mask)
                region_mask[labels == label] = 255
                dilated = cv2.dilate(region_mask, kernel, iterations=2)
                dilated = cv2.bitwise_and(dilated, mask)
                if np.sum(dilated > 0) > 100:
                    split_regions.append((dilated > 0).astype(np.uint8))
            if len(split_regions) > 1:
                return split_regions

        return [(mask > 0).astype(np.uint8)]

    def _split_by_distance_transform(self, mask: np.ndarray) -> List[np.ndarray]:
        """使用距離變換 + 分水嶺算法分離"""
        # 1. 距離變換
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # 2. 確定前景
        threshold_ratio = 0.3
        _, sure_fg = cv2.threshold(
            dist_transform,
            threshold_ratio * dist_transform.max(),
            255,
            cv2.THRESH_BINARY
        )
        sure_fg = np.uint8(sure_fg)

        # 3. 確定背景
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=3)

        # 4. 未知區域
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 5. 標記前景區域
        num_labels, markers = cv2.connectedComponents(sure_fg)

        if num_labels <= 2:
            return [(mask > 0).astype(np.uint8)]

        # 6. 調整標記
        markers = markers + 1
        markers[unknown == 255] = 0

        # 7. 分水嶺算法
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(mask_3ch, markers)

        # 8. 提取區域
        split_regions = []
        for label in range(2, num_labels + 1):
            region_mask = np.zeros_like(mask)
            region_mask[markers == label] = 255
            region_mask = cv2.dilate(region_mask, kernel, iterations=2)
            region_mask = cv2.bitwise_and(region_mask, mask)
            if np.sum(region_mask > 0) > 100:
                split_regions.append((region_mask > 0).astype(np.uint8))

        return split_regions if len(split_regions) > 1 else [(mask > 0).astype(np.uint8)]

    def _assign_text_to_regions(self, text: str, regions: List[np.ndarray]) -> List[tuple]:
        """根據區域大小分配文字"""
        if len(regions) == 1:
            return [(regions[0], text)]

        # 根據換行符分割
        text_parts = [part.strip() for part in text.split('\n') if part.strip()]

        # 如果沒有換行符，根據區域大小比例分配
        if len(text_parts) <= 1:
            text_parts = self._split_text_by_region_sizes(text, regions)

        assignments = []
        if len(text_parts) >= len(regions):
            for i, region in enumerate(regions):
                if i < len(regions) - 1:
                    assignments.append((region, text_parts[i]))
                else:
                    remaining_text = '\n'.join(text_parts[i:])
                    assignments.append((region, remaining_text))
        else:
            for i, text_part in enumerate(text_parts):
                assignments.append((regions[i], text_part))
            for i in range(len(text_parts), len(regions)):
                assignments.append((regions[i], ""))

        return assignments

    def _split_text_by_region_sizes(self, text: str, regions: List[np.ndarray]) -> List[str]:
        """根據區域大小比例分割文字"""
        region_sizes = [np.sum(region > 0) for region in regions]
        total_size = sum(region_sizes)
        text_length = len(text)

        char_counts = []
        for size in region_sizes:
            proportion = size / total_size
            char_count = max(1, int(text_length * proportion))
            char_counts.append(char_count)

        text_parts = []
        start_idx = 0
        for i, char_count in enumerate(char_counts):
            if i == len(char_counts) - 1:
                text_parts.append(text[start_idx:])
            else:
                end_idx = min(start_idx + char_count, len(text))
                text_parts.append(text[start_idx:end_idx])
                start_idx = end_idx

        return text_parts


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
    renderer = SimplifiedRenderer()

    # 測試分離功能
    print("\n執行分離...")
    split_regions = renderer._split_connected_regions(mask)

    print(f"\n✅ 分離結果:")
    print(f"  成功分離出 {len(split_regions)} 個區域")

    # 顯示每個區域的資訊
    print(f"\n區域詳情 (前 10 個):")
    for i, region in enumerate(split_regions[:10], 1):
        pixel_count = np.sum(region > 0)
        coords = np.column_stack(np.where(region > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            width = x_max - x_min
            height = y_max - y_min
            print(f"  區域 {i:2d}: {pixel_count:5d} 像素, "
                  f"x=[{x_min:3d},{x_max:3d}], y=[{y_min:3d},{y_max:3d}], "
                  f"{width}x{height}")

    if len(split_regions) > 10:
        print(f"  ... 還有 {len(split_regions) - 10} 個區域")

    # 測試文字分配
    test_text = "這是一段測試文字，用來測試分配功能"
    print(f"\n文字分配測試:")
    print(f"  原始文字: '{test_text}' ({len(test_text)} 字元)")

    assignments = renderer._assign_text_to_regions(test_text, split_regions)

    print(f"\n分配結果 (前 10 個):")
    for i, (region_mask, text) in enumerate(assignments[:10], 1):
        pixel_count = np.sum(region_mask > 0)
        print(f"  區域 {i:2d} ({pixel_count:5d} px): '{text}'")

    if len(assignments) > 10:
        print(f"  ... 還有 {len(assignments) - 10} 個區域")

    # 視覺化結果
    print("\n" + "=" * 70)
    print("創建視覺化...")
    print("=" * 70)
    visualize_split_result(mask, split_regions)

    print("\n✅ 測試完成！")
    print(f"   原始 mask 已成功分離為 {len(split_regions)} 個獨立區域")


def visualize_split_result(original_mask, split_regions):
    """視覺化分離結果"""
    h, w = original_mask.shape

    # 原始 mask
    vis_original = np.zeros((h, w, 3), dtype=np.uint8)
    vis_original[original_mask > 0] = [255, 255, 255]

    # 分離結果
    vis_split = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (255, 150, 100), (150, 100, 255), (100, 255, 150),
    ]

    for i, region_mask in enumerate(split_regions):
        color = colors[i % len(colors)]
        vis_split[region_mask > 0] = color

    # 合併
    combined = np.hstack([vis_original, vis_split])

    # 保存
    output_path = Path("tests/testcases/integrated_split_result.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_img = Image.fromarray(combined)
    combined_img.save(str(output_path))

    print(f"\n視覺化結果已保存至: {output_path}")
    print(f"  左側: 原始 mask (白色)")
    print(f"  右側: 分離後的 {len(split_regions)} 個區域 (不同顏色)")


if __name__ == "__main__":
    test_real_mask_split()
