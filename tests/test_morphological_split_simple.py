"""
簡單測試 Morphological Split 功能（不依賴外部模組）
"""

import numpy as np
import cv2
from PIL import Image
from typing import List


class SimpleSplitTester:
    """簡化版的分離測試器"""

    def _split_connected_regions(self, mask: np.ndarray) -> List[np.ndarray]:
        """使用形態學操作分離連在一起的 speech bubbles"""
        # 確保 mask 是二值化的 uint8 格式
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        # 使用連通組件分析找出所有獨立的區域
        num_labels, labels = cv2.connectedComponents(mask_uint8)

        # 將每個連通組件轉換為獨立的 mask
        split_regions = []
        for label in range(1, num_labels):  # 跳過 label 0 (背景)
            region_mask = (labels == label).astype(np.uint8)
            split_regions.append(region_mask)

        # 如果只有一個區域，嘗試使用形態學操作進一步分離
        if len(split_regions) == 1:
            split_regions = self._morphological_split(mask_uint8)

        # 按區域大小排序（從大到小）
        split_regions.sort(key=lambda m: np.sum(m > 0), reverse=True)

        return split_regions

    def _morphological_split(self, mask: np.ndarray) -> List[np.ndarray]:
        """使用形態學操作嘗試分離單一大區域"""
        # 使用腐蝕操作來分離連接的區域
        # 調整 kernel size 計算，使其更積極分離
        kernel_size = max(5, int(np.sqrt(np.sum(mask > 0)) / 15))
        print(f"  使用 kernel size: {kernel_size}")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 腐蝕（增加迭代次數）
        eroded = cv2.erode(mask, kernel, iterations=3)

        # 找出連通組件
        num_labels, labels = cv2.connectedComponents(eroded)

        # 如果成功分離出多個區域
        if num_labels > 2:  # 超過 1 個區域（排除背景）
            split_regions = []
            for label in range(1, num_labels):
                # 使用原始 mask 的範圍
                region_mask = np.zeros_like(mask)
                region_mask[labels == label] = 255

                # 膨脹回原來的大小
                dilated = cv2.dilate(region_mask, kernel, iterations=2)

                # 確保不超出原始 mask 的範圍
                dilated = cv2.bitwise_and(dilated, mask)

                if np.sum(dilated > 0) > 100:  # 過濾太小的區域
                    split_regions.append((dilated > 0).astype(np.uint8))

            if len(split_regions) > 1:
                return split_regions

        # 如果無法分離，返回原始 mask
        return [(mask > 0).astype(np.uint8)]


def create_test_mask_with_two_bubbles():
    """創建一個包含兩個連接對話框的測試 mask"""
    mask = np.zeros((500, 400), dtype=np.uint8)

    # 第一個對話框（上方，較大）
    cv2.ellipse(mask, (200, 120), (90, 70), 0, 0, 360, 255, -1)

    # 第二個對話框（下方，較小）
    cv2.ellipse(mask, (200, 350), (70, 60), 0, 0, 360, 255, -1)

    # 添加一個較細的連接部分（模擬兩個對話框連在一起，但連接較弱）
    # 使用較窄的連接，讓形態學操作更容易分離
    cv2.rectangle(mask, (190, 190), (210, 290), 255, -1)

    return mask


def test_split_regions():
    """測試區域分離功能"""
    print("=" * 60)
    print("測試 Morphological Split 功能")
    print("=" * 60)

    # 創建測試器
    tester = SimpleSplitTester()

    # 創建測試 mask
    test_mask = create_test_mask_with_two_bubbles()

    print(f"\n原始 mask:")
    print(f"  形狀: {test_mask.shape}")
    print(f"  非零像素數: {np.sum(test_mask > 0)}")

    # 執行分離
    split_regions = tester._split_connected_regions(test_mask)

    print(f"\n分離結果:")
    print(f"  成功分離出 {len(split_regions)} 個區域")

    for i, region in enumerate(split_regions, 1):
        pixel_count = np.sum(region > 0)
        coords = np.column_stack(np.where(region > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            width = x_max - x_min
            height = y_max - y_min
            print(f"\n  區域 {i}:")
            print(f"    像素數: {pixel_count}")
            print(f"    範圍: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
            print(f"    大小: {width} x {height}")

    # 視覺化結果
    print("\n" + "=" * 60)
    print("創建視覺化圖像...")
    print("=" * 60)

    visualize_split_result(test_mask, split_regions)

    print("\n✅ 測試完成！")
    print(f"   原始 1 個區域 -> 分離後 {len(split_regions)} 個區域")


def visualize_split_result(original_mask, split_regions):
    """視覺化分離結果"""
    h, w = original_mask.shape

    # 創建三張圖：原始、分離結果、疊加
    # 1. 原始 mask
    vis_original = np.zeros((h, w, 3), dtype=np.uint8)
    vis_original[original_mask > 0] = [255, 255, 255]

    # 2. 分離結果（不同區域用不同顏色）
    vis_split = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [
        (255, 100, 100),  # 紅色
        (100, 255, 100),  # 綠色
        (100, 100, 255),  # 藍色
        (255, 255, 100),  # 黃色
        (255, 100, 255),  # 紫色
    ]

    for i, region_mask in enumerate(split_regions):
        color = colors[i % len(colors)]
        vis_split[region_mask > 0] = color

    # 3. 合併圖像（並排顯示）
    combined = np.hstack([vis_original, vis_split])

    # 保存圖像
    from pathlib import Path
    output_path = Path("tests/testcases/morphological_split_result.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_img = Image.fromarray(combined)
    combined_img.save(str(output_path))

    print(f"\n視覺化結果已保存至: {output_path}")
    print(f"  左側: 原始 mask (白色)")
    print(f"  右側: 分離結果 (不同顏色代表不同區域)")


if __name__ == "__main__":
    test_split_regions()
