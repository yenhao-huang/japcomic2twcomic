"""
測試真實的 mask 圖像分離
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_mask_from_image(image_path: str) -> np.ndarray:
    """從圖像加載 mask"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取圖像: {image_path}")

    # 二值化
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return mask


def analyze_mask(mask: np.ndarray):
    """分析 mask 的特徵"""
    print("\n=== Mask 分析 ===")
    print(f"形狀: {mask.shape}")
    print(f"非零像素數: {np.sum(mask > 0)}")

    # 連通組件分析
    num_labels, labels = cv2.connectedComponents(mask)
    print(f"連通組件數: {num_labels - 1} (排除背景)")

    # 計算輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"外部輪廓數: {len(contours)}")

    if len(contours) > 0:
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            print(f"\n輪廓 {i+1}:")
            print(f"  面積: {area}")
            print(f"  周長: {perimeter}")
            print(f"  邊界框: ({x}, {y}, {w}, {h})")
            print(f"  寬高比: {aspect_ratio:.2f}")


def advanced_split(mask: np.ndarray, method: str = "distance_transform") -> list:
    """
    使用進階方法分離 mask

    Args:
        mask: 二值化 mask (uint8, 0-255)
        method: 分離方法
            - "distance_transform": 距離變換 + 分水嶺
            - "skeleton": 骨架化 + 分支點檢測
            - "aggressive_erosion": 激進腐蝕
    """
    if method == "distance_transform":
        return split_by_distance_transform(mask)
    elif method == "skeleton":
        return split_by_skeleton(mask)
    elif method == "aggressive_erosion":
        return split_by_aggressive_erosion(mask)
    else:
        raise ValueError(f"未知方法: {method}")


def split_by_distance_transform(mask: np.ndarray) -> list:
    """
    使用距離變換 + 分水嶺算法分離
    這個方法對於有多個"凸起"的區域效果較好
    """
    print("\n=== 使用距離變換 + 分水嶺算法 ===")

    # 1. 距離變換
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # 2. 找出局部最大值（峰值）
    # 使用閾值來確定"確定前景"
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 3. 找出"確定背景"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # 4. 找出"未知區域"
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. 標記前景區域
    num_labels, markers = cv2.connectedComponents(sure_fg)
    print(f"找到 {num_labels - 1} 個前景區域")

    # 6. 背景標記為 1，前景從 2 開始
    markers = markers + 1
    markers[unknown == 255] = 0

    # 7. 應用分水嶺算法
    # 需要 3 通道圖像
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_3ch, markers)

    # 8. 提取分離的區域
    split_regions = []
    for label in range(2, num_labels + 1):
        region_mask = np.zeros_like(mask)
        region_mask[markers == label] = 255

        # 膨脹回原來的大小
        region_mask = cv2.dilate(region_mask, kernel, iterations=2)
        region_mask = cv2.bitwise_and(region_mask, mask)

        if np.sum(region_mask > 0) > 100:
            split_regions.append((region_mask > 0).astype(np.uint8))

    print(f"分離出 {len(split_regions)} 個區域")
    return split_regions if len(split_regions) > 1 else [(mask > 0).astype(np.uint8)]


def split_by_skeleton(mask: np.ndarray) -> list:
    """
    使用骨架化方法分離
    找到骨架的分支點或細窄連接處
    """
    print("\n=== 使用骨架化方法 ===")

    # 1. 骨架化
    skeleton = cv2.ximgproc.thinning(mask)

    # 2. 找到分支點（連接數 > 2 的點）
    # 使用 hit-or-miss 變換
    # TODO: 實現分支點檢測

    # 暫時返回原始 mask
    return [(mask > 0).astype(np.uint8)]


def split_by_aggressive_erosion(mask: np.ndarray) -> list:
    """
    使用更激進的腐蝕策略
    """
    print("\n=== 使用激進腐蝕 ===")

    # 1. 計算更大的 kernel
    kernel_size = max(7, int(np.sqrt(np.sum(mask > 0)) / 10))
    print(f"Kernel size: {kernel_size}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 2. 更多次的腐蝕
    eroded = cv2.erode(mask, kernel, iterations=5)

    # 3. 連通組件分析
    num_labels, labels = cv2.connectedComponents(eroded)
    print(f"腐蝕後找到 {num_labels - 1} 個區域")

    if num_labels > 2:
        split_regions = []
        for label in range(1, num_labels):
            region_mask = np.zeros_like(mask)
            region_mask[labels == label] = 255

            # 膨脹回原來的大小
            dilated = cv2.dilate(region_mask, kernel, iterations=4)
            dilated = cv2.bitwise_and(dilated, mask)

            if np.sum(dilated > 0) > 100:
                split_regions.append((dilated > 0).astype(np.uint8))

        if len(split_regions) > 1:
            print(f"成功分離出 {len(split_regions)} 個區域")
            return split_regions

    print("無法分離，返回原始 mask")
    return [(mask > 0).astype(np.uint8)]


def visualize_results(original_mask, results_dict):
    """視覺化所有方法的結果"""
    from PIL import Image, ImageDraw, ImageFont

    h, w = original_mask.shape
    num_methods = len(results_dict) + 1  # +1 for original

    # 創建大圖
    combined_w = w * num_methods
    combined_h = h
    combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

    # 原始 mask
    vis_original = np.zeros((h, w, 3), dtype=np.uint8)
    vis_original[original_mask > 0] = [255, 255, 255]
    combined[:, 0:w] = vis_original

    # 各方法結果
    colors = [
        (255, 100, 100),  # 紅色
        (100, 255, 100),  # 綠色
        (100, 100, 255),  # 藍色
        (255, 255, 100),  # 黃色
        (255, 100, 255),  # 紫色
    ]

    for i, (method_name, regions) in enumerate(results_dict.items(), 1):
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for j, region_mask in enumerate(regions):
            color = colors[j % len(colors)]
            vis[region_mask > 0] = color

        combined[:, i*w:(i+1)*w] = vis

    # 保存
    output_path = Path("tests/testcases/real_mask_split_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_img = Image.fromarray(combined)
    combined_img.save(str(output_path))

    print(f"\n視覺化結果已保存至: {output_path}")


def main():
    # 如果提供了圖像路徑，使用該圖像
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 使用測試圖像
        image_path = "tests/testcases/png.png"

    print(f"載入圖像: {image_path}")

    try:
        mask = load_mask_from_image(image_path)
    except Exception as e:
        print(f"錯誤: {e}")
        return

    # 分析 mask
    analyze_mask(mask)

    # 測試不同的分離方法
    results = {}

    # 方法 1: 距離變換
    try:
        results["Distance Transform"] = split_by_distance_transform(mask)
    except Exception as e:
        print(f"距離變換失敗: {e}")
        results["Distance Transform"] = [(mask > 0).astype(np.uint8)]

    # 方法 2: 激進腐蝕
    try:
        results["Aggressive Erosion"] = split_by_aggressive_erosion(mask)
    except Exception as e:
        print(f"激進腐蝕失敗: {e}")
        results["Aggressive Erosion"] = [(mask > 0).astype(np.uint8)]

    # 視覺化結果
    visualize_results(mask, results)

    print("\n=== 總結 ===")
    for method, regions in results.items():
        print(f"{method}: {len(regions)} 個區域")


if __name__ == "__main__":
    main()
