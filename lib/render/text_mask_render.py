"""
基於 Text Mask 的漫畫圖片渲染器 (重構版本)
"""

import json
import cv2
from pathlib import Path
from typing import List, Dict, Optional, TypedDict
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from lib.schema import LayoutOutputSchema, TranslationOutputSchema
from lib.text_mask.yolo import decompress_mask_from_base64_png


class TextRegion(TypedDict):
    """文字區域資訊"""
    translated_text: str  # 翻譯後的文字
    segmentation_mask: np.ndarray  # 二值化的 segmentation mask


class ImgInfo(TypedDict):
    """圖片資訊"""
    source_image: str  # 原始圖片路徑
    text_regions: List[TextRegion]  # 文字區域列表


class TextMaskBasedRenderer:
    """基於 Text Mask 的圖像渲染器"""

    def __init__(self, font_path: Optional[str] = None, default_font_size: int = 20):
        self.font_path = font_path
        self.default_font_size = default_font_size
        self.font_cache = {}

    def render_images_batch(self, img_infos: List[ImgInfo], output_dir: str):
        """3. 批次渲染多張圖片"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\n開始批次渲染 {len(img_infos)} 張圖片")

        for idx, img_info in enumerate(img_infos, 1):
            print(f"\n[{idx}/{len(img_infos)}] 處理中: {img_info['source_image']}")
            self.render_image_single(img_info, output_dir)

        print(f"\n全部完成！輸出已儲存至: {output_dir}")

    def render_image_single(self, img_info: ImgInfo, output_dir: str):
        """5. 渲染單張圖片"""
        source_image_path = img_info['source_image']
        text_regions = img_info['text_regions']

        # 6. 載入原始圖片
        source_path = Path(source_image_path)
        if not source_path.exists():
            print(f"  警告: 找不到來源圖片: {source_path}")
            return

        img = Image.open(source_path)
        print(f"  圖片大小: {img.size}")
        print(f"  文字區域數量: {len(text_regions)}")

        # 7. 根據所有 segmentation_mask 填白
        for region in text_regions:
            img = self.apply_inpaint(img, region['segmentation_mask'])
        print(f"  已完成區域填白")

        # 8. 渲染所有翻譯文字到文字區域
        img = self.render_text_in_text_regions(img, text_regions)
        print(f"  已完成文字渲染")

        # 12. 儲存渲染後的圖片
        output_filename = source_path.stem + "_rendered.jpg"
        output_path = Path(output_dir) / output_filename
        img.save(str(output_path), quality=95)
        print(f"  已儲存至: {output_path}")

    def apply_inpaint(self, img: Image.Image, segmentation_mask: np.ndarray) -> Image.Image:
        """7. 根據 segmentation mask 將區域填白"""
        img_array = np.array(img)

        if segmentation_mask.shape[0] != img_array.shape[0] or segmentation_mask.shape[1] != img_array.shape[1]:
            mask = cv2.resize(segmentation_mask, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = segmentation_mask

        img_array[mask > 0] = [255, 255, 255]
        return Image.fromarray(img_array)

    def render_text_in_text_regions(self, img: Image.Image, text_regions: List[TextRegion]) -> Image.Image:
        """8. 在所有文字區域渲染翻譯文字"""
        for region in text_regions:
            img = self.render_text_in_each_region(img, region['segmentation_mask'], region['translated_text'])
        return img

    def render_text_in_each_region(self, img: Image.Image, segmentation_mask: np.ndarray, translated_text: str) -> Image.Image:
        """10. 根據 segmentation mask 來渲染文字"""
        if not translated_text or not translated_text.strip():
            return img

        img_array = np.array(img)

        if segmentation_mask.shape[0] != img_array.shape[0] or segmentation_mask.shape[1] != img_array.shape[1]:
            mask = cv2.resize(segmentation_mask, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = segmentation_mask

        # 透過 Morphological Split 將連在一起的 speech bubbles 分離
        split_regions = self._split_connected_regions(mask)
        print(f"    分離後區域數: {len(split_regions)} (原始: 1)")

        # 根據區域大小和換行符號分配文字
        text_assignments = self._assign_text_to_regions(translated_text, split_regions)
        #if translated_text == "來吧！\n氣氛越來越熱烈了！":
        #    raise "1"
        text_layer = Image.new('RGB', img.size, (255, 255, 255))
        draw = ImageDraw.Draw(text_layer)

        # 為每個分離的區域渲染對應的文字
        for region_mask, region_text in text_assignments:
            if region_text.strip():
                self._render_text_in_mask(draw, region_text, region_mask)

        text_layer_np = np.array(text_layer)
        img_array[mask > 0] = text_layer_np[mask > 0]

        return Image.fromarray(img_array)

    def _render_text_in_mask(self, draw: ImageDraw.Draw, text: str, mask: np.ndarray):
        """11. 根據 segmentation mask 的多邊形形狀渲染文字"""
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        mask_width = x_max - x_min
        mask_height = y_max - y_min

        is_vertical = True
        font_size = self._calculate_font_size(text, mask_width, mask_height, is_vertical)
        font = self._get_font(font_size)

        if is_vertical:
            self._draw_vertical_text_in_mask(draw, text, mask, font, x_min, y_min, x_max, y_max)

    def _draw_vertical_text_in_mask(self, draw: ImageDraw.Draw, text: str, mask: np.ndarray,
                                    font: ImageFont.ImageFont, x_min: int, y_min: int, x_max: int, y_max: int):
        """在 mask 區域內垂直繪製文字"""
        mask_width = x_max - x_min
        mask_height = y_max - y_min
        padding = int(mask_width * 0.15)
        char_height = font.size + 4
        chars_per_column = max(1, int((mask_height - 2 * padding) / char_height))

        lines = []
        temp_line = ""
        for char in text:
            if char == '\n' or len(temp_line) >= chars_per_column:
                if temp_line:
                    lines.append(temp_line)
                temp_line = ""
                if char == '\n':
                    continue
            temp_line += char
        if temp_line:
            lines.append(temp_line)

        current_x = x_max - padding - font.size

        for line in lines:
            if current_x < 0 or current_x >= mask.shape[1]:
                break

            column_mask = mask[:, int(current_x):int(current_x + font.size)]
            if column_mask.size > 0:
                y_indices = np.where(np.any(column_mask > 0, axis=1))[0]
                if len(y_indices) > 0:
                    col_y_min = max(y_indices[0], y_min)
                    col_y_max = min(y_indices[-1], y_max)
                else:
                    col_y_min = y_min
                    col_y_max = y_max
            else:
                col_y_min = y_min
                col_y_max = y_max

            line_y = col_y_min + padding

            for char in line:
                if line_y + char_height > col_y_max:
                    break

                draw.text((current_x, line_y), char, font=font, fill=(0, 0, 0), stroke_width=5, stroke_fill=(255, 255, 255))
                line_y += char_height

            current_x -= (font.size + 8)
            if current_x < x_min:
                break

    def _calculate_font_size(self, text: str, box_width: int, box_height: int, is_vertical: bool) -> int:
        """計算適當的字型大小"""
        if is_vertical:
            size = min(int(box_width * 0.7), int(box_height / max(len(text), 1) * 1.2), 30)
        else:
            size = min(int(box_height * 0.6), int(box_width / max(len(text), 1) * 1.5), 30)
        return max(size, 12)

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """取得字型（帶快取）"""
        if size in self.font_cache:
            return self.font_cache[size]

        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, size)
            else:
                font_paths = [
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/arphic/uming.ttc",
                    "/System/Library/Fonts/STHeiti Light.ttc",
                    "C:\\Windows\\Fonts\\msyh.ttc",
                ]

                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, size)
                        break
                    except:
                        continue
                else:
                    font = ImageFont.load_default()
        except Exception as e:
            print(f"警告: 無法載入字型，使用預設字型: {e}")
            font = ImageFont.load_default()

        self.font_cache[size] = font
        return font

    def _split_connected_regions(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        使用形態學操作分離連在一起的 speech bubbles

        Args:
            mask: 二值化的 segmentation mask

        Returns:
            分離後的區域列表，每個元素是一個獨立的 mask
        """
        # 確保 mask 是二值化的 uint8 格式
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        # 使用連通組件分析找出所有獨立的區域 (背景: label=0、連通區域1 label=1、...)
        num_labels, labels = cv2.connectedComponents(mask_uint8)

        # 將每個連通組件轉換為獨立的 mask
        split_regions = []
        for label in range(1, num_labels):  # 跳過 label 0 (背景)
            region_mask = (labels == label).astype(np.uint8)
            split_regions.append(region_mask)

        # 如果只有一個區域，嘗試使用形態學操作進一步分離
        if len(split_regions) == 1:
            split_regions = self._morphological_split(mask_uint8)

        # 按區域大小排序（從大到小），因為通常較大的區域包含較多文字
        split_regions.sort(key=lambda m: np.sum(m > 0), reverse=True)

        return split_regions

    def _morphological_split(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        使用形態學操作嘗試分離單一大區域

        優先使用距離變換 + 分水嶺算法（更強大），失敗時回退到腐蝕方法

        Args:
            mask: 二值化的 mask (uint8, 0-255)

        Returns:
            分離後的區域列表
        """
        # 方法 1: 嘗試使用距離變換 + 分水嶺算法
        split_regions = self._split_by_distance_transform(mask)
        if len(split_regions) > 1:
            return split_regions

        # 方法 2: 回退到腐蝕方法
        # 使用腐蝕操作來分離連接的區域
        kernel_size = max(5, int(np.sqrt(np.sum(mask > 0)) / 15))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 腐蝕（使用 3 次迭代來更好地分離）
        eroded = cv2.erode(mask, kernel, iterations=3)

        # 找出連通組件
        num_labels, labels = cv2.connectedComponents(eroded)

        # 如果成功分離出多個區域
        if num_labels > 2:  # 超過 1 個區域（排除背景）
            split_regions = []
            for label in range(1, num_labels):
                # 使用原始 mask 的範圍，但根據腐蝕後的標籤來分配
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

    def _split_by_distance_transform(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        使用距離變換 + 分水嶺算法分離區域

        這個方法對於有多個"凸起"或"峰值"的區域效果很好
        適合分離形狀複雜但有明顯分離趨勢的區域

        Args:
            mask: 二值化的 mask (uint8, 0-255)

        Returns:
            分離後的區域列表
        """
        # Debug: 保存原始 mask
        #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        #debug_mask_img = Image.fromarray(mask)
        #debug_mask_img.save(f'/tmp2/howard/japcomic2twcomic/debug/debug_distance_transform_input_mask.png')
        # 1. 距離變換：計算每個前景像素到最近背景像素的距離
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Debug: 保存 dist_transform 圖像和數值
        #dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #debug_dist_img = Image.fromarray(dist_normalized)
        #debug_dist_img.save(f'/tmp2/howard/japcomic2twcomic/debug/debug_distance_transform.png')
        # 保存原始數值資料 (numpy array)
        #np.save(f'/tmp2/howard/japcomic2twcomic/debug/debug_distance_transform.npy', dist_transform)
        
        # 2. 找出"確定前景"（距離較遠的點，通常是區域的中心）
        # 使用自適應閾值，根據最大距離的比例
        threshold_ratio = 0.3  # 可調整：0.2-0.4 之間
        _, sure_fg = cv2.threshold(
            dist_transform,
            threshold_ratio * dist_transform.max(),
            255,
            cv2.THRESH_BINARY
        )
        sure_fg = np.uint8(sure_fg)

        # 3. 找出"確定背景"（原始 mask 擴張後的區域）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=3)

        # 4. 找出"未知區域"（確定背景 - 確定前景）
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 5. 標記前景區域（連通組件標記）
        num_labels, markers = cv2.connectedComponents(sure_fg)

        # 如果只有一個前景區域，無法分離
        if num_labels <= 2:  # 1 個背景 + 1 個前景
            return [(mask > 0).astype(np.uint8)]

        # 6. 調整標記：背景為 1，前景從 2 開始，未知區域為 0
        markers = markers + 1
        markers[unknown == 255] = 0

        # 7. 應用分水嶺算法（需要 3 通道圖像）
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(mask_3ch, markers)

        # 8. 提取分離的區域
        split_regions = []
        for label in range(2, num_labels + 1):
            region_mask = np.zeros_like(mask)
            region_mask[markers == label] = 255

            # 膨脹一下以恢復邊界
            region_mask = cv2.dilate(region_mask, kernel, iterations=2)

            # 確保不超出原始 mask 的範圍
            region_mask = cv2.bitwise_and(region_mask, mask)

            # 過濾太小的區域
            if np.sum(region_mask > 0) > 100:
                split_regions.append((region_mask > 0).astype(np.uint8))

        return split_regions if len(split_regions) > 1 else [(mask > 0).astype(np.uint8)]

    def _assign_text_to_regions(
        self,
        text: str,
        regions: List[np.ndarray]
    ) -> List[tuple[np.ndarray, str]]:
        """
        根據區域大小和換行符號將文字分配到各個區域

        Args:
            text: 要分配的翻譯文字
            regions: 分離後的區域列表

        Returns:
            [(region_mask, assigned_text), ...] 的列表
        """
        # 如果只有一個區域，直接返回
        if len(regions) == 1:
            return [(regions[0], text)]

        # 根據換行符號分割文字
        text_parts = [part.strip() for part in text.split('\n') if part.strip()]

        # 如果沒有換行符號，根據區域大小比例分配文字
        if len(text_parts) <= 1:
            text_parts = self._split_text_by_region_sizes(text, regions)

        # 確保 text_parts 和 regions 數量匹配
        assignments = []

        if len(text_parts) >= len(regions):
            # 文字部分較多，將多餘的文字合併到最後一個區域
            for i, region in enumerate(regions):
                if i < len(regions) - 1:
                    assignments.append((region, text_parts[i]))
                else:
                    # 最後一個區域包含剩餘所有文字
                    remaining_text = '\n'.join(text_parts[i:])
                    assignments.append((region, remaining_text))
        else:
            # 區域較多，每個文字部分分配到一個區域
            for i, text_part in enumerate(text_parts):
                assignments.append((regions[i], text_part))
            # 剩餘的區域保持空白
            for i in range(len(text_parts), len(regions)):
                assignments.append((regions[i], ""))

        return assignments

    def _split_text_by_region_sizes(
        self,
        text: str,
        regions: List[np.ndarray]
    ) -> List[str]:
        """
        根據區域大小比例分割文字

        Args:
            text: 要分割的文字
            regions: 區域列表

        Returns:
            分割後的文字列表
        """
        # 計算每個區域的大小
        region_sizes = [np.sum(region > 0) for region in regions]
        total_size = sum(region_sizes)

        # 計算每個區域應該分配的字元數
        text_length = len(text)
        char_counts = []
        for size in region_sizes:
            proportion = size / total_size
            char_count = max(1, int(text_length * proportion))
            char_counts.append(char_count)

        # 根據字元數分割文字
        text_parts = []
        start_idx = 0
        for i, char_count in enumerate(char_counts):
            if i == len(char_counts) - 1:
                # 最後一個區域包含剩餘所有文字
                text_parts.append(text[start_idx:])
            else:
                end_idx = min(start_idx + char_count, len(text))
                text_parts.append(text[start_idx:end_idx])
                start_idx = end_idx

        return text_parts

def load_data(text_mask_json_path: str, translation_json_path: str) -> List[ImgInfo]:
    """
    1. 載入 Text Mask 和翻譯資料
    2. 解碼 segmentation mask
    3. 依照圖片、區域分組

    Args:
        text_mask_json_path: Text Mask JSON 檔案路徑
        translation_json_path: 翻譯 JSON 檔案路徑

    Returns:
        List[ImgInfo]: 每個元素包含:
            - source_image: 原始圖片路徑
            - text_regions: 文字區域列表，每個區域包含:
                - translated_text: 翻譯後的文字
                - segmentation_mask: 二值化的 segmentation mask
    """
    text_mask_json_path = Path(text_mask_json_path)
    translation_json_path = Path(translation_json_path)

    # 驗證檔案存在
    if not text_mask_json_path.exists():
        raise FileNotFoundError(f"找不到 Text Mask JSON 檔案: {text_mask_json_path}")
    if not translation_json_path.exists():
        raise FileNotFoundError(f"找不到翻譯 JSON 檔案: {translation_json_path}")

    # 載入 Text Mask JSON
    print(f"載入 Text Mask JSON 檔案: {text_mask_json_path}")
    with open(text_mask_json_path, 'r', encoding='utf-8') as f:
        text_mask_data: List[LayoutOutputSchema] = json.load(f)

    # 載入翻譯 JSON
    print(f"載入翻譯 JSON 檔案: {translation_json_path}")
    with open(translation_json_path, 'r', encoding='utf-8') as f:
        translation_data: List[TranslationOutputSchema] = json.load(f)

    print(f"找到 {len(text_mask_data)} 張圖片的 Text Mask 資料")
    print(f"找到 {len(translation_data)} 張圖片的翻譯資料")

    # 解碼所有的 segmentation_mask 從 Base64 PNG 格式轉為 binary mask
    print("解碼 segmentation_mask...")
    for item in text_mask_data:
        for region in item['region_result']:
            # 將 Base64-encoded PNG 解碼為 binary mask (numpy array)
            region['segmentation_mask'] = decompress_mask_from_base64_png(region['segmentation_mask'])
    print("segmentation_mask 解碼完成")

    # 建立 cropped_image_path 到 (source_image, segmentation_mask) 的映射
    crop_path_to_mask: Dict[str, tuple[str, np.ndarray]] = {}
    for text_mask_item in text_mask_data:
        source_image = text_mask_item['source']
        regions = text_mask_item.get('region_result', [])

        for region in regions:
            crop_path = region['cropped_image_path']
            segmentation_mask = region['segmentation_mask']
            crop_path_to_mask[crop_path] = (source_image, segmentation_mask)

    print(f"建立了 {len(crop_path_to_mask)} 個 cropped_image_path 到 mask 的映射")

    # 按原始圖片分組整理資料
    # 結構: {source_image: List[TextRegion]}
    img_to_regions: Dict[str, List[TextRegion]] = {}
    matched_count = 0
    unmatched_count = 0

    for trans_item in translation_data:
        source_path = trans_item.get('source', '')  # 這是 cropped_image_path
        bounding_boxes = trans_item.get('bounding_boxes', [])

        if source_path in crop_path_to_mask:
            source_image, segmentation_mask = crop_path_to_mask[source_path]

            # 初始化該圖片的 regions 列表
            if source_image not in img_to_regions:
                img_to_regions[source_image] = []

            # 為每個翻譯文字框建立一個 TextRegion
            for bbox in bounding_boxes:
                text_region: TextRegion = {
                    'translated_text': bbox['translated_text'],
                    'segmentation_mask': segmentation_mask
                }
                img_to_regions[source_image].append(text_region)
                matched_count += 1
        else:
            # 如果找不到對應的 mask，記錄警告但繼續處理
            print(f"  警告: 找不到對應的 text_mask region: {source_path}")
            unmatched_count += 1

    print(f"成功匹配 {matched_count} 個翻譯文字框")
    if unmatched_count > 0:
        print(f"警告: {unmatched_count} 個翻譯項目找不到對應的 text_mask region")

    # 轉換為 List[ImgInfo] 格式
    result: List[ImgInfo] = []
    for source_image, text_regions in img_to_regions.items():
        img_info: ImgInfo = {
            'source_image': source_image,
            'text_regions': text_regions
        }
        result.append(img_info)

    print(f"整理完成，共 {len(result)} 張圖片")

    return result


def process_render(
    text_mask_json_path: str,
    translation_json_path: str,
    output_dir: str,
    font_path: Optional[str] = None
):
    """
    處理渲染流程

    Args:
        text_mask_json_path: Text Mask JSON 檔案路徑
        translation_json_path: 翻譯 JSON 檔案路徑
        output_dir: 渲染圖片的輸出目錄
        font_path: 字型檔案路徑
    """
    # 1. 載入資料
    img_infos = load_data(text_mask_json_path, translation_json_path)

    # 2. 初始化渲染器
    renderer = TextMaskBasedRenderer(font_path=font_path)

    # 3. 批次渲染圖片
    renderer.render_images_batch(img_infos, output_dir)


def main():
    """主要 CLI 進入點"""
    parser = argparse.ArgumentParser(
        description="基於 Text Mask 的漫畫圖片渲染器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本用法
  python -m lib.render.text_mask_render \\
      --text_mask results/text_mask/yolo/text_mask.json \\
      --translation results/translation/hunyuan.json \\
      --output results/render/

  # 使用自訂字型
  python -m lib.render.text_mask_render \\
      --text_mask results/text_mask/yolo/text_mask.json \\
      --translation results/translation/hunyuan.json \\
      --output results/render/ \\
      --font /path/to/font.ttf
        """
    )

    parser.add_argument(
        '--text_mask',
        type=str,
        required=True,
        help='Text Mask JSON 檔案路徑 (例如: results/text_mask/yolo/text_mask.json)'
    )

    parser.add_argument(
        '--translation',
        type=str,
        required=True,
        help='翻譯 JSON 檔案路徑 (例如: results/translation/hunyuan.json)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='渲染圖片的輸出目錄 (例如: results/render/)'
    )

    parser.add_argument(
        '--font',
        type=str,
        default=None,
        help='支援繁體中文的自訂字型檔案路徑'
    )

    args = parser.parse_args()

    process_render(
        text_mask_json_path=args.text_mask,
        translation_json_path=args.translation,
        output_dir=args.output,
        font_path=args.font
    )


if __name__ == "__main__":
    main()
