"""
基於 Layout 的漫畫圖片渲染器
步驟:
1. 載入 results/layout/yolo/layout.json (inpaint_region)
2. 載入 results/translation/hunyuan.json (translated_text)
3. 根據 inpaint_region 的 segmentation mask 來填白
4. 將翻譯文字渲染回 mask 上
5. 儲存到 results/render/
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont

from lib.schema import LayoutOutputSchema, TranslationOutputSchema, RegionSchema, TranslatedBoundingBox


class LayoutBasedRenderer:
    """基於 Layout 的圖像渲染器，將翻譯文字渲染到原始漫畫圖片上"""

    def __init__(self, font_path: Optional[str] = None, default_font_size: int = 20):
        """
        初始化渲染器

        Args:
            font_path: 支援繁體中文的字型檔案路徑
            default_font_size: 預設字型大小
        """
        self.font_path = font_path
        self.default_font_size = default_font_size
        self.font_cache = {}

    def apply_inpaint_mask(self, image: np.ndarray, regions: List[RegionSchema]) -> np.ndarray:
        """
        根據 segmentation mask 將區域填白

        Args:
            image: 原始圖片 (cv2 的 BGR 格式)
            regions: Layout 區域列表，包含 segmentation_mask

        Returns:
            填白後的圖片
        """
        result = image.copy()

        for region in regions:
            mask = np.array(region['segmentation_mask'], dtype=np.uint8)

            # 確保 mask 尺寸與圖片一致
            if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                # 如果 mask 尺寸不同，需要調整
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 將 mask 為 1 的區域填白
            result[mask > 0] = [255, 255, 255]

        return result

    def match_translation_to_layout(self,
                                   layout_regions: List[RegionSchema],
                                   translation_bboxes: List[TranslatedBoundingBox]) -> List[Dict]:
        """
        將翻譯文字框與 Layout 區域進行匹配

        Args:
            layout_regions: Layout 區域列表
            translation_bboxes: 翻譯文字框列表

        Returns:
            匹配後的列表，每個元素包含 region 和對應的 translated_text
        """
        matched = []

        for region in layout_regions:
            # region['box'] = [[x1, y1], [x2, y2]]
            region_x1, region_y1 = region['box'][0]
            region_x2, region_y2 = region['box'][1]
            region_center_x = (region_x1 + region_x2) / 2
            region_center_y = (region_y1 + region_y2) / 2

            # 尋找最接近的翻譯文字框
            best_match = None
            min_distance = float('inf')

            for bbox in translation_bboxes:
                # bbox['box'] = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                bbox_points = bbox['box']
                bbox_x1 = min(p[0] for p in bbox_points)
                bbox_y1 = min(p[1] for p in bbox_points)
                bbox_x2 = max(p[0] for p in bbox_points)
                bbox_y2 = max(p[1] for p in bbox_points)
                bbox_center_x = (bbox_x1 + bbox_x2) / 2
                bbox_center_y = (bbox_y1 + bbox_y2) / 2

                # 計算中心點距離
                distance = np.sqrt((region_center_x - bbox_center_x)**2 +
                                 (region_center_y - bbox_center_y)**2)

                if distance < min_distance:
                    min_distance = distance
                    best_match = bbox

            matched.append({
                'region': region,
                'translated_text': best_match['translated_text'] if best_match else '',
                'box': region['box']
            })

        return matched

    def render_text_on_image(self, image: np.ndarray, matched_regions: List[Dict]) -> np.ndarray:
        """
        將翻譯文字渲染到圖片上

        Args:
            image: 填白後的圖片 (cv2 的 BGR 格式)
            matched_regions: 匹配後的區域列表

        Returns:
            渲染了翻譯文字的圖片
        """
        # 轉換為 PIL 格式以獲得更好的文字渲染效果
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        for matched in matched_regions:
            text = matched['translated_text']

            # 跳過空文字
            if not text or not text.strip():
                continue

            # box = [[x1, y1], [x2, y2]]
            x1, y1 = matched['box'][0]
            x2, y2 = matched['box'][1]

            # 渲染文字（帶描邊效果）
            self._render_text_in_box(draw, text, (x1, y1, x2, y2))

        # 轉換回 cv2 格式 (BGR)
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result

    def _render_text_in_box(self, draw: ImageDraw.Draw, text: str,
                           box: tuple):
        """
        在邊界框內渲染文字

        Args:
            draw: PIL ImageDraw 物件
            text: 要渲染的文字
            box: (x1, y1, x2, y2) 邊界框
        """
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        # 判斷文字應該垂直還是水平排列
        is_vertical = True

        # 計算字型大小
        font_size = self._calculate_font_size(text, box_width, box_height, is_vertical)
        font = self._get_font(font_size)

        if is_vertical:
            self._draw_vertical_text(draw, text, box, font)
        else:
            self._draw_horizontal_text(draw, text, box, font)

    def _draw_vertical_text(self, draw: ImageDraw.Draw, text: str,
                           box: tuple, font: ImageFont.ImageFont):
        """
        垂直繪製文字 (漫畫格式：由上到下，由右到左)

        Args:
            draw: PIL ImageDraw 物件
            text: 要繪製的文字
            box: 邊界框 (x1, y1, x2, y2)
            font: 要使用的字型
        """
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        # 增加邊距，不要緊貼框線
        padding = int(box_width * 0.15)
        current_x = x2 - padding - font.size
        current_y = y1 + padding

        char_height = font.size + 4
        chars_per_column = max(1, int((box_height - 2 * padding) / char_height))

        lines = []
        temp_line = ""
        for char in text:
            if char == '\n' or len(temp_line) >= chars_per_column:
                lines.append(temp_line)
                temp_line = ""
                if char == '\n': continue
            temp_line += char
        if temp_line: lines.append(temp_line)

        for line in lines:
            line_y = current_y
            for char in line:
                # 智能描邊：黑色文字 + 白色粗描邊
                draw.text((current_x, line_y), char, font=font,
                          fill=(0, 0, 0),              # 字體顏色：黑色
                          stroke_width=5,              # 粗描邊：5 像素
                          stroke_fill=(255, 255, 255)) # 描邊顏色：白色
                line_y += char_height
            current_x -= (font.size + 8) # 移至左側下一列

    def _draw_horizontal_text(self, draw: ImageDraw.Draw, text: str,
                             box: tuple, font: ImageFont.ImageFont):
        """
        水平繪製文字 (由左到右)

        Args:
            draw: PIL ImageDraw 物件
            text: 要繪製的文字
            box: 邊界框 (x1, y1, x2, y2)
            font: 要使用的字型
        """
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        # 簡單的文字換行
        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = current_line + word if not current_line else current_line + " " + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]

            if line_width <= box_width - 10:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # 如果文字仍然無法容納，就使用整段文字
        if not lines:
            lines = [text]

        # 在邊界框中居中繪製文字
        line_height = font.size + 4
        total_height = len(lines) * line_height
        current_y = y1 + (box_height - total_height) // 2

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            text_x = x1 + (box_width - line_width) // 2

            # 智能描邊：黑色文字 + 白色粗描邊
            draw.text((text_x, current_y), line, font=font,
                      fill=(0, 0, 0),              # 字體顏色：黑色
                      stroke_width=5,              # 粗描邊：5 像素
                      stroke_fill=(255, 255, 255)) # 描邊顏色：白色
            current_y += line_height

    def _calculate_font_size(self, text: str, box_width: int, box_height: int,
                           is_vertical: bool) -> int:
        """
        計算適當的字型大小

        Args:
            text: 要渲染的文字
            box_width: 邊界框寬度
            box_height: 邊界框高度
            is_vertical: 是否為垂直文字

        Returns:
            字型大小
        """
        if is_vertical:
            # 垂直文字，基於寬度和字元數量
            size = min(int(box_width * 0.7), int(box_height / max(len(text), 1) * 1.2), 30)
        else:
            # 水平文字，基於高度
            size = min(int(box_height * 0.6), int(box_width / max(len(text), 1) * 1.5), 30)

        return max(size, 12)  # 最小字型大小

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """
        取得字型（帶快取）

        Args:
            size: 字型大小

        Returns:
            PIL ImageFont 物件
        """
        if size in self.font_cache:
            return self.font_cache[size]

        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, size)
            else:
                # 嘗試常見的中文字型路徑
                font_paths = [
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/arphic/uming.ttc",
                    "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
                    "C:\\Windows\\Fonts\\msyh.ttc",  # Windows
                ]

                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, size)
                        break
                    except:
                        continue
                else:
                    # 後備方案
                    font = ImageFont.load_default()
        except Exception as e:
            print(f"警告: 無法載入字型，使用預設字型: {e}")
            font = ImageFont.load_default()

        self.font_cache[size] = font
        return font


def process_layout_based_render(
    layout_json_path: str,
    translation_json_path: str,
    output_dir: str,
    font_path: Optional[str] = None
):
    """
    處理基於 Layout 的渲染流程

    Args:
        layout_json_path: Layout JSON 檔案路徑 (例如: results/layout/yolo/layout.json)
        translation_json_path: 翻譯 JSON 檔案路徑 (例如: results/translation/hunyuan.json)
        output_dir: 渲染圖片的輸出目錄 (例如: results/render/)
        font_path: 自訂字型檔案的路徑
    """
    layout_json_path = Path(layout_json_path)
    translation_json_path = Path(translation_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入 Layout JSON
    print(f"載入 Layout JSON 檔案: {layout_json_path}")
    with open(layout_json_path, 'r', encoding='utf-8') as f:
        layout_data: List[LayoutOutputSchema] = json.load(f)

    # 載入翻譯 JSON
    print(f"載入翻譯 JSON 檔案: {translation_json_path}")
    with open(translation_json_path, 'r', encoding='utf-8') as f:
        translation_data: List[TranslationOutputSchema] = json.load(f)

    print(f"找到 {len(layout_data)} 張圖片的 Layout 資料")
    print(f"找到 {len(translation_data)} 張圖片的翻譯資料")

    # 建立翻譯資料的索引 (根據原始圖片路徑)
    # translation 的 source 是 cropped image path，需要從中提取原始圖片名稱
    translation_map = {}
    for trans_item in translation_data:
        # source 格式: "results/layout/cropped/109994537_p0_master1200_bbox_000.png"
        # 提取原始圖片名稱: 109994537_p0_master1200
        source = Path(trans_item['source']).stem  # 去掉 .png
        # 去掉 _bbox_XXX 後綴
        original_name = '_'.join(source.split('_')[:-2])  # 109994537_p0_master1200

        if original_name not in translation_map:
            translation_map[original_name] = []

        translation_map[original_name].append(trans_item)

    # 初始化渲染器
    renderer = LayoutBasedRenderer(font_path=font_path)

    # 處理每張圖片
    for idx, layout_item in enumerate(layout_data):
        source_path = layout_item['source']
        regions = layout_item['region_result']

        print(f"\n[{idx + 1}/{len(layout_data)}] 處理中: {source_path}")

        # 載入原始圖片
        full_source_path = Path(source_path)
        if not full_source_path.exists():
            print(f"  警告: 找不到來源圖片: {full_source_path}")
            continue

        image = cv2.imread(str(full_source_path))
        if image is None:
            print(f"  錯誤: 無法讀取圖片: {full_source_path}")
            continue

        print(f"  圖片大小: {image.shape[1]}x{image.shape[0]}")
        print(f"  Layout 區域數量: {len(regions)}")

        # 步驟 1: 根據 segmentation mask 填白
        inpainted_image = renderer.apply_inpaint_mask(image, regions)
        print(f"  已完成區域填白")

        # 步驟 2: 匹配翻譯文字
        original_name = full_source_path.stem  # 109994537_p0_master1200
        translation_items = translation_map.get(original_name, [])

        if not translation_items:
            print(f"  警告: 找不到對應的翻譯資料: {original_name}")
            # 仍然儲存填白後的圖片
            output_filename = full_source_path.stem + "_rendered.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), inpainted_image)
            print(f"  已儲存填白圖片至: {output_path}")
            continue

        # 收集所有翻譯文字框
        all_translation_bboxes = []
        for trans_item in translation_items:
            all_translation_bboxes.extend(trans_item['bounding_boxes'])

        print(f"  找到 {len(all_translation_bboxes)} 個翻譯文字框")

        # 匹配 Layout 區域與翻譯文字框
        matched_regions = renderer.match_translation_to_layout(regions, all_translation_bboxes)
        print(f"  已完成區域與翻譯文字匹配")

        # 步驟 3: 渲染翻譯文字
        result_image = renderer.render_text_on_image(inpainted_image, matched_regions)
        print(f"  已完成文字渲染")

        # 步驟 4: 儲存結果
        output_filename = full_source_path.stem + "_rendered.jpg"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), result_image)
        print(f"  已儲存至: {output_path}")

    print(f"\n全部完成！輸出已儲存至: {output_dir}")


def main():
    """主要 CLI 進入點"""
    parser = argparse.ArgumentParser(
        description="基於 Layout 的漫畫圖片渲染器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本用法
  python -m lib.render.layout_render \\
      --layout results/layout/yolo/layout.json \\
      --translation results/translation/hunyuan.json \\
      --output results/render/

  # 使用自訂字型
  python -m lib.render.layout_render \\
      --layout results/layout/yolo/layout.json \\
      --translation results/translation/hunyuan.json \\
      --output results/render/ \\
      --font /path/to/font.ttf
        """
    )

    parser.add_argument(
        '--layout',
        type=str,
        required=True,
        help='Layout JSON 檔案路徑 (例如: results/layout/yolo/layout.json)'
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

    process_layout_based_render(
        layout_json_path=args.layout,
        translation_json_path=args.translation,
        output_dir=args.output,
        font_path=args.font
    )


if __name__ == "__main__":
    main()
