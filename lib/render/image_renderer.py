"""
將翻譯後的文字渲染到漫畫圖片上的圖像渲染器
步驟:
1. 載入圖片與對應 polygon 和翻譯文字
2. 將 polygon 內的原文字區域以白色填充
3. 在 polygon 內渲染翻譯後的文字
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


class TranslatedImageRenderer:
    """將翻譯後的繁體中文文字渲染到原始漫畫圖片上"""

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

    def get_text_mask_threshold(self, roi: np.ndarray) -> np.ndarray:
        """
        偵測 ROI 中的文字區域，返回文字遮罩

        Args:
            roi: 感興趣區域 (BGR 格式)

        Returns:
            文字遮罩 (255=文字, 0=背景)
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 使用 Otsu 自動閾值，加上 INV 是因為我們要文字為白(255)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 膨脹處理：向外擴張 2 像素，確保覆蓋文字邊緣的半透明區域
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def polygon_to_bbox(self, polygon: List[List[int]]) -> Tuple[int, int, int, int]:
        """
        將多邊形轉換為邊界框

        Args:
            polygon: [x, y] 座標列表

        Returns:
            (x1, y1, x2, y2) 邊界框
        """
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    def render_text_on_image(self, image: np.ndarray, bounding_boxes: List[dict]) -> np.ndarray:
        """
        將所有翻譯文字渲染到圖片上

        Args:
            image: 原始圖片 (cv2 的 BGR 格式)
            bounding_boxes: 包含 'box' 和 'translated_text' 的邊界框字典列表

        Returns:
            渲染了翻譯文字的圖片
        """
        # 第一階段：使用 cv2 偵測文字區域並將背景變白
        result_image = image.copy()

        for bbox_data in bounding_boxes:
            polygon = bbox_data['box']
            translated_text = bbox_data.get('translated_text', '')

            # 跳過空文字
            if not translated_text or not translated_text.strip():
                continue

            # 取得 ROI 區域
            x1, y1, x2, y2 = self.polygon_to_bbox(polygon)

            # 確保座標在圖片範圍內
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = result_image[y1:y2, x1:x2]

            # 偵測文字區域
            text_mask = self.get_text_mask_threshold(roi)

            # 將文字區域變成白色
            result_image[y1:y2, x1:x2][text_mask == 255] = [255, 255, 255]

        # 第二階段：轉換為 PIL 格式渲染翻譯文字
        pil_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        for bbox_data in bounding_boxes:
            polygon = bbox_data['box']
            translated_text = bbox_data.get('translated_text', '')

            # 跳過空文字
            if not translated_text or not translated_text.strip():
                continue

            # 將多邊形轉換為邊界框
            x1, y1, x2, y2 = self.polygon_to_bbox(polygon)

            # 渲染文字
            self._render_text_in_box(draw, translated_text, (x1, y1, x2, y2))

        # 轉換回 cv2 格式 (BGR)
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result

    def _render_text_in_box(self, draw: ImageDraw.Draw, text: str,
                           box: Tuple[int, int, int, int]):
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
        is_vertical = box_height > box_width * 1.5

        # 計算字型大小
        font_size = self._calculate_font_size(text, box_width, box_height, is_vertical)
        font = self._get_font(font_size)

        if is_vertical:
            self._draw_vertical_text(draw, text, box, font)
        else:
            self._draw_horizontal_text(draw, text, box, font)

    def _draw_vertical_text(self, draw: ImageDraw.Draw, text: str,
                           box: Tuple[int, int, int, int], font: ImageFont.ImageFont):
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
                # 使用 stroke_width 增加白色描邊，讓文字更立體
                draw.text((current_x, line_y), char, font=font, fill=(0, 0, 0),
                          stroke_width=2, stroke_fill=(255, 255, 255))
                line_y += char_height
            current_x -= (font.size + 8) # 移至左側下一列

    def _draw_horizontal_text(self, draw: ImageDraw.Draw, text: str,
                             box: Tuple[int, int, int, int], font: ImageFont.ImageFont):
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

            draw.text((text_x, current_y), line, font=font, fill=(0, 0, 0))
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


def process_translated_json(json_path: str, output_dir: Optional[str] = None,
                           font_path: Optional[str] = None):
    """
    處理翻譯後的 JSON 檔案並生成輸出圖片

    Args:
        json_path: translated_sakura.json 的路徑
        output_dir: 渲染圖片的輸出目錄 (預設：與 json 相同的目錄)
        font_path: 自訂字型檔案的路徑
    """
    json_path = Path(json_path)

    if output_dir is None:
        output_dir = json_path.parent / "rendered"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入 JSON
    print(f"載入 JSON 檔案: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"找到 {len(data)} 張圖片需要處理")

    # 初始化渲染器
    renderer = TranslatedImageRenderer(font_path=font_path)

    # 處理每張圖片
    for idx, item in enumerate(data):
        source_path = item['source']
        image_path = item.get('image_path', '')
        bounding_boxes = item['bounding_boxes']

        print(f"\n[{idx + 1}/{len(data)}] 處理中: {source_path}")

        # 從來源路徑載入原始圖片
        full_source_path = Path(source_path)
        if not full_source_path.exists():
            print(f"  警告: 找不到來源圖片: {full_source_path}")
            continue

        image = cv2.imread(str(full_source_path))
        if image is None:
            print(f"  錯誤: 無法讀取圖片: {full_source_path}")
            continue

        print(f"  圖片大小: {image.shape[1]}x{image.shape[0]}")
        print(f"  文字框數量: {len(bounding_boxes)}")

        # 渲染翻譯文字
        result_image = renderer.render_text_on_image(image, bounding_boxes)

        # 生成輸出檔名
        output_filename = full_source_path.stem + "_translated.jpg"
        output_path = output_dir / output_filename

        # 儲存結果
        cv2.imwrite(str(output_path), result_image)
        print(f"  已儲存至: {output_path}")

    print(f"\n全部完成！輸出已儲存至: {output_dir}")


def main():
    """主要 CLI 進入點"""
    parser = argparse.ArgumentParser(
        description="從 JSON 檔案將翻譯文字渲染到漫畫圖片上",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本用法 (輸出到 results/2/rendered/)
  python -m utils.image_renderer results/2/translated_sakura.json

  # 指定自訂輸出目錄
  python -m utils.image_renderer results/2/translated_sakura.json -o output/

  # 使用自訂字型
  python -m utils.image_renderer results/2/translated_sakura.json -f /path/to/font.ttf
        """
    )

    parser.add_argument(
        '--json_path',
        type=str,
        help='翻譯後 JSON 檔案的路徑 (例如: translated_sakura.json)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='渲染圖片的輸出目錄 (預設: <json_dir>/rendered/)'
    )

    parser.add_argument(
        '-f', '--font',
        type=str,
        default=None,
        help='支援繁體中文的自訂字型檔案路徑'
    )

    args = parser.parse_args()

    process_translated_json(
        json_path=args.json_path,
        output_dir=args.output,
        font_path=args.font
    )


if __name__ == "__main__":
    main()
