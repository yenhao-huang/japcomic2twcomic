"""
在原始圖片上繪製翻譯後的 bounding boxes
用於視覺化檢查翻譯結果

輸入:
- results/translation/hunyuan.json: 翻譯結果 (包含 source, bounding_boxes 等資訊)
- source images: 原始圖片 (從 JSON 中的 source 欄位取得路徑)

輸出:
- results/translation_bbox/: 繪製 bbox 後的圖片
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import argparse
from PIL import Image, ImageDraw, ImageFont

from lib.schema import TranslationOutputSchema, TranslatedBoundingBox


class TranslatedBboxDrawer:
    """在原始圖片上繪製翻譯後的 bounding boxes 的工具類"""

    def __init__(
        self,
        bbox_color: tuple = (0, 255, 0),  # 綠色 (BGR)
        bbox_thickness: int = 2,
        text_color: tuple = (255, 0, 0),  # 藍色 (BGR)
        show_translated_text: bool = True,
        font_path: Optional[str] = None,
        font_size: int = 16
    ):
        """
        初始化 TranslatedBboxDrawer

        Args:
            bbox_color: bbox 邊框顏色 (BGR)
            bbox_thickness: bbox 邊框粗細
            text_color: 文字標籤顏色 (BGR)
            show_translated_text: 是否顯示翻譯後的文字標籤
            font_path: 字型路徑 (用於顯示中文標籤)
            font_size: 字型大小
        """
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness
        self.text_color = text_color
        self.show_translated_text = show_translated_text
        self.font_size = font_size

        # 載入字型 (用於中文標籤)
        if font_path:
            self.font = ImageFont.truetype(font_path, font_size)
        else:
            # 使用預設專案字型
            default_font_path = Path(__file__).parent / "fonts" / "Noto_Sans_CJK_Regular.otf"
            if default_font_path.exists():
                self.font = ImageFont.truetype(str(default_font_path), font_size)
            else:
                print(f"警告: 找不到字型檔案，文字標籤可能無法正確顯示中文")
                self.font = ImageFont.load_default()

    def draw_from_files(
        self,
        translation_json: str,
        output_dir: str
    ):
        """
        從翻譯 JSON 檔案讀取資料並繪製 bbox

        Args:
            translation_json: 翻譯結果 JSON 路徑 (例如: results/translation/hunyuan.json)
            output_dir: 輸出目錄 (例如: results/translation_bbox/)
        """
        # 1. 載入翻譯 JSON
        translation_data = self._load_translation_data(translation_json)

        # 2. 建立輸出目錄
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 3. 批次處理
        self._process_batch(translation_data, output_dir)

    def _load_translation_data(self, json_path: str) -> List[TranslationOutputSchema]:
        """載入翻譯結果 JSON"""
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"找不到翻譯結果 JSON: {json_file}")

        print(f"載入翻譯結果 JSON: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data: List[TranslationOutputSchema] = json.load(f)

        print(f"成功載入 {len(data)} 筆翻譯資料")
        return data

    def _process_batch(
        self,
        translation_data: List[TranslationOutputSchema],
        output_dir: str
    ):
        """批次處理所有圖片"""
        total = len(translation_data)
        print(f"\n開始處理 {total} 筆資料")

        # 將資料按照 source 圖片分組
        source_groups = {}
        for item in translation_data:
            source = item['source']
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(item)

        print(f"共有 {len(source_groups)} 張來源圖片")

        # 處理每張來源圖片
        for idx, (source_path, items) in enumerate(source_groups.items(), 1):
            source_file = Path(source_path)

            if not source_file.exists():
                print(f"[{idx}/{len(source_groups)}] 警告: 找不到來源圖片: {source_path}")
                continue

            # 收集所有 bounding boxes
            all_bboxes = []
            for item in items:
                bboxes = item.get('bounding_boxes', [])
                all_bboxes.extend(bboxes)

            print(f"\n[{idx}/{len(source_groups)}] 處理: {source_file.name}")
            print(f"  Bounding boxes: {len(all_bboxes)}")

            # 繪製 bbox
            output_filename = f"{source_file.stem}_with_translation_bbox{source_file.suffix}"
            self._draw_single_image(
                source_file,
                all_bboxes,
                output_dir,
                output_filename
            )

        print(f"\n全部完成！輸出已儲存至: {output_dir}")

    def _draw_single_image(
        self,
        image_path: Path,
        bboxes: List[TranslatedBoundingBox],
        output_dir: str,
        output_filename: str
    ):
        """
        在單張圖片上繪製 bbox

        Args:
            image_path: 原始圖片路徑
            bboxes: bounding boxes 列表
            output_dir: 輸出目錄
            output_filename: 輸出檔名
        """
        # 讀取圖片 (使用 PIL 以支援中文字型)
        img_cv = cv2.imread(str(image_path))
        if img_cv is None:
            print(f"  警告: 無法讀取圖片: {image_path}")
            return

        print(f"  圖片大小: {img_cv.shape[1]}x{img_cv.shape[0]}")

        # 轉換為 PIL Image (用於繪製中文文字)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 繪製每個 bbox
        for idx, bbox in enumerate(bboxes):
            box = bbox['box']  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            translated_text = bbox.get('translated_text', '').strip()

            # 轉換為 numpy array 並確保是整數座標
            pts = np.array(box, dtype=np.int32)

            # 使用 PIL 繪製多邊形邊框
            pts_list = [(int(pt[0]), int(pt[1])) for pt in box]
            draw.polygon(
                pts_list,
                outline=self._bgr_to_rgb(self.bbox_color),
                width=self.bbox_thickness
            )

            # 可選：繪製翻譯文字標籤
            if self.show_translated_text and translated_text:
                # 取 bbox 左上角作為標籤位置
                label_x = int(box[0][0])
                label_y = int(box[0][1]) - self.font_size - 5

                # 如果標籤會超出圖片上方，則放在 bbox 內部
                if label_y < 0:
                    label_y = int(box[0][1]) + 5

                # 截斷文字以避免過長 (只顯示前 15 個字元)
                display_text = translated_text[:15] + "..." if len(translated_text) > 15 else translated_text
                # 將換行符號替換為空格，使標籤更緊湊
                display_text = display_text.replace('\n', ' ')

                # 使用 PIL 測量文字大小
                try:
                    bbox_text = draw.textbbox((0, 0), display_text, font=self.font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                except:
                    # 回退估算
                    text_width = len(display_text) * self.font_size
                    text_height = self.font_size

                # 繪製半透明白色背景矩形
                padding = 4
                bg_bbox = [
                    (label_x - padding, label_y - padding),
                    (label_x + text_width + padding, label_y + text_height + padding)
                ]

                # 建立半透明背景
                overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(
                    bg_bbox,
                    fill=(255, 255, 255, 200)
                )
                img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img_pil)

                # 繪製文字
                draw.text(
                    (label_x, label_y),
                    display_text,
                    font=self.font,
                    fill=self._bgr_to_rgb(self.text_color)
                )

        # 轉換回 OpenCV 格式並儲存
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        output_filepath = Path(output_dir) / output_filename
        cv2.imwrite(str(output_filepath), img_cv)
        print(f"  已儲存至: {output_filepath}")

    def _bgr_to_rgb(self, bgr_color: tuple) -> tuple:
        """將 BGR 顏色轉換為 RGB"""
        return (bgr_color[2], bgr_color[1], bgr_color[0])


def main():
    """主要 CLI 進入點"""
    parser = argparse.ArgumentParser(
        description="在原始圖片上繪製翻譯後的 bounding boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本用法
  python -m lib.utils.draw_translated_text_bbox_on_image \\
      --translation results/translation/hunyuan.json \\
      --output results/translation_bbox/

  # 自訂 bbox 顏色和粗細
  python -m lib.utils.draw_translated_text_bbox_on_image \\
      --translation results/translation/hunyuan.json \\
      --output results/translation_bbox/ \\
      --bbox_color 255 0 0 \\
      --thickness 3

  # 不顯示翻譯文字標籤
  python -m lib.utils.draw_translated_text_bbox_on_image \\
      --translation results/translation/hunyuan.json \\
      --output results/translation_bbox/ \\
      --no_text_label

  # 使用自訂字型
  python -m lib.utils.draw_translated_text_bbox_on_image \\
      --translation results/translation/hunyuan.json \\
      --output results/translation_bbox/ \\
      --font /path/to/custom/font.ttf \\
      --font_size 20
        """
    )

    parser.add_argument(
        '--translation',
        type=str,
        required=True,
        help='翻譯結果 JSON 檔案路徑 (例如: results/translation/hunyuan.json)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='輸出目錄 (例如: results/translation_bbox/)'
    )

    parser.add_argument(
        '--bbox_color',
        type=int,
        nargs=3,
        default=[0, 255, 0],
        metavar=('B', 'G', 'R'),
        help='Bbox 邊框顏色 (BGR 格式，例如: 0 255 0 為綠色)'
    )

    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Bbox 邊框粗細 (預設: 2)'
    )

    parser.add_argument(
        '--text_color',
        type=int,
        nargs=3,
        default=[255, 0, 0],
        metavar=('B', 'G', 'R'),
        help='文字標籤顏色 (BGR 格式，例如: 255 0 0 為藍色)'
    )

    parser.add_argument(
        '--no_text_label',
        action='store_true',
        help='不顯示翻譯文字標籤'
    )

    parser.add_argument(
        '--font',
        type=str,
        default=None,
        help='字型路徑 (用於顯示中文標籤)。預設使用 lib/utils/fonts/Noto_Sans_CJK_Regular.otf'
    )

    parser.add_argument(
        '--font_size',
        type=int,
        default=16,
        help='文字標籤字型大小 (預設: 16)'
    )

    args = parser.parse_args()

    # 初始化 TranslatedBboxDrawer
    drawer = TranslatedBboxDrawer(
        bbox_color=tuple(args.bbox_color),
        bbox_thickness=args.thickness,
        text_color=tuple(args.text_color),
        show_translated_text=not args.no_text_label,
        font_path=args.font,
        font_size=args.font_size
    )

    # 執行繪製
    drawer.draw_from_files(
        translation_json=args.translation,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
