"""
在渲染完成的圖片上繪製 bounding boxes
用於視覺化檢查文字分配和渲染結果

輸入:
- results/render/: 已渲染的圖片
- results/text_allocate/text_allocate.json: bounding box 資訊

輸出:
- results/render_with_bbox/: 繪製 bbox 後的圖片
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import argparse
from PIL import Image, ImageDraw, ImageFont

from lib.schema import TextAllocaterOutput, AllocatedBoundingBox


class BboxDrawer:
    """在渲染圖片上繪製 bounding boxes 的工具類"""

    def __init__(
        self,
        bbox_color: tuple = (0, 255, 0),  # 綠色 (BGR)
        bbox_thickness: int = 2,
        text_color: tuple = (255, 0, 0),  # 藍色 (BGR)
        show_text_label: bool = True,
        font_path: Optional[str] = None,
        font_size: int = 16
    ):
        """
        初始化 BboxDrawer

        Args:
            bbox_color: bbox 邊框顏色 (BGR)
            bbox_thickness: bbox 邊框粗細
            text_color: 文字標籤顏色 (BGR)
            show_text_label: 是否顯示文字標籤
            font_path: 字型路徑 (用於顯示中文標籤)
            font_size: 字型大小
        """
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness
        self.text_color = text_color
        self.show_text_label = show_text_label
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
        render_dir: str,
        text_allocate_json: str,
        output_dir: str
    ):
        """
        從檔案讀取資料並繪製 bbox

        Args:
            render_dir: 已渲染圖片的目錄 (例如: results/render/)
            text_allocate_json: text_allocate.json 路徑
            output_dir: 輸出目錄 (例如: results/render_with_bbox/)
        """
        # 1. 載入 text_allocate.json
        allocated_data = self._load_allocated_text(text_allocate_json)

        # 2. 建立輸出目錄
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 3. 批次處理
        self._process_batch(allocated_data, render_dir, output_dir)

    def _load_allocated_text(self, json_path: str) -> List[TextAllocaterOutput]:
        """載入 text_allocate.json"""
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"找不到 text_allocate.json: {json_file}")

        print(f"載入 Text Allocate JSON: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data: List[TextAllocaterOutput] = json.load(f)

        print(f"成功載入 {len(data)} 張圖片的資料")
        return data

    def _process_batch(
        self,
        allocated_data: List[TextAllocaterOutput],
        render_dir: str,
        output_dir: str
    ):
        """批次處理所有圖片"""
        total = len(allocated_data)
        print(f"\n開始處理 {total} 張圖片")

        render_path = Path(render_dir)

        for idx, item in enumerate(allocated_data, 1):
            source_image = item['source']
            bboxes = item.get('bounding_boxes', [])

            # 根據 source_image 推測渲染後的圖片路徑
            # 例如: data/2/109994537_p0_master1200.jpg -> 109994537_p0_master1200_rendered.jpg
            source_stem = Path(source_image).stem
            rendered_image_name = f"{source_stem}_rendered.jpg"
            rendered_image_path = render_path / rendered_image_name

            if not rendered_image_path.exists():
                print(f"[{idx}/{total}] 警告: 找不到渲染圖片: {rendered_image_path}")
                continue

            print(f"\n[{idx}/{total}] 處理: {rendered_image_name}")
            print(f"  Bounding boxes: {len(bboxes)}")

            # 繪製 bbox
            self._draw_single_image(
                rendered_image_path,
                bboxes,
                output_dir,
                rendered_image_name
            )

        print(f"\n全部完成！輸出已儲存至: {output_dir}")

    def _draw_single_image(
        self,
        image_path: Path,
        bboxes: List[AllocatedBoundingBox],
        output_dir: str,
        output_filename: str
    ):
        """
        在單張圖片上繪製 bbox

        Args:
            image_path: 渲染後的圖片路徑
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
            text = bbox.get('text', '').strip()

            # 轉換為 numpy array 並確保是整數座標
            pts = np.array(box, dtype=np.int32)

            # 使用 PIL 繪製多邊形邊框
            pts_list = [(int(pt[0]), int(pt[1])) for pt in box]
            draw.polygon(
                pts_list,
                outline=self._bgr_to_rgb(self.bbox_color),
                width=self.bbox_thickness
            )

            # 可選：繪製文字標籤
            if self.show_text_label and text:
                # 取 bbox 左上角作為標籤位置
                label_x = int(box[0][0])
                label_y = int(box[0][1]) - self.font_size - 5

                # 截斷文字以避免過長
                display_text = text[:10] + "..." if len(text) > 10 else text

                # 使用 PIL 測量文字大小
                try:
                    bbox_text = draw.textbbox((0, 0), display_text, font=self.font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                except:
                    # 回退估算
                    text_width = len(display_text) * self.font_size
                    text_height = self.font_size

                # 繪製白色背景矩形
                padding = 2
                draw.rectangle(
                    [
                        (label_x - padding, label_y - padding),
                        (label_x + text_width + padding, label_y + text_height + padding)
                    ],
                    fill=(255, 255, 255)
                )

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
        description="在渲染圖片上繪製 bounding boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本用法
  python -m lib.utils.draw_bbox_on_render \\
      --render_dir results/render/ \\
      --text_allocate results/text_allocate/text_allocate.json \\
      --output results/render_with_bbox/

  # 自訂 bbox 顏色和粗細
  python -m lib.utils.draw_bbox_on_render \\
      --render_dir results/render/ \\
      --text_allocate results/text_allocate/text_allocate.json \\
      --output results/render_with_bbox/ \\
      --bbox_color 255 0 0 \\
      --thickness 3

  # 不顯示文字標籤
  python -m lib.utils.draw_bbox_on_render \\
      --render_dir results/render/ \\
      --text_allocate results/text_allocate/text_allocate.json \\
      --output results/render_with_bbox/ \\
      --no_text_label
        """
    )

    parser.add_argument(
        '--render_dir',
        type=str,
        required=True,
        help='已渲染圖片的目錄 (例如: results/render/)'
    )

    parser.add_argument(
        '--text_allocate',
        type=str,
        required=True,
        help='Text Allocate JSON 檔案路徑 (例如: results/text_allocate/text_allocate.json)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='輸出目錄 (例如: results/render_with_bbox/)'
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
        help='不顯示文字標籤'
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

    # 初始化 BboxDrawer
    drawer = BboxDrawer(
        bbox_color=tuple(args.bbox_color),
        bbox_thickness=args.thickness,
        text_color=tuple(args.text_color),
        show_text_label=not args.no_text_label,
        font_path=args.font,
        font_size=args.font_size
    )

    # 執行繪製
    drawer.draw_from_files(
        render_dir=args.render_dir,
        text_allocate_json=args.text_allocate,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
