"""
基於 Text Detection 的漫畫圖片渲染器
處理流程:
1. 根據 text_mask 進行 inpaint (填白)
2. 根據 allocated_text 在對應 bbox 填上文字
"""

import json
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
import argparse


from lib.schema import TextAllocaterOutput, AllocatedBoundingBox, LayoutOutputSchema
from lib.text_mask.yolo import decompress_mask_from_base64_png


class TextDetRender:
    """
    基於 Text Detection 的渲染器

    渲染流程:
    1. Inpaint: 根據 text_mask.json 中的 segmentation_mask 將文字區域填白
    2. Text Render: 根據 allocated_text.json 中的 bbox 和分配的翻譯文字進行渲染
    """

    def __init__(
        self,
        font_path: Optional[str] = None,
        default_font_size: int = 20,
        inpaint_shrink_ratio: float = 0.8,
        default_is_vertical: bool = True,
        min_font_size: int = 12
    ):
        """
        初始化渲染器

        Args:
            font_path: 自訂字型路徑
            default_font_size: 預設字型大小
            inpaint_shrink_ratio: Inpaint 填白範圍的縮小比例 (0.0-1.0)
                - 1.0: 不縮小，使用完整的 mask 範圍
                - 0.9: 縮小 10%，避免填白到邊框外
                - 0.8: 縮小 20%，更保守的填白範圍
            default_is_vertical: 預設文字方向
                - True: 垂直排版（日漫、港漫）預設值
                - False: 水平排版（美漫、韓漫）
            min_font_size: 最低字體大小限制 (預設: 12)
                - 低於此大小的文字框會使用此最低限制
                - 避免文字太小而無法閱讀
        """
        self.font_path = font_path
        self.default_font_size = default_font_size
        self.inpaint_shrink_ratio = max(0.0, min(1.0, inpaint_shrink_ratio))  # 限制在 0.0-1.0
        self.default_is_vertical = default_is_vertical
        self.min_font_size = max(8, min_font_size)  # 最小值至少為 8，避免設置過小
        self.font_cache: Dict[int, ImageFont.ImageFont] = {}
        self.size_test_cache: Dict[tuple, bool] = {}  # 字型大小測試快取

    def _clean_str(self, text: str) -> str:
        """
        清理文字中無用的標點符號，並合併連續符號

        Args:
            text: 原始文字

        Returns:
            清理後的文字
        """
        # 移除連續的省略號，只保留一個
        text = re.sub(r'\.{3,}', '…', text)
        text = re.sub(r'…{2,}', '…', text)

        # 移除無用文字符號
        text = text.replace('「', '').replace('」', '')
        text = text.replace('《', '').replace('》', '')
        text = text.replace('…', '')
        text = text.replace(' ', '')

        # 合併連續符號為單一字元 (用於垂直排版)
        # e.g., !! -> ‼, ?? -> ⁇, !? -> ⁉, ?! -> ⁈
        text = re.sub(r'[!！]{2,}', '‼', text)
        text = re.sub(r'[?？]{2,}', '⁇', text)
        text = re.sub(r'[!！][?？]', '⁉', text)
        text = re.sub(r'[?？][!！]', '⁈', text)

        # 合併連續破折號為單一破折號
        # e.g., —— -> —, ── -> ─, -- -> -
        text = re.sub(r'[—―]{2,}', '—', text)  # Em dash / Horizontal bar
        text = re.sub(r'[-–]{2,}', '—', text)  # Hyphen / En dash -> Em dash
        text = re.sub(r'[─]{2,}', '─', text)   # Box drawing horizontal

        # 移除開頭和結尾的無意義標點
        text = text.strip('.,。，、')

        return text

    def render_from_files(
        self,
        text_mask_json: str,
        allocated_text_json: str,
        output_dir: str
    ):
        """
        從 JSON 檔案讀取資料並進行渲染

        Args:
            text_mask_json: text_mask.json 路徑 (例如: results/layout/yolo/layout.json)
            allocated_text_json: text_allocate.json 路徑 (例如: results/text_allocate/text_allocate.json)
            output_dir: 輸出目錄
        """
        # 1. 載入 text_mask.json (layout.json)
        text_mask_data = self._load_text_mask(text_mask_json)

        # 2. 載入 allocated_text.json (已按原始圖片分組)
        allocated_text_data = self._load_allocated_text(allocated_text_json)

        # 3. 建立映射: source_image -> segmentation_mask (合併所有 regions)
        source_to_mask = self._build_source_to_mask_mapping(text_mask_data)

        # 4. 批次渲染
        self._render_batch(allocated_text_data, source_to_mask, output_dir)

    def _load_text_mask(self, text_mask_json: str) -> List[LayoutOutputSchema]:
        """載入並解碼 text_mask.json"""
        text_mask_path = Path(text_mask_json)
        if not text_mask_path.exists():
            raise FileNotFoundError(f"找不到 text_mask.json: {text_mask_path}")

        print(f"載入 Text Mask JSON: {text_mask_path}")
        with open(text_mask_path, 'r', encoding='utf-8') as f:
            data: List[LayoutOutputSchema] = json.load(f)

        # 解碼所有 segmentation_mask (從 Base64 PNG 轉為 numpy array)
        print("解碼 segmentation_mask...")
        for item in data:
            for region in item.get('region_result', []):
                mask_base64 = region['segmentation_mask']
                region['segmentation_mask'] = decompress_mask_from_base64_png(mask_base64)

        print(f"成功載入 {len(data)} 張圖片的 text_mask 資料")
        return data

    def _load_allocated_text(self, allocated_text_json: str) -> List[TextAllocaterOutput]:
        """載入 allocated_text.json"""
        allocated_text_path = Path(allocated_text_json)
        if not allocated_text_path.exists():
            raise FileNotFoundError(f"找不到 allocated_text.json: {allocated_text_path}")

        print(f"載入 Allocated Text JSON: {allocated_text_path}")
        with open(allocated_text_path, 'r', encoding='utf-8') as f:
            data: List[TextAllocaterOutput] = json.load(f)

        print(f"成功載入 {len(data)} 張圖片的 allocated_text 資料")
        return data

    def _build_source_to_mask_mapping(
        self,
        text_mask_data: List[LayoutOutputSchema]
    ) -> Dict[str, np.ndarray]:
        """
        建立 source_image 到 合併後的 segmentation_mask 的映射
        將同一張圖片的所有 region masks 合併為一個 mask

        Returns:
            Dict[source_image, merged_segmentation_mask]
        """
        source_to_mask: Dict[str, np.ndarray] = {}

        for item in text_mask_data:
            source_image = item['source']
            regions = item.get('region_result', [])

            if not regions:
                continue

            # 取得第一個 region 的 mask 來確定尺寸
            first_mask = regions[0]['segmentation_mask']
            merged_mask = np.zeros_like(first_mask, dtype=np.uint8)

            # 合併所有 regions 的 masks
            for region in regions:
                seg_mask = region['segmentation_mask']
                # 使用 bitwise OR 合併 masks
                merged_mask = cv2.bitwise_or(merged_mask, seg_mask.astype(np.uint8))

            source_to_mask[source_image] = merged_mask

        print(f"建立了 {len(source_to_mask)} 張圖片的 mask 映射")
        return source_to_mask

    def _render_batch(
        self,
        allocated_text_data: List[TextAllocaterOutput],
        source_to_mask: Dict[str, np.ndarray],
        output_dir: str
    ):
        """批次渲染所有圖片"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total = len(allocated_text_data)
        print(f"\n開始批次渲染 {total} 張圖片")

        for idx, item in enumerate(allocated_text_data, 1):
            source_image = item['source']
            bboxes = item.get('bounding_boxes', [])

            print(f"\n[{idx}/{total}] 處理: {source_image}")

            # 取得對應的 mask
            if source_image not in source_to_mask:
                print(f"  警告: 找不到對應的 mask，跳過此圖片")
                continue

            seg_mask = source_to_mask[source_image]

            # 渲染單張圖片
            self._render_single_image(source_image, seg_mask, bboxes, output_dir)

        print(f"\n全部完成！輸出已儲存至: {output_dir}")

    def _render_single_image(
        self,
        source_image: str,
        seg_mask: np.ndarray,
        bboxes: List[AllocatedBoundingBox],
        output_dir: str
    ):
        """
        渲染單張圖片

        Args:
            source_image: 原始圖片路徑
            seg_mask: 合併後的 segmentation mask
            bboxes: 文字框列表
            output_dir: 輸出目錄
        """
        # 1. 載入原始圖片
        source_path = Path(source_image)
        if not source_path.exists():
            print(f"  警告: 找不到來源圖片: {source_path}")
            return

        img = Image.open(source_path)
        # 確保圖片為 RGB 模式，避免灰階圖片在繪製文字時出現顏色格式錯誤
        if img.mode != 'RGB':
            img = img.convert('RGB')
        print(f"  圖片大小: {img.size}")
        print(f"  文字框數量: {len(bboxes)}")

        img = self._apply_inpaint(img, seg_mask)
        # 2b. Inpaint: 根據 segmentation_mask 填白 (只處理與 bbox 重疊的區域)
        #img = self._apply_inpaint_with_bbox(img, seg_mask, bboxes)
        print(f"  完成 Inpaint (填白)")

        # 3. Text Render: 在 bbox 填上文字
        img = self._render_text_in_bboxes(img, bboxes)
        print(f"  完成文字渲染")

        # 4. 儲存結果
        output_filename = source_path.stem + "_rendered.jpg"
        output_filepath = Path(output_dir) / output_filename
        img.save(str(output_filepath), quality=95)
        print(f"  已儲存至: {output_filepath}")

    def _apply_inpaint(self, img: Image.Image, seg_mask: np.ndarray) -> Image.Image:
        """
        根據 segmentation mask 將區域填白 (Inpaint)

        Args:
            img: PIL Image
            seg_mask: segmentation mask (numpy array)

        Returns:
            填白後的 PIL Image
        """
        img_array = np.array(img)

        # 確保 mask 大小與圖片一致
        if seg_mask.shape[0] != img_array.shape[0] or seg_mask.shape[1] != img_array.shape[1]:
            mask = cv2.resize(
                seg_mask,
                (img_array.shape[1], img_array.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask = seg_mask.copy()

        # 如果設定了縮小比例，縮小 mask 範圍
        if self.inpaint_shrink_ratio < 1.0:
            mask = self._shrink_mask(mask, self.inpaint_shrink_ratio)

        # 填白 (將 mask > 0 的區域設為白色)
        # 處理不同通道數的圖片
        if len(img_array.shape) == 2:
            # 灰階圖片
            img_array[mask > 0] = 255
        elif img_array.shape[2] == 1:
            # 單通道圖片
            img_array[mask > 0] = [255]
        elif img_array.shape[2] == 3:
            # RGB 圖片
            img_array[mask > 0] = [255, 255, 255]
        elif img_array.shape[2] == 4:
            # RGBA 圖片
            img_array[mask > 0] = [255, 255, 255, 255]

        return Image.fromarray(img_array)

    def _apply_inpaint_by_bbox(self, img: Image.Image, bboxes: List[AllocatedBoundingBox]) -> Image.Image:
        """
        直接根據 bounding box 將區域填白 (Inpaint)
        不使用 segmentation mask，直接填白整個 bbox 區域

        Args:
            img: PIL Image
            bboxes: 文字框列表

        Returns:
            填白後的 PIL Image
        """
        img_array = np.array(img)

        # 如果沒有 bbox，直接返回原圖
        if len(bboxes) == 0:
            return img

        for bbox in bboxes:
            # box 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (左上, 右上, 右下, 左下)
            box = bbox['box']
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(img_array.shape[1], int(max(xs)))
            y2 = min(img_array.shape[0], int(max(ys)))

            # 如果設定了縮小比例，縮小填白範圍
            if self.inpaint_shrink_ratio < 1.0:
                width = x2 - x1
                height = y2 - y1
                shrink_x = int(width * (1 - self.inpaint_shrink_ratio) / 2)
                shrink_y = int(height * (1 - self.inpaint_shrink_ratio) / 2)
                x1 += shrink_x
                y1 += shrink_y
                x2 -= shrink_x
                y2 -= shrink_y

            # 填白該 bbox 區域 (處理不同通道數的圖片)
            if len(img_array.shape) == 2:
                img_array[y1:y2, x1:x2] = 255
            elif img_array.shape[2] == 1:
                img_array[y1:y2, x1:x2] = [255]
            elif img_array.shape[2] == 3:
                img_array[y1:y2, x1:x2] = [255, 255, 255]
            elif img_array.shape[2] == 4:
                img_array[y1:y2, x1:x2] = [255, 255, 255, 255]

        return Image.fromarray(img_array)

    def _apply_inpaint_with_bbox(self, img: Image.Image, seg_mask: np.ndarray, bboxes: List[AllocatedBoundingBox]) -> Image.Image:
        """
        根據 segmentation mask 將區域填白 (Inpaint)
        只處理與 bounding box 有重疊的 mask 區域，沒有重疊的保留原圖

        Args:
            img: PIL Image
            seg_mask: segmentation mask (numpy array)
            bboxes: 文字框列表

        Returns:
            填白後的 PIL Image
        """
        img_array = np.array(img)

        # 如果沒有 bbox，直接返回原圖
        if len(bboxes) == 0:
            return img

        # 確保 mask 大小與圖片一致
        if seg_mask.shape[0] != img_array.shape[0] or seg_mask.shape[1] != img_array.shape[1]:
            mask = cv2.resize(
                seg_mask,
                (img_array.shape[1], img_array.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask = seg_mask.copy()

        # 如果設定了縮小比例，縮小 mask 範圍
        if self.inpaint_shrink_ratio < 1.0:
            mask = self._shrink_mask(mask, self.inpaint_shrink_ratio)

        # 建立 bbox 區域的 mask
        bbox_mask = np.zeros(mask.shape, dtype=np.uint8)
        for bbox in bboxes:
            # box 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (左上, 右上, 右下, 左下)
            box = bbox['box']
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(img_array.shape[1], int(max(xs)))
            y2 = min(img_array.shape[0], int(max(ys)))
            bbox_mask[y1:y2, x1:x2] = 1

        # 只保留與 bbox 重疊的 mask 區域
        final_mask = (mask > 0) & (bbox_mask > 0)

        # 填白 (將 final_mask 為 True 的區域設為白色，處理不同通道數)
        if len(img_array.shape) == 2:
            img_array[final_mask] = 255
        elif img_array.shape[2] == 1:
            img_array[final_mask] = [255]
        elif img_array.shape[2] == 3:
            img_array[final_mask] = [255, 255, 255]
        elif img_array.shape[2] == 4:
            img_array[final_mask] = [255, 255, 255, 255]

        return Image.fromarray(img_array)

    def _shrink_mask(self, mask: np.ndarray, shrink_ratio: float) -> np.ndarray:
        """
        縮小 mask 的範圍，避免填白超出文字框

        Args:
            mask: 原始 mask (numpy array)
            shrink_ratio: 縮小比例 (0.0-1.0)
                - 1.0: 不縮小
                - 0.9: 縮小 10%
                - 0.8: 縮小 20%

        Returns:
            縮小後的 mask
        """
        if shrink_ratio >= 1.0:
            return mask

        # 確保 mask 是二值化的
        mask_binary = (mask > 0).astype(np.uint8) * 255

        # 計算腐蝕的 kernel 大小
        # 根據 mask 的大小和縮小比例動態計算
        mask_area = np.sum(mask_binary > 0)
        if mask_area == 0:
            return mask

        # 計算需要縮小的像素數
        shrink_amount = 1.0 - shrink_ratio
        # kernel_size 基於 mask 的平均半徑和縮小比例
        avg_radius = np.sqrt(mask_area / np.pi)
        kernel_size = max(1, int(avg_radius * shrink_amount * 0.2))

        # 使用腐蝕來縮小 mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        shrunk_mask = cv2.erode(mask_binary, kernel, iterations=1)

        return shrunk_mask

    def _render_text_in_bboxes(
        self,
        img: Image.Image,
        bboxes: List[AllocatedBoundingBox]
    ) -> Image.Image:
        """
        在所有 bbox 中渲染文字

        Args:
            img: PIL Image
            bboxes: List[AllocatedBoundingBox]

        Returns:
            渲染後的 PIL Image
        """
        draw = ImageDraw.Draw(img)

        for bbox in bboxes:
            text = bbox.get('text', '').strip()
            if not text:
                continue

            box = bbox['box']  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

            # 計算 bounding box 的邊界
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            box_width = x_max - x_min
            box_height = y_max - y_min

            # 判斷文字方向 (垂直 or 水平)
            is_vertical = self.default_is_vertical

            # 將 text 中無用標點符號處理，e.g.,...
            text = self._clean_str(text)
            
            # 計算字型大小
            font_size = self._calculate_font_size(text, box_width, box_height, is_vertical)
            font = self._get_font(font_size)

            # 渲染文字
            if is_vertical:
                self._draw_vertical_text(draw, text, font, x_min, y_min, x_max, y_max)
            else:
                self._draw_horizontal_text(draw, text, font, x_min, y_min, x_max, y_max)

        return img

    # 需要在垂直排版時旋轉的字元（破折號、省略號等橫向標點）
    VERTICAL_ROTATE_CHARS = {
        '—',  # Em dash (U+2014)
        '─',  # Box drawing horizontal (U+2500)
        '―',  # Horizontal bar (U+2015)
        '-',  # Hyphen-minus
        '–',  # En dash (U+2013)
        '～',  # Fullwidth tilde
        '~',  # Tilde
        '⋯',  # Midline horizontal ellipsis
    }

    def _draw_vertical_text(
        self,
        draw: ImageDraw.Draw,
        text: str,
        font: ImageFont.ImageFont,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float
    ):
        """垂直繪製文字 (從右到左，從上到下，水平置中)"""
        box_width = x_max - x_min
        box_height = y_max - y_min
        padding = int(box_width * 0.15)

        # 計算每列可容納的字元數
        char_height = font.size + 4
        chars_per_column = max(1, int((box_height - 2 * padding) / char_height))

        # 將文字分成多列
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

        # 計算實際需要的總寬度
        column_width = font.size + 8
        total_text_width = len(lines) * column_width - 8  # 最後一列不需要間距

        # 計算起始 x 位置，使文字水平置中
        # 從右到左繪製，所以起始位置是中心偏右
        center_x = (x_min + x_max) / 2
        start_x = center_x + (total_text_width / 2) - font.size

        current_x = start_x
        center_y = (y_min + y_max) / 2

        # 用最長列計算統一的起始 y 位置，確保每列第一個字對齊
        max_line_chars = max(len(line) for line in lines) if lines else 0
        max_text_height = max_line_chars * char_height
        start_y = center_y - (max_text_height / 2)

        for line in lines:
            if current_x < x_min:
                break

            line_y = start_y

            for char in line:
                if line_y + char_height > y_max:
                    break

                # 檢查是否需要旋轉（破折號等橫向標點）
                if char in self.VERTICAL_ROTATE_CHARS:
                    self._draw_rotated_char(
                        draw, char, font, current_x, line_y, char_height
                    )
                else:
                    # 繪製文字 (黑色文字 + 白色描邊)
                    draw.text(
                        (current_x, line_y),
                        char,
                        font=font,
                        fill=(0, 0, 0),
                        stroke_width=2,
                        stroke_fill=(255, 255, 255)
                    )
                line_y += char_height

            current_x -= column_width

    def _draw_rotated_char(
        self,
        draw: ImageDraw.Draw,
        char: str,
        font: ImageFont.ImageFont,
        x: float,
        y: float,
        char_height: float
    ):
        """
        繪製需要旋轉 90 度的字元（用於垂直排版中的破折號等）

        Args:
            draw: ImageDraw 物件
            char: 要繪製的字元
            font: 字型
            x: 繪製位置 x
            y: 繪製位置 y
            char_height: 字元高度（用於計算置中）
        """
        # 建立一個小的透明圖片來繪製單個字元
        char_size = font.size + 4
        char_img = Image.new('RGBA', (char_size * 2, char_size * 2), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)

        # 在小圖片中央繪製字元
        char_draw.text(
            (char_size // 2, char_size // 2),
            char,
            font=font,
            fill=(0, 0, 0, 255),
            stroke_width=2,
            stroke_fill=(255, 255, 255, 255)
        )

        # 旋轉 90 度（順時針）
        rotated = char_img.rotate(-90, expand=True, resample=Image.BICUBIC)

        # 計算繪製位置（置中）
        paste_x = int(x - (rotated.width - font.size) // 2)
        paste_y = int(y + (char_height - rotated.height) // 2)

        # 取得原始圖片並貼上旋轉後的字元
        # 由於 draw 是從 Image 建立的，我們需要取得原始 Image
        original_img = draw._image
        original_img.paste(rotated, (paste_x, paste_y), rotated)

    def _draw_horizontal_text(
        self,
        draw: ImageDraw.Draw,
        text: str,
        font: ImageFont.ImageFont,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float
    ):
        """水平繪製文字 (從左到右，從上到下)"""
        box_width = x_max - x_min
        box_height = y_max - y_min
        padding = int(box_height * 0.1)

        # 將文字按換行符分割
        lines = text.split('\n')

        line_height = font.size + 4
        current_y = y_min + padding

        for line in lines:
            if current_y + line_height > y_max:
                break

            # 計算文字寬度以置中對齊
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(line) * font.size * 0.6

            # 置中對齊
            text_x = x_min + (box_width - text_width) / 2

            # 繪製文字 (黑色文字 + 白色描邊)
            draw.text(
                (text_x, current_y),
                line,
                font=font,
                fill=(0, 0, 0),
                stroke_width=2,
                stroke_fill=(255, 255, 255)
            )
            current_y += line_height

    def _calculate_font_size(
        self,
        text: str,
        box_width: float,
        box_height: float,
        is_vertical: bool,
        padding_ratio: float = 0.15,
        stroke_width: int = 2,
        line_spacing: float = 1.2,
        max_iterations: int = 15
    ) -> int:
        """
        智能計算字型大小算法

        核心策略:
        1. 使用二分搜尋找到最佳字型大小
        2. 實際測量文字渲染後的尺寸
        3. 考慮 padding、描邊、行距等因素
        4. 確保文字完全填充但不溢出 bbox

        Args:
            text: 要渲染的文字
            box_width: bbox 寬度
            box_height: bbox 高度
            is_vertical: 是否為垂直文字
            padding_ratio: 邊距比例
            stroke_width: 描邊寬度
            line_spacing: 行間距倍數
            max_iterations: 最大迭代次數

        Returns:
            最佳字型大小
        """
        # 1. 計算可用空間（扣除 padding）
        if is_vertical:
            padding = int(box_width * padding_ratio)
            available_width = box_width - 2 * padding
            available_height = box_height - 2 * padding
        else:
            padding = int(box_height * 0.1)
            available_width = box_width - 2 * padding
            available_height = box_height - 2 * padding

        # 2. 設定搜尋範圍
        min_size = self.min_font_size
        max_size = min(int(available_width * 1.5), 60)  # 合理上限
        best_size = min_size

        # 3. 二分搜尋最佳字型大小
        while min_size <= max_size and max_iterations > 0:
            mid_size = (min_size + max_size) // 2
            max_iterations -= 1

            # 測試此字型大小是否合適
            fits = self._test_font_size_fits(
                text=text,
                font_size=mid_size,
                available_width=available_width,
                available_height=available_height,
                is_vertical=is_vertical,
                stroke_width=stroke_width,
                line_spacing=line_spacing
            )

            if fits:
                # 字型大小合適，嘗試更大的
                best_size = mid_size
                min_size = mid_size + 1
            else:
                # 字型太大，縮小範圍
                max_size = mid_size - 1

        # 確保返回的字型大小不低於最低限制
        return max(best_size, self.min_font_size)

    def _test_font_size_fits(
        self,
        text: str,
        font_size: int,
        available_width: float,
        available_height: float,
        is_vertical: bool,
        stroke_width: int,
        line_spacing: float
    ) -> bool:
        """
        測試指定字型大小是否能完全填充在可用空間內

        Args:
            text: 要測試的文字
            font_size: 字型大小
            available_width: 可用寬度
            available_height: 可用高度
            is_vertical: 是否為垂直文字
            stroke_width: 描邊寬度
            line_spacing: 行間距倍數

        Returns:
            True: 能填充進去
            False: 太大了，填不進去
        """
        # 使用快取避免重複測量
        cache_key = (text, font_size, available_width, available_height, is_vertical)
        if cache_key in self.size_test_cache:
            return self.size_test_cache[cache_key]

        font = self._get_font(font_size)

        if is_vertical:
            result = self._test_vertical_fit(
                text, font, available_width, available_height,
                stroke_width, line_spacing
            )
        else:
            result = self._test_horizontal_fit(
                text, font, available_width, available_height,
                stroke_width, line_spacing
            )

        self.size_test_cache[cache_key] = result
        return result

    def _test_vertical_fit(
        self,
        text: str,
        font: ImageFont.ImageFont,
        available_width: float,
        available_height: float,
        stroke_width: int,
        line_spacing: float
    ) -> bool:
        """測試垂直排版是否合適"""
        # 計算單個字元的實際高度（包含描邊）
        test_char = text[0] if text else '測'
        dummy_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        try:
            bbox = draw.textbbox((0, 0), test_char, font=font, stroke_width=stroke_width)
            char_height = (bbox[3] - bbox[1]) * line_spacing
            char_width = bbox[2] - bbox[0]
        except:
            # 回退估算
            char_height = font.size * line_spacing
            char_width = font.size

        # 計算每列可容納的字數
        chars_per_column = int(available_height / char_height)
        if chars_per_column < 1:
            return False

        # 計算需要的列數
        num_columns = (len(text) + chars_per_column - 1) // chars_per_column

        # 計算列間距（稍微留一點空間）
        column_spacing = char_width * 0.3
        total_width_needed = num_columns * char_width + (num_columns - 1) * column_spacing

        # 檢查是否能填充進去
        return total_width_needed <= available_width

    def _test_horizontal_fit(
        self,
        text: str,
        font: ImageFont.ImageFont,
        available_width: float,
        available_height: float,
        stroke_width: int,
        line_spacing: float
    ) -> bool:
        """測試水平排版是否合適"""
        dummy_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        # 處理換行符
        lines = text.split('\n') if '\n' in text else [text]

        # 如果沒有換行符，嘗試智能分行
        if len(lines) == 1 and len(text) > 10:
            lines = self._smart_line_break(text, font, available_width, draw, stroke_width)

        # 計算每行的高度
        try:
            bbox = draw.textbbox((0, 0), '測試', font=font, stroke_width=stroke_width)
            line_height = (bbox[3] - bbox[1]) * line_spacing
        except:
            line_height = font.size * line_spacing

        # 檢查總高度
        total_height_needed = len(lines) * line_height
        if total_height_needed > available_height:
            return False

        # 檢查每行寬度
        for line in lines:
            if not line.strip():
                continue
            try:
                bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
                line_width = bbox[2] - bbox[0]
                if line_width > available_width:
                    return False
            except:
                # 回退估算
                line_width = len(line) * font.size * 0.6
                if line_width > available_width:
                    return False

        return True

    def _smart_line_break(
        self,
        text: str,
        font: ImageFont.ImageFont,
        available_width: float,
        draw: ImageDraw.Draw,
        stroke_width: int,
        max_lines: int = 5
    ) -> List[str]:
        """
        智能分行：根據可用寬度自動分行

        策略:
        1. 實際測量每行文字寬度
        2. 在不超過可用寬度的前提下儘量多放文字
        3. 優先在標點符號處分行
        4. 避免單字成行

        Args:
            text: 要分行的文字
            font: 字型
            available_width: 可用寬度
            draw: ImageDraw 物件
            stroke_width: 描邊寬度
            max_lines: 最大行數

        Returns:
            分行後的文字列表
        """
        if len(text) <= 3:
            return [text]

        lines = []
        current_line = ""

        # 標點符號列表（優先在這些位置斷行）
        punctuation = '，。！？；：、,;:!?'

        for i, char in enumerate(text):
            test_line = current_line + char

            # 測量加上這個字元後的寬度
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font, stroke_width=stroke_width)
                line_width = bbox[2] - bbox[0]
            except:
                # 回退估算
                line_width = len(test_line) * font.size * 0.6

            # 如果超過可用寬度，需要斷行
            if line_width > available_width and current_line:
                lines.append(current_line)
                current_line = char

                # 如果已經達到最大行數，剩餘文字放到最後一行
                if len(lines) >= max_lines - 1:
                    current_line += text[i+1:]
                    break
            else:
                current_line = test_line

                # 如果遇到標點符號，且下一個字元會超出寬度，則在此斷行
                if char in punctuation and i < len(text) - 1:
                    next_test = current_line + text[i+1]
                    try:
                        bbox = draw.textbbox((0, 0), next_test, font=font, stroke_width=stroke_width)
                        next_width = bbox[2] - bbox[0]
                    except:
                        next_width = len(next_test) * font.size * 0.6

                    if next_width > available_width * 0.8:  # 接近限制時斷行
                        lines.append(current_line)
                        current_line = ""

                        if len(lines) >= max_lines - 1:
                            current_line = text[i+1:]
                            break

        # 添加最後一行
        if current_line:
            lines.append(current_line)

        return lines if lines else [text]

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """取得字型 (帶快取)"""
        if size in self.font_cache:
            return self.font_cache[size]

        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, size)
            else:
                # 預設使用專案字型
                default_font_path = Path(__file__).parent.parent / "utils" / "fonts" / "Noto_Sans_CJK_Regular.otf"

                if default_font_path.exists():
                    font = ImageFont.truetype(str(default_font_path), size)
                else:
                    # 回退到系統字型
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


def main():
    """主要 CLI 進入點"""
    parser = argparse.ArgumentParser(
        description="基於 Text Detection 的漫畫圖片渲染器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本用法（垂直排版，日漫）
  python -m lib.render.text_det_render \\
      --text_mask results/layout/yolo/layout.json \\
      --allocated_text results/text_allocate/text_allocate.json \\
      --output results/render/

  # 使用自訂字型和縮小 inpaint 範圍（推薦）
  python -m lib.render.text_det_render \\
      --text_mask results/layout/yolo/layout.json \\
      --allocated_text results/text_allocate/text_allocate.json \\
      --output results/render/ \\
      --font /path/to/font.ttf \\
      --inpaint_shrink_ratio 0.9

  # 水平排版（美漫、韓漫）
  python -m lib.render.text_det_render \\
      --text_mask results/layout/yolo/layout.json \\
      --allocated_text results/text_allocate/text_allocate.json \\
      --output results/render/ \\
      --horizontal

  # 更保守的 inpaint 範圍（縮小 15%）+ 水平排版
  python -m lib.render.text_det_render \\
      --text_mask results/layout/yolo/layout.json \\
      --allocated_text results/text_allocate/text_allocate.json \\
      --output results/render/ \\
      --inpaint_shrink_ratio 0.85 \\
      --horizontal

  # 設置最低字體大小為 16，避免小文字不易閱讀
  python -m lib.render.text_det_render \\
      --text_mask results/layout/yolo/layout.json \\
      --allocated_text results/text_allocate/text_allocate.json \\
      --output results/render/ \\
      --min_font_size 16
        """
    )

    parser.add_argument(
        '--text_mask',
        type=str,
        required=True,
        help='Text Mask JSON 檔案路徑 (例如: results/layout/yolo/text_mask.json)'
    )

    parser.add_argument(
        '--allocated_text',
        type=str,
        required=True,
        help='Allocated Text JSON 檔案路徑 (例如: results/text_allocate/text_allocate.json)'
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

    parser.add_argument(
        '--inpaint_shrink_ratio',
        type=float,
        default=0.8,
        help='Inpaint 填白範圍的縮小比例 (0.0-1.0)。1.0=不縮小，0.9=縮小10%%，0.8=縮小20%%（推薦：0.85-0.95）'
    )

    parser.add_argument(
        '--vertical',
        action='store_true',
        default=True,
        help='使用垂直排版（日漫、港漫）。預設為 True'
    )

    parser.add_argument(
        '--horizontal',
        action='store_true',
        help='使用水平排版（美漫、韓漫）。會覆蓋 --vertical 設定'
    )

    parser.add_argument(
        '--min_font_size',
        type=int,
        default=15,
        help='最低字體大小限制（預設：20）。低於此大小的文字框會使用此最低限制，避免文字太小。超出邊界也會使用此字體大小，優先保證可讀性'
    )

    args = parser.parse_args()

    # 判斷文字方向
    is_vertical = not args.horizontal if args.horizontal else args.vertical

    # 初始化渲染器
    renderer = TextDetRender(
        font_path=args.font,
        inpaint_shrink_ratio=args.inpaint_shrink_ratio,
        default_is_vertical=is_vertical,
        min_font_size=args.min_font_size
    )

    # 執行渲染
    renderer.render_from_files(
        text_mask_json=args.text_mask,
        allocated_text_json=args.allocated_text,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
