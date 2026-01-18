"""
Text Allocater Module
將翻譯文字分配到 text detection 的 bounding boxes
"""
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Literal
from collections import defaultdict
import re

from lib.schema import (
    BoundingBox,
    TranslationOutputSchema,
    TextAllocaterOutput,
    AllocatedBoundingBox
)


class TextAllocater:
    """
    文字分配器類別
    負責將翻譯結果(segmentation boxes)的文字分配到 text detection boxes
    """

    def __init__(
        self,
        translation_json_path: str,
        text_region_json_path: str,
        output_path: str = "results/text_allocate/text_allocate.json",
        split_strategy: Literal["even", "newline", "region_newline", "smart"] = "even"
    ):
        """
        初始化 TextAllocater

        Args:
            translation_json_path: 翻譯結果 JSON 檔案路徑 (e.g., results/translation/hunyuan.json)
            text_region_json_path: 文字區域檢測結果 JSON 檔案路徑 (e.g., results/text_region/text_region.json)
            output_path: 輸出檔案路徑
            split_strategy: 文字分割策略
                - "even": 均分策略 (預設)
                - "newline": 換行策略
                - "region_newline": Region + 換行策略
                - "smart": 智能分割策略 (優先在標點符號處分割)
        """
        self.translation_json_path = translation_json_path
        self.text_region_json_path = text_region_json_path
        self.output_path = output_path
        self.split_strategy = split_strategy

        self.translation_info: List[TranslationOutputSchema] = []
        self.text_region_info: List[Dict] = []
        self.img2textdet_bboxes: Dict[str, List[BoundingBox]] = {}
        self.grouped_translation_info: Dict[str, List[TranslationOutputSchema]] = {}

    def run(self) -> List[TextAllocaterOutput]:
        """
        執行文字分配流程

        Returns:
            List[TextAllocaterOutput]: 分配結果列表
        """
        # Step 1: Load translation_info and text_region_info
        self._load_data()

        # Step 2: Build img2textdet_bboxes mapping
        self._build_img2textdet_mapping()

        # Step 2.5: Group translation_info by source
        self._group_translation_by_source()

        # Step 3: Allocate text to detection boxes
        text_allocate_res = self._allocate_text()

        # Step 4: Save results
        self._save_results(text_allocate_res)

        return text_allocate_res

    def _load_data(self):
        """載入翻譯資料和文字區域檢測資料"""
        # Load translation info
        with open(self.translation_json_path, 'r', encoding='utf-8') as f:
            self.translation_info = json.load(f)

        # Load text region info
        with open(self.text_region_json_path, 'r', encoding='utf-8') as f:
            self.text_region_info = json.load(f)

        print(f"Loaded {len(self.translation_info)} translation records")
        print(f"Loaded {len(self.text_region_info)} text region records")

    def _build_img2textdet_mapping(self):
        """建立 image_path -> text detection bboxes 的映射"""
        for record in self.text_region_info:
            img_path = record['image_path']
            bboxes = record['bounding_boxes']
            self.img2textdet_bboxes[img_path] = bboxes

        print(f"Built mapping for {len(self.img2textdet_bboxes)} images")

    def _group_translation_by_source(self):
        """將 translation_info 的 bounding boxes 依照相同 source 整理在一起"""
        self.grouped_translation_info = defaultdict(list)

        for record in self.translation_info:
            source = record['source']
            self.grouped_translation_info[source].append(record)

        print(f"Grouped translation info into {len(self.grouped_translation_info)} unique sources")

    def _allocate_text(self) -> List[TextAllocaterOutput]:
        """
        為每個圖片的 text detection boxes 分配翻譯文字

        Returns:
            List[TextAllocaterOutput]: 分配結果列表
        """
        text_allocate_res: List[TextAllocaterOutput] = []

        # Iterate over grouped translation info (by source)
        for source, img_examples in self.grouped_translation_info.items():
            # Get text detection bboxes for this source
            textdet_bboxes = self.img2textdet_bboxes.get(source, [])

            # Collect all segmentation bboxes from all cropped images with same source
            all_segmentation_bboxes = []
            all_texts = []
            all_translated_texts = []

            for img_example in img_examples:
                all_segmentation_bboxes.extend(img_example['bounding_boxes'])
                all_texts.append(img_example['text'])
                if img_example.get('translated_text'):
                    all_translated_texts.append(img_example['translated_text'])

            if not textdet_bboxes:
                print(f"Warning: No text detection boxes found for {source}")
                # Still create output with empty bboxes
                text_allocate_unit: TextAllocaterOutput = {
                    'source': source,
                    'text': '\n'.join(all_texts),
                    'translated_text': '\n'.join(all_translated_texts),
                    'image_path': source,
                    'bounding_boxes': []
                }
                text_allocate_res.append(text_allocate_unit)
                continue

            # Map each textdet_box to nearest seg_box
            seg_box2textdet: Dict[int, List[int]] = defaultdict(list)

            for textdet_idx, textdet_box in enumerate(textdet_bboxes):
                seg_idx = self._find_nearest_box(textdet_box, all_segmentation_bboxes)
                seg_box2textdet[seg_idx].append(textdet_idx)

            # Allocate text for each group
            updated_bboxes: List[AllocatedBoundingBox] = []

            # 處理所有 segmentation boxes，包括沒有對應 textdet_box 的
            for seg_idx, seg_box in enumerate(all_segmentation_bboxes):
                textdet_indices = seg_box2textdet.get(seg_idx, [])

                # 如果這個 seg_box 沒有對應的 textdet_box，直接使用 seg_box 本身
                if not textdet_indices:
                    translated_text = seg_box.get('translated_text', '')
                    allocated_bbox: AllocatedBoundingBox = {
                        'box': seg_box['box'],
                        'text': translated_text,
                        'score': None
                    }
                    updated_bboxes.append(allocated_bbox)
                    print(f"Seg box {seg_idx} has no matching textdet_box, preserving as-is")
                    continue

            for seg_idx, textdet_indices in seg_box2textdet.items():
                seg_box = all_segmentation_bboxes[seg_idx]
                translated_text = seg_box['translated_text']

                # Get the group of text detection boxes
                text_det_group = [textdet_bboxes[i] for i in textdet_indices]

                # Split segmentation box based on detection boxes positions
                split_seg_boxes = self._split_segmentation_box(
                    seg_box['box'],
                    text_det_group
                )

                # Allocate text to split segmentation boxes (not detection boxes)
                allocated_group = self._allocate_text_to_split_regions(
                    translated_text,
                    split_seg_boxes,
                    text_det_group
                )

                updated_bboxes.extend(allocated_group)

            # Create output for this source
            text_allocate_unit: TextAllocaterOutput = {
                'source': source,
                'text': '\n'.join(all_texts),
                'translated_text': '\n'.join(all_translated_texts),
                'image_path': source,
                'bounding_boxes': updated_bboxes
            }

            text_allocate_res.append(text_allocate_unit)

        return text_allocate_res

    def _find_nearest_box(
        self,
        textdet_box: BoundingBox,
        segmentation_bboxes: List[Dict]
    ) -> int:
        """
        找到與 textdet_box 最近的 segmentation box

        Args:
            textdet_box: text detection bounding box
            segmentation_bboxes: segmentation bounding boxes 列表

        Returns:
            int: 最近的 segmentation box 的索引
        """
        textdet_center = self._get_box_center(textdet_box['box'])

        min_distance = float('inf')
        nearest_idx = 0

        for idx, seg_box in enumerate(segmentation_bboxes):
            seg_center = self._get_box_center(seg_box['box'])
            distance = self._calculate_distance(textdet_center, seg_center)

            if distance < min_distance:
                min_distance = distance
                nearest_idx = idx

        return nearest_idx

    def _get_box_center(self, box: List[List[float]]) -> Tuple[float, float]:
        """
        計算 bounding box 的中心點

        Args:
            box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            Tuple[float, float]: (center_x, center_y)
        """
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return (center_x, center_y)

    def _calculate_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """
        計算兩點之間的歐幾里得距離

        Args:
            point1: (x1, y1)
            point2: (x2, y2)

        Returns:
            float: 距離
        """
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2
        )

    def _allocate_text_to_group(
        self,
        translated_text: str,
        text_det_group: List[BoundingBox],
        seg_box: List[List[float]]
    ) -> List[AllocatedBoundingBox]:
        """
        將翻譯文字分配給一組 text detection boxes

        如果只有一個 textdet box,直接分配全部文字
        如果有多個 textdet boxes,根據 box 的位置(垂直或水平)和大小比例分配文字

        Args:
            translated_text: 翻譯後的文字
            text_det_group: text detection boxes 列表
            seg_box: 原始的 segmentation box

        Returns:
            List[AllocatedBoundingBox]: 分配結果列表
        """
        allocated_group: List[AllocatedBoundingBox] = []

        # Case 1: Only one text detection box
        if len(text_det_group) == 1:
            print("Without merge")
            textdet_box = text_det_group[0]
            allocated_bbox: AllocatedBoundingBox = {
                'box': textdet_box['box'],
                'text': translated_text,
                'score': textdet_box.get('score')
            }
            allocated_group.append(allocated_bbox)
            return allocated_group

        print("Start merge")
        # Case 2: Multiple text detection boxes - need to split text
        # Sort boxes by position (top to bottom for vertical, left to right for horizontal)
        sorted_group = self._sort_boxes_by_reading_order(text_det_group)

        # Split text based on strategy
        if self.split_strategy == "even":
            text_parts = self._split_text_even(translated_text, len(sorted_group))
        elif self.split_strategy == "newline":
            text_parts = self._split_text_by_newline(translated_text, len(sorted_group))
        elif self.split_strategy == "region_newline":
            text_parts = self._split_text_by_region_newline(translated_text, sorted_group)
        elif self.split_strategy == "smart":
            text_parts = self._split_text_smart(translated_text, len(sorted_group))
        else:
            text_parts = self._split_text_even(translated_text, len(sorted_group))

        # Assign each part to corresponding box
        for textdet_box, text_part in zip(sorted_group, text_parts):
            allocated_bbox: AllocatedBoundingBox = {
                'box': textdet_box['box'],
                'text': text_part,
                'score': textdet_box.get('score')
            }
            allocated_group.append(allocated_bbox)
            print(allocated_bbox)
        return allocated_group

    def _split_segmentation_box(
        self,
        seg_box: List[List[float]],
        text_det_group: List[BoundingBox]
    ) -> List[List[List[float]]]:
        """
        依據 detection boxes 的水平位置分割 segmentation box

        日本漫畫是直書，所以依據 detection boxes 的 x 座標（由右到左）來分割 segmentation box

        Args:
            seg_box: 原始 segmentation box [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_det_group: detection boxes 列表

        Returns:
            List[List[List[float]]]: 分割後的 segmentation boxes 列表
        """
        if len(text_det_group) <= 1:
            return [seg_box]

        # 排序 detection boxes（由右到左，日本漫畫閱讀順序）
        sorted_det_boxes = self._sort_boxes_by_reading_order(text_det_group)

        # 計算 seg_box 的邊界
        seg_x_coords = [p[0] for p in seg_box]
        seg_y_coords = [p[1] for p in seg_box]
        seg_left = min(seg_x_coords)
        seg_right = max(seg_x_coords)
        seg_top = min(seg_y_coords)
        seg_bottom = max(seg_y_coords)

        # 計算每個 detection box 的 x 中心
        det_centers_x = []
        for det_box in sorted_det_boxes:
            center_x, _ = self._get_box_center(det_box['box'])
            det_centers_x.append(center_x)

        # 計算分割線位置（在相鄰 detection boxes 中間）
        split_points = []
        for i in range(len(det_centers_x) - 1):
            # 分割線在兩個相鄰 detection box 中間
            split_x = (det_centers_x[i] + det_centers_x[i + 1]) / 2
            split_points.append(split_x)

        # 生成分割後的 segmentation boxes
        split_boxes = []

        # 由右到左建立分割後的 boxes
        current_right = seg_right

        for i, split_x in enumerate(split_points):
            # 建立一個分割後的 box
            split_box = [
                [split_x, seg_top],      # 左上
                [current_right, seg_top],  # 右上
                [current_right, seg_bottom],  # 右下
                [split_x, seg_bottom]     # 左下
            ]
            split_boxes.append(split_box)
            current_right = split_x

        # 最後一個 box（最左邊）
        last_box = [
            [seg_left, seg_top],
            [current_right, seg_top],
            [current_right, seg_bottom],
            [seg_left, seg_bottom]
        ]
        split_boxes.append(last_box)

        return split_boxes

    def _allocate_text_to_split_regions(
        self,
        translated_text: str,
        split_seg_boxes: List[List[List[float]]],
        text_det_group: List[BoundingBox]
    ) -> List[AllocatedBoundingBox]:
        """
        將翻譯文字分配給分割後的 segmentation boxes

        使用分割後的 segmentation box 作為 text region，而非原始的 detection box，
        這樣可以獲得更大的文字顯示空間

        Args:
            translated_text: 翻譯後的文字
            split_seg_boxes: 分割後的 segmentation boxes 列表
            text_det_group: 對應的 text detection boxes 列表（用於取得 score）

        Returns:
            List[AllocatedBoundingBox]: 分配結果列表
        """
        allocated_group: List[AllocatedBoundingBox] = []
        num_regions = len(split_seg_boxes)

        # Case 1: Only one region
        if num_regions == 1:
            print("Single region - using full segmentation box")
            allocated_bbox: AllocatedBoundingBox = {
                'box': split_seg_boxes[0],
                'text': translated_text,
                'score': text_det_group[0].get('score') if text_det_group else None
            }
            allocated_group.append(allocated_bbox)
            return allocated_group

        print(f"Multiple regions ({num_regions}) - splitting text")

        # Split text based on strategy
        if self.split_strategy == "even":
            text_parts = self._split_text_even(translated_text, num_regions)
        elif self.split_strategy == "newline":
            text_parts = self._split_text_by_newline(translated_text, num_regions)
        elif self.split_strategy == "smart":
            text_parts = self._split_text_smart(translated_text, num_regions)
        else:
            text_parts = self._split_text_even(translated_text, num_regions)

        # 排序 detection boxes 以取得對應的 score
        sorted_det_boxes = self._sort_boxes_by_reading_order(text_det_group)

        # Assign each text part to corresponding split segmentation box
        for i, (split_box, text_part) in enumerate(zip(split_seg_boxes, text_parts)):
            score = sorted_det_boxes[i].get('score') if i < len(sorted_det_boxes) else None
            allocated_bbox: AllocatedBoundingBox = {
                'box': split_box,
                'text': text_part,
                'score': score
            }
            allocated_group.append(allocated_bbox)
            print(f"Region {i}: {allocated_bbox}")

        return allocated_group

    def _sort_boxes_by_reading_order(
        self,
        boxes: List[BoundingBox]
    ) -> List[BoundingBox]:
        """
        根據閱讀順序排序 boxes
        日本漫畫預設為垂直排列: 文字由上到下,對話框由右到左

        Args:
            boxes: BoundingBox 列表

        Returns:
            List[BoundingBox]: 排序後的列表
        """
        if len(boxes) <= 1:
            return boxes

        # Calculate centers
        centers = [self._get_box_center(box['box']) for box in boxes]

        # For Japanese manga: vertical layout with right-to-left reading order
        # 日本漫畫: 垂直排列,由右到左閱讀
        # Sort by x coordinate in descending order (right to left)
        sorted_boxes = sorted(
            zip(boxes, centers),
            key=lambda x: -x[1][0]  # Negative for right-to-left
        )

        return [box for box, _ in sorted_boxes]

    def _split_text_even(
        self,
        text: str,
        num_boxes: int
    ) -> List[str]:
        """
        策略 1: 均分策略 - 根據字數平均分配

        Args:
            text: 要分割的文字
            num_boxes: box 的數量

        Returns:
            List[str]: 分割後的文字列表
        """
        if num_boxes <= 0:
            return []

        if num_boxes == 1:
            return [text]

        text_length = len(text)
        chars_per_box = text_length // num_boxes
        remainder = text_length % num_boxes

        parts = []
        start = 0

        for i in range(num_boxes):
            # Distribute remainder evenly
            length = chars_per_box + (1 if i < remainder else 0)
            end = start + length
            parts.append(text[start:end])
            start = end

        return parts

    def _split_text_by_newline(
        self,
        text: str,
        num_boxes: int
    ) -> List[str]:
        """
        策略 2: 換行策略 - 根據換行符號分割文字

        如果換行符號數量與 box 數量匹配,則根據換行分割
        否則 fallback 到均分策略

        Args:
            text: 要分割的文字
            num_boxes: box 的數量

        Returns:
            List[str]: 分割後的文字列表
        """
        if num_boxes <= 0:
            return []

        if num_boxes == 1:
            return [text]

        # 根據換行符號分割
        lines = text.split('\n')

        # 如果行數剛好等於 box 數量,直接分配
        if len(lines) == num_boxes:
            return lines

        # 如果行數小於 box 數量,需要進一步分割某些行
        if len(lines) < num_boxes:
            # 找出最長的行進行分割
            result = lines.copy()
            while len(result) < num_boxes:
                # 找出最長的行
                max_idx = max(range(len(result)), key=lambda i: len(result[i]))
                long_line = result[max_idx]

                # 將長行分成兩半
                mid = len(long_line) // 2
                result[max_idx] = long_line[:mid]
                result.insert(max_idx + 1, long_line[mid:])

            return result

        # 如果行數多於 box 數量,需要合併某些行
        if len(lines) > num_boxes:
            # 將多餘的行合併
            lines_per_box = len(lines) / num_boxes
            result = []

            for i in range(num_boxes):
                start_idx = int(i * lines_per_box)
                end_idx = int((i + 1) * lines_per_box)
                merged_line = '\n'.join(lines[start_idx:end_idx])
                result.append(merged_line)

            return result

        return lines

    def _split_text_by_region_newline(
        self,
        text: str,
        sorted_boxes: List[BoundingBox]
    ) -> List[str]:
        """
        策略 3: Region + 換行策略 - 考慮 box 的位置區域和換行符號

        分析 boxes 的空間分布,將它們分群(region)
        然後在每個 region 內根據換行符號分配文字

        Args:
            text: 要分割的文字
            sorted_boxes: 已排序的 bounding boxes

        Returns:
            List[str]: 分割後的文字列表
        """
        num_boxes = len(sorted_boxes)

        if num_boxes <= 0:
            return []

        if num_boxes == 1:
            return [text]

        # 計算每個 box 的中心點
        centers = [self._get_box_center(box['box']) for box in sorted_boxes]

        # 判斷是垂直還是水平排列
        y_coords = [c[1] for c in centers]
        x_coords = [c[0] for c in centers]

        y_variance = max(y_coords) - min(y_coords)
        x_variance = max(x_coords) - min(x_coords)

        is_vertical = y_variance > x_variance

        # 根據位置差異將 boxes 分群成 regions
        regions = self._cluster_boxes_into_regions(centers, is_vertical)

        # 根據換行符號分割文字
        lines = text.split('\n')

        # 為每個 region 分配文字行
        result = []
        line_idx = 0

        for region_indices in regions:
            region_size = len(region_indices)

            # 為這個 region 分配對應數量的文字行
            region_lines = []
            for _ in range(region_size):
                if line_idx < len(lines):
                    region_lines.append(lines[line_idx])
                    line_idx += 1
                else:
                    region_lines.append("")

            # 如果分配的行數不足,使用均分策略分割剩餘的行
            if region_size > len(region_lines):
                # 合併已有的行
                merged_text = '\n'.join(region_lines)
                # 使用均分策略
                region_lines = self._split_text_even(merged_text, region_size)

            result.extend(region_lines)

        # 如果還有剩餘的文字行,分配給最後一個 box
        if line_idx < len(lines):
            remaining = '\n'.join(lines[line_idx:])
            if result:
                result[-1] = result[-1] + '\n' + remaining if result[-1] else remaining

        return result[:num_boxes]  # 確保返回正確數量

    def _cluster_boxes_into_regions(
        self,
        centers: List[Tuple[float, float]],
        is_vertical: bool
    ) -> List[List[int]]:
        """
        將 boxes 根據位置分群成 regions

        使用簡單的距離閾值方法:
        - 垂直排列時,根據 y 座標差異分群
        - 水平排列時,根據 x 座標差異分群

        Args:
            centers: box 中心點列表
            is_vertical: 是否為垂直排列

        Returns:
            List[List[int]]: 每個 region 包含的 box 索引列表
        """
        if len(centers) <= 1:
            return [[0]] if centers else []

        # 選擇主要座標(垂直用 y,水平用 x)
        coords = [c[1] if is_vertical else c[0] for c in centers]

        # 計算相鄰 boxes 之間的距離
        distances = [coords[i+1] - coords[i] for i in range(len(coords) - 1)]

        # 計算平均距離
        avg_distance = sum(distances) / len(distances) if distances else 0

        # 使用 1.5 倍平均距離作為閾值來判斷是否為新 region
        threshold = avg_distance * 1.5

        # 分群
        regions = [[0]]
        for i in range(1, len(centers)):
            if distances[i-1] > threshold:
                # 開始新 region
                regions.append([i])
            else:
                # 加入當前 region
                regions[-1].append(i)

        return regions

    def _split_text_smart(
        self,
        text: str,
        num_boxes: int
    ) -> List[str]:
        """
        策略 4: 智能分割策略 - 優先在標點符號處分割

        嘗試在句子邊界(標點符號)處分割文字,
        使每個 box 包含完整的語意單元

        Args:
            text: 要分割的文字
            num_boxes: box 的數量

        Returns:
            List[str]: 分割後的文字列表
        """
        if num_boxes <= 0:
            return []

        if num_boxes == 1:
            return [text]

        # 定義標點符號優先級(由高到低)
        # 中文和日文標點符號
        punctuation_patterns = [
            r'[。!?!!??]+',  # 句子結束標點
            r'[、,，;；:：]+',  # 逗號、分號等
            r'[\n\r]+',  # 換行
            r'\s+',  # 空白
        ]

        # 嘗試找到合適的分割點
        best_split = None

        for pattern in punctuation_patterns:
            splits = self._try_split_by_pattern(text, num_boxes, pattern)
            if splits and len(splits) == num_boxes:
                best_split = splits
                break

        # 如果找不到合適的分割點,使用均分策略
        if not best_split:
            best_split = self._split_text_even(text, num_boxes)

        return best_split

    def _try_split_by_pattern(
        self,
        text: str,
        num_boxes: int,
        pattern: str
    ) -> List[str]:
        """
        嘗試根據指定的 pattern 分割文字

        Args:
            text: 要分割的文字
            num_boxes: 目標 box 數量
            pattern: 正則表達式 pattern

        Returns:
            List[str]: 分割後的文字列表,如果無法完美分割則返回 None
        """
        # 找到所有匹配的位置
        matches = list(re.finditer(pattern, text))

        if len(matches) < num_boxes - 1:
            return None

        # 如果匹配數量剛好是 num_boxes - 1,直接在這些位置分割
        if len(matches) == num_boxes - 1:
            parts = []
            last_end = 0

            for match in matches:
                # 包含標點符號在前一段
                parts.append(text[last_end:match.end()])
                last_end = match.end()

            # 加入最後一段
            if last_end < len(text):
                parts.append(text[last_end:])

            return parts if len(parts) == num_boxes else None

        # 如果匹配數量多於需要的,選擇最均勻的分割點
        # 計算理想的分割位置
        text_len = len(text)
        ideal_positions = [int(text_len * (i+1) / num_boxes) for i in range(num_boxes - 1)]

        # 為每個理想位置找最近的標點符號
        selected_matches = []
        used_indices = set()

        for ideal_pos in ideal_positions:
            # 找到最接近理想位置的匹配
            best_match = None
            best_distance = float('inf')
            best_idx = -1

            for idx, match in enumerate(matches):
                if idx in used_indices:
                    continue

                distance = abs(match.end() - ideal_pos)
                if distance < best_distance:
                    best_distance = distance
                    best_match = match
                    best_idx = idx

            if best_match:
                selected_matches.append(best_match)
                used_indices.add(best_idx)

        # 根據選中的匹配進行分割
        if len(selected_matches) == num_boxes - 1:
            # 按位置排序
            selected_matches.sort(key=lambda m: m.end())

            parts = []
            last_end = 0

            for match in selected_matches:
                parts.append(text[last_end:match.end()])
                last_end = match.end()

            # 加入最後一段
            if last_end < len(text):
                parts.append(text[last_end:])

            return parts if len(parts) == num_boxes else None

        return None

    # 保留舊方法作為 alias,保持向後兼容
    def _split_text_by_boxes(
        self,
        text: str,
        num_boxes: int
    ) -> List[str]:
        """
        將文字根據 box 數量分割 (向後兼容的方法)
        使用當前設定的策略

        Args:
            text: 要分割的文字
            num_boxes: box 的數量

        Returns:
            List[str]: 分割後的文字列表
        """
        return self._split_text_even(text, num_boxes)

    def _save_results(self, results: List[TextAllocaterOutput]):
        """
        儲存分配結果到 JSON 檔案

        Args:
            results: 分配結果列表
        """
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {self.output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="將翻譯文字分配到 text detection 的 bounding boxes"
    )
    parser.add_argument(
        "--translation-json",
        type=str,
        default="results/translation/2/hunyuan.json",
        help="翻譯結果 JSON 檔案路徑"
    )
    parser.add_argument(
        "--text-region-json",
        type=str,
        default="results/text_detection/2/text_region.json",
        help="文字區域檢測結果 JSON 檔案路徑"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/text_allocate/2/text_allocate.json",
        help="輸出檔案路徑"
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["even", "newline", "region_newline", "smart"],
        default="region_newline",
        help="文字分割策略: even(均分), newline(換行), region_newline(區域+換行), smart(智能)"
    )

    args = parser.parse_args()

    allocater = TextAllocater(
        translation_json_path=args.translation_json,
        text_region_json_path=args.text_region_json,
        output_path=args.output,
        split_strategy=args.split_strategy
    )

    results = allocater.run()
    print(f"Text allocation completed. Processed {len(results)} images.")
