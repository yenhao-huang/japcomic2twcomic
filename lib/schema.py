"""
OCR 輸出資料結構的 Schema 定義
"""
from typing import List, Optional, TypedDict, Dict

'''
OCR 資料結構定義
'''
class BoundingBox(TypedDict):
    """文字區域的邊界框資訊"""
    box: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text: str
    score: Optional[float]


class OcrOutputSchema(TypedDict):
    """OCR 辨識結果的標準格式"""
    source: str  # 圖片來源路徑
    text: str  # 所有辨識出的文字（換行分隔）
    image_path: str  # 標註圖片的路徑 (=source)
    bounding_boxes: List[BoundingBox]  # 每個文字區域的座標、文字和信心分數

class OcrErrorSchema(TypedDict):
    """OCR 處理失敗的錯誤格式"""
    source: str
    error: str

'''
翻譯資料結構定義
'''
class TranslatedBoundingBox(TypedDict):
    """包含翻譯文字的邊界框資訊"""
    box: List[List[float]]  # [左上, 右上, 右下, 左下] = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 
    text: str
    score: Optional[float]
    translated_text: str  # 翻譯後的文字


class TranslationOutputSchema(TypedDict):
    """翻譯結果的標準格式"""
    source: str  # 圖片來源路徑
    text: str  # 原始辨識出的文字（換行分隔）
    translated_text: str  # 翻譯後的完整文字
    image_path: str  # 標註圖片的路徑
    bounding_boxes: List[TranslatedBoundingBox]  # 每個文字區域的座標、原文、翻譯和信心分數


class TranslationErrorSchema(TypedDict):
    """翻譯處理失敗的錯誤格式"""
    source: str
    error: str

'''
Layout 資料結構定義
'''
class RegionSchema(TypedDict):
    box: List[List[float]] # [左上, 右下] = [[x1,y1], [x2,y2]]
    segmentation_mask: List[List[int]] # [H, W] binary mask
    cropped_image_path: str  # Cropped image file path

class LayoutOutputSchema(TypedDict):
    source: str  # 圖片來源路徑
    region_result: List[RegionSchema]
