#!/bin/bash
# Test script for PaddleOCR-VL with layout.json input

# Usage example with layout.json
python lib/ocr/paddleocr_vl_manga.py \
  --model-path /tmp2/share_data/PaddleOCR-VL-For-Manga \
  --processor-path /tmp2/share_data/PaddleOCR-VL-For-Manga \
  --task ocr \
  --layout-json results/layout/yolo/layout.json \
  --output-dir results/ocr/paddleocrvl \
  --max-new-tokens 256
