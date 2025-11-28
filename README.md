# 日漫繁化流程

## 完整流程

```bash
# 1. OCR 辨識
python utils/ocr.py --input-dir data/2 --output-dir results/2

# 2. 翻譯 (日→繁中)
python utils/translation_model.py \
  --input_path results/2/ocr.json \
  --output_path results/2/translated.json \
  --model_path model/sakura-1.5b-qwen2.5-v1.0-fp16.gguf

# 3. 渲染翻譯文字到圖片
python utils/image_renderer.py results/2/translated.json -o results/2/rendered
```

## 流程說明

1. **OCR** ([ocr.py](utils/ocr.py)): 使用 PaddleOCR 辨識圖片文字，輸出 bounding boxes + 文字
2. **Translation** ([translation_model.py](utils/translation_model.py)): 使用 SakuraLLM 翻譯日文→繁中
3. **Render** ([image_renderer.py](utils/image_renderer.py)): 將翻譯文字渲染回原圖對應位置