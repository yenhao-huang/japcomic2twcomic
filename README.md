# 日漫繁化流程

source ../venv_manager/ocr/bin/activate

## II. Pipeline2: pixel-level inpaint

1. get segmentation mask and cropping
layout/yolo.py

input data/2
output results/layout/yolo/layout.json, results/layout/yolo/cropped_dir
```bash
python -m lib.text_mask.yolo --input_dir data/2 --output_dir results/text_mask/2
```
2. detect bbox
```bash
python -m lib.text_detection.ppocrv5 --text-det-model-dir PaddleOCR/output/inference/PP-OCRv5_comic_det_infer/ --input-dir data/2/ --output-dir results/text_detection/2/ 
```
3. recognition text by OCR engine 

```bash
python -m lib.ocr.paddleocr_vl_manga   --layout-json results/text_mask/2/text_mask.json --output-dir results/ocr/2
```

4. translation
```bash
python -m lib.translation.hunyuan --input_path results/ocr/2/ocr.json --output_path results/translation/2/hunyuan.json
```

5. text region refinement 
從 segmentation 取得 bbox 太過粗糙，更精細化

```bash
python -m lib.text_allocater.text_allocater --translation-json results/translation/2/hunyuan.json --text-region-json results/text_detection/2/text_region.json --output results/text_allocate/2/text_allocate.json --split-strategy region_newline
```

6. render
畫圖、inpaint

```bash
python -m lib.render.text_det_render     --text_mask results/text_mask/2/text_mask.json     --allocated_text results/text_allocate/2/text_allocate.json  --output results/render/2/
```

7. 可視化 (Optional)

python -m lib.utils.draw_bbox_on_render \
    --render_dir results/render/ \
    --text_allocate results/text_allocate/text_allocate.json \
    --output results/render/render_with_bbox/

python -m lib.utils.draw_translated_text_bbox_on_image \
    --translation results/translation/hunyuan.json \
    --output results/translation/translation_bbox/

## I. Pipeline1: region-level inpaint


### 0. Format converter
```bash
python lib/utils/jpg_converter.py --input data/1 --output data/1/jpg/
```
### 1. Layout + OCR 辨識
```bash
python lib/ocr/paddleocrv5.py --input-dir data/2 --output-dir results/2

### (recommened)
python lib/ocr/paddleocr_mix.py --input-dir data/2 --output-dir results/2 --text-det-model-dir PaddleOCR/output/inference/PP-OCRv5_comic_det_infer/

python lib/ocr/paddleocr_vl_manga.py   --input-dir data/2   --output-dir results/2_test   --task ocr

python lib/layout/docling_parse.py --data-dir data/2 --output-dir results/2/layout --save-vis
```

Use Finetune Module

```bash
python lib/ocr/paddleocrv5.py --input-dir data/2/ --output-dir results/2_test --text-det-model-dir PaddleOCR/output/inference/PP-OCRv5_comic_det_infer/ --text-rec-model-dir PaddleOCR/output/inference/PP-OCRv5_comic_rec_infer
```


### 2. 翻譯 (日→繁中)
```bash
python lib/translation/sakura.py \
  --input_path results/2/ocr.json \
  --output_path results/2/translated.json \
  --model_path model/sakura-1.5b-qwen2.5-v1.0-fp16.gguf

python lib/translation/google_api.py --input_path results/2/ocr.json --output_path results/2/translated.json

python -m lib.translation.mistral_translation --data-dir results/2/ocr.json


python -m lib.translation.hunyuan --model_path /tmp2/share_data/HY-MT1.5-1.8B
python -m lib.translation.hunyuan --model_path /tmp2/share_data/HY-MT1.5-7B
```
### 3. 渲染翻譯文字到圖片
```bash
python lib/render/image_renderer.py --json_path results/2/translated.json -o results/2/rendered/
```

## Finetune PaddleOCR Engine

Finetune detecction module
```bash
cd PaddleOCR

# 1. Prepare Dataset
python ui/dataset_annotator.py
python core/split_dataset.py --input data/comic_benchmark/det/annotations.txt --output data/comic_benchmark/det
cp -r data/comic_benchmark/det PaddleOCR/comic_benchmark

# 2 Download the PP-OCRv5_server_det pre-trained model
cd PaddleOCR
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams

# 3. Model Fine-tune
python3 tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_comic_det.yml \
    -o Global.pretrained_model=./PP-OCRv5_server_det_pretrained.pdparams \
    Train.dataset.data_dir=./comic_benchmark/det \
    Train.dataset.label_file_list='[./comic_benchmark/det/train.txt]' \
    Eval.dataset.data_dir=./comic_benchmark/det \
    Eval.dataset.label_file_list='[./comic_benchmark/det/val.txt]'


python3 tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_comic_det.yml \
    -o Global.pretrained_model=./PP-OCRv5_server_det_pretrained.pdparams \
    Train.dataset.data_dir=./comic_benchmark/det-example \
    Train.dataset.label_file_list='[./comic_benchmark/det/train.txt]' \
    Eval.dataset.data_dir=./comic_benchmark/det \
    Eval.dataset.label_file_list='[./comic_benchmark/det/val.txt]'


# 4. Model Export
python3 tools/export_model.py -c configs/det/PP-OCRv5/PP-OCRv5_comic_det.yml -o \
    Global.pretrained_model=output/PP-OCRv5_comic_det/latest.pdparams \
    Global.save_inference_dir="output/inference/PP-OCRv5_comic_det_infer/"
```

Finetune recognition module
```bash
# 1. Prepare Dataset
python core/prep_ret_dataset.py

# 2. Download the PP-OCRv5_server_rec pre-trained model
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams

# 3. Model Fine-tune
python3 tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_comic_rec.yml \
   -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams \
    Train.dataset.data_dir=./comic_benchmark/rec \
    Train.dataset.label_file_list='[./comic_benchmark/rec/train.txt]' \
    Eval.dataset.data_dir=./comic_benchmark/rec \
    Eval.dataset.label_file_list='[./comic_benchmark/rec/val.txt]'

# 4. Model Evaluation
python tools/eval.py -c configs/rec/PP-OCRv5/PP-OCRv5_comic_rec.yml \
    -o Global.pretrained_model=./output/PP-OCRv5_comic_rec_epoch200/latest.pdparams \
    Eval.dataset.data_dir=./comic_benchmark/rec     \
    Eval.dataset.label_file_list='[./comic_benchmark/rec/val.txt]'

# 5. Model Export
python3 tools/export_model.py -c configs/rec/PP-OCRv5/PP-OCRv5_comic_rec.yml -o \
    Global.pretrained_model=output/PP-OCRv5_comic_rec/latest.pdparams \
    Global.save_inference_dir="output/inference/PP-OCRv5_comic_rec_infer/"
```


## Evaluation
### OCR Part
prepare ocr groundtruth
```bash
# 1. get annotations
python ui/dataset_annotator.py

# 2. convert to corresponding format
python3 core/ocr_gt_converter.py  \
    --input data/comic_benchmark/det/annotations.txt \
    --output data/benchmark/ocr_groundtruth
```

evaluation
```bash
# text: CER metrics; box: mIoU、precision
python core/eval/eval_ocr_gt.py \
    --pred-file results/2/ocr.json \
    --gt-dir data/benchmark/ocr_groundtruth
```

### Translation Part
```bash
python ui/translation_gt_annotater.py 

# text: CER metrics;
python core/eval/eval_translation_gt.py -p results/2/translated.json -g data/benchmark/translation_groundtruth
```

## 流程說明

1. **OCR** ([ocr.py](utils/ocr.py)): 使用 PaddleOCR 辨識圖片文字，輸出 bounding boxes + 文字
2. **Translation** ([translation_model.py](utils/translation_model.py)): 使用 SakuraLLM 翻譯日文→繁中
3. **Render** ([image_renderer.py](utils/image_renderer.py)): 將翻譯文字渲染回原圖對應位置

## Issues

AttributeError: module 'httpcore' has no attribute 'SyncHTTPTransport'
(ocr) howard@wingene-76:/tmp2/howard/japcomic2twcomic$ pip install googletrans