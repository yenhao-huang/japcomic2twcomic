# 日漫繁化流程

source ../venv_manager/ocr/bin/activate

## II. Pipeline2: pixel-level inpaint

1. detect bbox and segmentation mask and cropping
layout/yolo.py

input data/2
output results/layout/yolo/layout.json, results/layout/yolo/cropped_dir

python -m lib.layout.yolo --input_dir data/2

2. ocr module: input layout outpu text
call lib.ocr.paddleocr_vl_manga.py
input:  results/layout/yolo/layout.json
output: results/ocr/paddleocr_vl_manga.json

python -m lib.ocr.paddleocr_vl_manga   --layout-json results/layout/yolo/layout.json --output-dir results/ocr/

3. translation
call lib/translation/hunyuan.py
input: results/ocr/ocr.json
output: results/translation/hunyuan.json

python -m lib.translation.hunyuan --input_path results/ocr/paddleocrvl_layout.json --output_path results/translation/hunyuan.json

4. render
call lib/render/image_render.py
input: results/translation/hunyuan.json
output: results/render/

python -m lib.render.layout_render     --layout results/layout/yolo/layout.json     --translation results/translation/hunyuan.json     --output results/render/     --font lib/utils/fonts/Noto_Sans_CJ
K_Regular.otf

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
# 2 Download the PP-OCRv5_server_det pre-trained model
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams

# 3. Model Fine-tune
python3 tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_comic_det.yml \
    -o Global.pretrained_model=./PP-OCRv5_server_det_pretrained.pdparams \
    Train.dataset.data_dir=./comic_benchmark/det \
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