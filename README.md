python utils/ocr_model.py --input_dir data/2 --output_path results/ocr.json

python utils/translation_model.py --input_path results/ocr.json --output_path results/translated.json
