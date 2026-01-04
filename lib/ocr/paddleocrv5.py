import os
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
# 啟用記憶體優化
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
os.environ['FLAGS_fast_eager_deletion_mode'] = '1'

from paddleocr import PaddleOCR
import base64
from io import BytesIO
from PIL import Image
import argparse
import sys

import json
import glob
from datetime import datetime
import paddle
import gc
import shutil


class OCR:
    def __init__(self, text_det_model_dir, text_recognition_model_dir=None, max_output_dirs: int = 5, output_dir: str = None):

        self.ocr = PaddleOCR(
            text_recognition_model_dir=text_recognition_model_dir,
            text_detection_model_dir=text_det_model_dir,
            use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
            use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
            use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
            lang="japan",
        )
        # 設置臨時目錄
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output_dir")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tmp_dir = os.path.join(output_dir, f"layout/{timestamp}")
        os.makedirs(self.tmp_dir, exist_ok=True)

        # 設置最大保留的輸出目錄數量
        self.max_output_dirs = max_output_dirs

        print(f"✅ OCR 模型已載入 (PID: {os.getpid()})")
        print(f"   GPU 記憶體限制: 40%")

    def _process_image_input(self, image_input):
        """
        處理不同格式的圖片輸入，轉換為檔案路徑

        Args:
            image_input: 圖片路徑 或 PIL Image 或 base64 字串

        Returns:
            str: 圖片檔案路徑
        """
        img_ext = ".jpg"  # 預設
        if isinstance(image_input, str):
            if image_input.startswith('data:image') or image_input.startswith('/9j'):
                # base64 輸入
                image_data = base64.b64decode(image_input.split(',')[-1])
                image = Image.open(BytesIO(image_data))
                img_format = image.format.lower() if image.format else "jpg"
                img_ext = f".{img_format}"
                img_path = os.path.join(self.tmp_dir, f"temp_ocr{img_ext}")
                image.save(img_path)
            else:
                # 檔案路徑
                img_path = image_input
                img_ext = os.path.splitext(img_path)[1] or ".jpg"
        else:
            # PIL Image
            img_format = image_input.format.lower() if image_input.format else "jpg"
            img_ext = f".{img_format}"
            img_path = os.path.join(self.tmp_dir, f"temp_ocr{img_ext}")
            image_input.save(img_path)

        return img_path

    def _cleanup_old_outputs(self):
        """清理舊的輸出目錄，只保留最新的 N 個"""
        output_dirs = sorted(
            [d for d in glob.glob(os.path.join(self.tmp_dir, "ocr_output_*")) if os.path.isdir(d)],
            key=os.path.getmtime
        )

        # 刪除超過限制的舊目錄
        if len(output_dirs) > self.max_output_dirs:
            for old_dir in output_dirs[:-self.max_output_dirs]:
                try:
                    shutil.rmtree(old_dir)
                    print(f"🗑️  已清理舊輸出目錄：{old_dir}")
                except Exception as e:
                    print(f"⚠️  清理目錄失敗：{old_dir}, 錯誤：{e}")

    def _clear_gpu_memory(self):
        """清理 GPU 記憶體"""
        try:
            # 清空 PaddlePaddle 的緩存
            #torch.cuda.empty_cache()
            #paddle.inference.Predictor.try_shrink_memory()
            # 強制 Python 垃圾回收
            gc.collect()
        except Exception as e:
            print(f"⚠️  清理 GPU 記憶體時出錯：{e}")

    def load_directory(self, directory_path, recursive=True):
        """
        載入目錄中的所有圖片檔案

        Args:
            directory_path: 圖片目錄路徑
            recursive: 是否遞迴搜尋子目錄（預設為 True）

        Returns:
            list[str]: 圖片檔案路徑列表
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"目錄不存在：{directory_path}")

        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"路徑不是目錄：{directory_path}")

        # 支援的圖片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

        image_paths = []

        if recursive:
            # 遞迴搜尋所有子目錄
            for root, dirs, files in os.walk(directory_path):
                for file in sorted(files):
                    if os.path.splitext(file.lower())[1] in image_extensions:
                        image_paths.append(os.path.join(root, file))
        else:
            # 只搜尋當前目錄
            for file in sorted(os.listdir(directory_path)):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path) and os.path.splitext(file.lower())[1] in image_extensions:
                    image_paths.append(file_path)

        return image_paths

    def predict_batch(self, image_inputs, show_progress=True):
        """
        批次執行 OCR 辨識

        Args:
            image_inputs: 圖片路徑列表 或 PIL Image 列表
            show_progress: 是否顯示進度

        Returns:
            list[dict]: 每張圖片的辨識結果列表
        """
        results = []
        total = len(image_inputs)

        for idx, image_input in enumerate(image_inputs, 1):
            if show_progress:
                print(f"🔄 處理進度：{idx}/{total} - {os.path.basename(image_input) if isinstance(image_input, str) else f'image_{idx}'}")

            try:
                result = self.predict(image_input)
                result['source'] = image_input if isinstance(image_input, str) else f'image_{idx}'
                results.append(result)
            except Exception as e:
                print(f"⚠️  處理失敗：{image_input if isinstance(image_input, str) else f'image_{idx}'}, 錯誤：{e}")
                results.append({
                    'source': image_input if isinstance(image_input, str) else f'image_{idx}',
                    'error': str(e)
                })

        return results

    def predict(self, image_input):
        """
        執行 OCR 辨識

        Args:
            image_input: 圖片路徑 或 PIL Image 或 base64 字串
        Returns:
            dict: {
                "text": str,                    # 所有辨識出的文字（換行分隔）
                "image_path": str,              # 標註圖片的路徑
                "bounding_boxes": list[dict]    # 每個文字區域的座標、文字和信心分數
                                                # 格式: [{"box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                                                #         "text": str, "score": float}, ...]
            }
        """
        try:
            # 處理輸入格式並取得圖片路徑
            img_path = self._process_image_input(image_input)

            # OCR 辨識
            result = self.ocr.predict(input=img_path)

            # 儲存結果（加入 timestamp）
            ocr_output_dir = self.tmp_dir
            os.makedirs(ocr_output_dir, exist_ok=True)

            for res in result:
                res.print()
                res.save_to_json(save_path=ocr_output_dir)
                res.save_to_img(save_path=ocr_output_dir)

            # 提取文字內容
            json_files = glob.glob(os.path.join(ocr_output_dir, "*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {ocr_output_dir}")

            with open(json_files[0], 'r', encoding='utf-8') as f:
                json_result = json.load(f)

            
            # 提取圖片連結
            result_image_path = sorted([f for f in os.listdir(ocr_output_dir) if f.endswith(tuple(['.jpg', '.jpeg', '.png']))])[0]

            # 提取 bounding boxes
            bounding_boxes = []
            if "dt_polys" in json_result:
                for i, box in enumerate(json_result["dt_polys"]):
                    bbox_info = {
                        "box": box,
                        "text": json_result["rec_texts"][i] if i < len(json_result["rec_texts"]) else "",
                        "score": json_result["rec_scores"][i] if "rec_scores" in json_result and i < len(json_result["rec_scores"]) else None
                    }
                    bounding_boxes.append(bbox_info)

            response = {
                "text": "\n".join(json_result["rec_texts"]),
                "image_path": result_image_path,
                "bounding_boxes": bounding_boxes
            }

            return response

        finally:
            # 每次推理後清理 GPU 記憶體
            self._clear_gpu_memory()
            # 清理舊的輸出目錄
            self._cleanup_old_outputs()


def main():
    parser = argparse.ArgumentParser(description='PaddleOCR-VL 圖片文字辨識')
    parser.add_argument('--image', type=str, help='單張圖片路徑')
    parser.add_argument('--input-dir', type=str, help='圖片目錄路徑（批次處理）')
    parser.add_argument('--no-recursive', action='store_true', help='不遞迴搜尋子目錄（僅用於 --input-dir）')
    parser.add_argument('--model-dir', type=str,
                        default='/tmp2/share_data/models--PaddlePaddle--PaddleOCR-VL/',
                        help='模型路徑')
    parser.add_argument('--text-det-model-dir', type=str,
                        default=None,
                        help='文字偵測模型路徑')
    parser.add_argument('--text-rec-model-dir', type=str,
                        default=None,
                        help='文字辨識模型路徑 (預設: PaddleOCR/output/inference/PP-OCRv5_comic_rec_infer_epoch200)')
    parser.add_argument('--output', type=str, help='輸出文字檔案路徑（單張圖片）或合併文字檔案路徑（批次處理）')
    parser.add_argument('--output-dir', type=str, help='輸出目錄路徑（批次處理時每張圖片分別儲存）')
    parser.add_argument('--save-image', type=str, help='儲存標註圖片路徑（選填）')

    args = parser.parse_args()

    # 檢查必須提供 --image 或 --input-dir 其中之一
    if not args.image and not args.input_dir:
        print(f"❌ 錯誤：請提供 --image 或 --input-dir 參數", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if args.image and args.input_dir:
        print(f"❌ 錯誤：--image 和 --input-dir 不能同時使用", file=sys.stderr)
        sys.exit(1)

    print(f"🔧 載入 OCR 模型...")
    ocr = OCR(
        text_det_model_dir=args.text_det_model_dir,
        text_recognition_model_dir=args.text_rec_model_dir,
        output_dir=args.output_dir
    )

    # 單張圖片處理
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ 錯誤：圖片檔案不存在 - {args.image}", file=sys.stderr)
            sys.exit(1)

        print(f"🖼️  處理圖片：{args.image}")
        result = ocr.predict(args.image)

        # 輸出辨識文字
        print("\n📝 完整 JSON 結果：")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 60)

        # 儲存輸出（如果指定）
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            print(f"\n💾 文字已儲存至：{args.output}")


    # 目錄批次處理
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"❌ 錯誤：目錄不存在 - {args.input_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"📁 載入目錄：{args.input_dir}")
        recursive = not args.no_recursive
        if not recursive:
            print(f"   模式：僅搜尋當前目錄（不遞迴）")
        else:
            print(f"   模式：遞迴搜尋所有子目錄")
        image_paths = ocr.load_directory(args.input_dir, recursive=recursive)

        if not image_paths:
            print(f"⚠️  警告：目錄中沒有找到圖片檔案", file=sys.stderr)
            sys.exit(0)

        print(f"✅ 找到 {len(image_paths)} 張圖片\n")

        # 批次處理
        results = ocr.predict_batch(image_paths)

        # 輸出結果摘要
        print("\n" + "=" * 60)
        print("📊 批次處理結果摘要：")
        print("=" * 60)

        success_count = sum(1 for r in results if 'error' not in r)
        error_count = len(results) - success_count

        print(f"✅ 成功：{success_count} 張")
        print(f"❌ 失敗：{error_count} 張")

        # 儲存結果為 JSON 檔
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            results_json_path = os.path.join(args.output_dir, "ocr.json")
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 批次結果已儲存至：{results_json_path}")


if __name__ == "__main__":
    main()
