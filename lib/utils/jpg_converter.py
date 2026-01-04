#!/usr/bin/env python3
"""
JPG Converter - 將目錄中的所有圖片轉換為 JPG 格式

使用方式:
    python jpg_converter.py --input data/1 --output data/1/jpg/
"""

import os
import argparse
from PIL import Image
from pathlib import Path


def convert_to_jpg(input_dir, output_dir, quality=95, recursive=True):
    """
    將目錄中的所有圖片轉換為 JPG 格式

    Args:
        input_dir: 輸入目錄路徑
        output_dir: 輸出目錄路徑
        quality: JPG 品質 (1-100)
        recursive: 是否遞迴處理子目錄
    """
    # 支援的圖片格式
    supported_formats = {'.png', '.webp', '.bmp', '.gif', '.tiff', '.tif', '.jpeg', '.jpg'}

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"輸入目錄不存在：{input_dir}")

    # 創建輸出目錄
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有圖片檔案
    if recursive:
        image_files = []
        for ext in supported_formats:
            image_files.extend(input_path.rglob(f'*{ext}'))
            image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    else:
        image_files = []
        for ext in supported_formats:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))

    # 去重並排序
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"⚠️  在 {input_dir} 中沒有找到支援的圖片檔案")
        return

    print(f"📁 找到 {len(image_files)} 個圖片檔案")
    print(f"🔄 開始轉換...\n")

    success_count = 0
    error_count = 0

    for idx, img_file in enumerate(image_files, 1):
        try:
            # 計算相對路徑以保持目錄結構
            rel_path = img_file.relative_to(input_path)

            # 創建輸出檔案路徑（將副檔名改為 .jpg）
            output_file = output_path / rel_path.with_suffix('.jpg')

            # 創建輸出子目錄
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # 開啟圖片
            image = Image.open(img_file)

            # 轉換為 RGB 模式（JPG 不支援透明度）
            if image.mode in ('RGBA', 'LA', 'P'):
                # 有透明度的格式，轉換為 RGB（白色背景）
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            elif image.mode != 'RGB':
                # 其他格式轉換為 RGB
                image = image.convert('RGB')

            # 儲存為 JPG
            image.save(output_file, 'JPEG', quality=quality, optimize=True)

            print(f"✅ [{idx}/{len(image_files)}] {rel_path} -> {output_file.name}")
            success_count += 1

        except Exception as e:
            print(f"❌ [{idx}/{len(image_files)}] {img_file.name} - 錯誤: {e}")
            error_count += 1

    print(f"\n{'='*60}")
    print(f"📊 轉換完成！")
    print(f"✅ 成功: {success_count} 個")
    print(f"❌ 失敗: {error_count} 個")
    print(f"💾 輸出目錄: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='將目錄中的所有圖片轉換為 JPG 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python jpg_converter.py --input data/1 --output data/1/jpg/
  python jpg_converter.py --input images --output images_jpg --quality 90
  python jpg_converter.py --input photos --output photos_jpg --no-recursive
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='輸入目錄路徑'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='輸出目錄路徑'
    )

    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPG 品質 (1-100，預設: 95)'
    )

    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不遞迴處理子目錄'
    )

    args = parser.parse_args()

    # 驗證品質參數
    if not 1 <= args.quality <= 100:
        print("❌ 錯誤：品質參數必須在 1-100 之間")
        return

    try:
        convert_to_jpg(
            input_dir=args.input,
            output_dir=args.output,
            quality=args.quality,
            recursive=not args.no_recursive
        )
    except Exception as e:
        print(f"❌ 錯誤：{e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
