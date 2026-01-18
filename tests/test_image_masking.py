from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color

o_dir = "tests/results/"

# 步驟 1: 將 JPG 轉成帶透明通道的 PNG 並保存
print("步驟 1: 轉換 JPG -> PNG")
with Image(filename='data/2/109994537_p1_master1200.jpg') as img:
    # 轉換為 PNG 格式並添加透明通道 (預設 alpha=255 完全不透明)
    img.format = 'png'
    img.alpha_channel = 'opaque'  # 使用 'opaque' 確保 alpha 通道設為 255 且保留

    if img.alpha_channel:
        print("✓ 已轉換為 PNG 有透明通道 (RGBA)")
    else:
        print("✗ PNG 沒有透明通道 (RGB)")

    # 保存時指定 PNG 格式並確保保留 alpha
    img.depth = 8
    img.type = 'truecolormatte'  # 明確指定為 RGBA
    img.save(filename=f"{o_dir}/原圖.png")
    print(f"✓ 已保存: {o_dir}/原圖.png")

# 步驟 2: 加載 PNG 並應用遮罩
print("\n步驟 2: 加載 PNG 並應用遮罩")
with Image(filename=f"{o_dir}/原圖.png") as img:
    if img.alpha_channel:
        print("✓ 已轉換為 PNG 有透明通道 (RGBA)")
    else:
        print("✗ PNG 沒有透明通道 (RGB)")
        
    # 獲取圖片尺寸
    width = img.width
    height = img.height
    print(f"圖片尺寸: {width}x{height}")

    # 建立不規則遮罩 (六角形)
    with Image(width=width, height=height, background=Color('black')) as mask:
        with Drawing() as draw:
            draw.fill_color = Color('white')
            # 畫一個正六邊形 (根據圖片尺寸調整)
            cx, cy = width // 2, height // 2  # 中心點
            r = min(width, height) // 3  # 半徑
            points = [
                (cx, cy - r),
                (cx + int(r * 0.866), cy - r // 2),
                (cx + int(r * 0.866), cy + r // 2),
                (cx, cy + r),
                (cx - int(r * 0.866), cy + r // 2),
                (cx - int(r * 0.866), cy - r // 2)
            ]
            draw.polygon(points)
            draw(mask)
            mask.save(filename=f'{o_dir}/mask.png')
            print(f"✓ 已保存遮罩: {o_dir}/mask.png")

        # 將圖片與遮罩合成
        # 使用遮罩的紅色通道覆蓋圖片的 alpha 通道
        # 白色(255) -> 不透明, 黑色(0) -> 透明
        img.composite_channel('alpha', mask, 'copy_red', 0, 0)
        img.format = 'png'  # 確保支持透明度
        img.save(filename=f'{o_dir}/result.png')
        print(f"✓ 已保存結果: {o_dir}/result.png")

        test_pixel = img[0, 0]
        print(f"左上角像素: {test_pixel}")