from manga_ocr import MangaOcr

mocr = MangaOcr()
text = mocr("data/2/109994537_p0_master1200.jpg")
print(text)