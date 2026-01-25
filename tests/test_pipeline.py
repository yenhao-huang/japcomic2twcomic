import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from PIL import Image
from core.manga_translator import MangaTranslator


# Initialize translator (models are lazy-loaded)
translator = MangaTranslator(config_path="configs/pipeline.yml")

# Translate a single image
image = Image.open("data/2/109994537_p0_master1200.jpg")
output_image, results = translator.translate_image(image)

# Save result
output_image.save("output.jpg")