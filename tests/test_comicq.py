from comiq import ComiQ
import cv2
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("MLLM_API_KEY")

# Initialize ComiQ with the API key
comiq = ComiQ(api_key=api_key)

# Process an image from a file path
image_path = "data/2/109994537_p0_master1200.jpg"
data = comiq.extract(image_path, ocr="paddleocr")

# Or process an image from a numpy array
image_array = cv2.imread(image_path)
data_from_array = comiq.extract(image_array)

# 'data' now contains a list of text bubbles with their text and locations.
print(data)