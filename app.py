from paddleocr import PaddleOCR
import cv2
import numpy as np
from google.colab import files
from IPython.core.display import display, HTML
import base64

# Upload Image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Read and preprocess image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to Base64 for HTML display
_, buffer = cv2.imencode(".png", image_rgb)
image_base64 = base64.b64encode(buffer).decode("utf-8")
image_src = f"data:image/png;base64,{image_base64}"

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Perform OCR
result = ocr.ocr(image_path, cls=True)

# Function to sort OCR results for better formatting
def sort_ocr_results(results):
    """Sorts OCR results in reading order (top to bottom, left to right)."""
    return sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))  # Sort by Y first, then X

# Sort OCR output
sorted_result = sort_ocr_results([line for res in result for line in res])

# Extract text in structured format like original image
formatted_text = "<pre>"

i = 0
while i < len(sorted_result):
    left_col = sorted_result[i][1][0]  # Extract first part (field name)
    right_col = ""
    i += 1

    # Handle multi-line values
    while i < len(sorted_result) and sorted_result[i][0][0][0] > sorted_result[i-1][0][1][0]:
        right_col += " " + sorted_result[i][1][0]
        i += 1

    formatted_text += f"{left_col.ljust(20)} {right_col.strip()}\n"

formatted_text += "</pre>"

# Create an HTML layout for displaying the image and formatted text
html_code = f"""
    <style>
        .text-container {{
            border: 1px solid #ccc;
            padding: 10px;
            font-family: monospace;
            background-color: #fff;
            outline: none;
            width: 100%;
            white-space: pre;
        }}
    </style>

    <div style="display: flex; align-items: flex-start;">
        <div style="flex: 1;">
            <img src="{image_src}" width="500">
        </div>
        <div style="flex: 1; padding-left: 20px;">
            <h3>Extracted Information</h3>
            <div contenteditable="true" class="text-container">
                {formatted_text}
            </div>
        </div>
    </div>
"""

# Display formatted output
display(HTML(html_code))
