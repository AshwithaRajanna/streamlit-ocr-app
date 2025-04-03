import streamlit as st
import cv2
import numpy as np
import base64
from paddleocr import PaddleOCR
from PIL import Image
import io

# Set Page Configuration
st.set_page_config(page_title="OCR App", layout="wide")

# Title of the App
st.title("📝 Extract Text from Images using PaddleOCR")

# Instructions
st.markdown("📌 **Paste an Image (CTRL + V) or Upload a File Below**")

# File Uploader (CTRL + V paste option enabled)
uploaded_file = st.file_uploader("Upload an image or paste one here", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image from uploaded file
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Convert image to OpenCV format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert image to Base64 for displaying
    _, buffer = cv2.imencode(".png", image_rgb)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    image_src = f"data:image/png;base64,{image_base64}"

    # Display Image in Streamlit
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en")

    # Perform OCR
    result = ocr.ocr(np.array(image), cls=True)

    # Function to sort OCR results for better formatting
    def sort_ocr_results(results):
        """Sorts OCR results in reading order (top to bottom, left to right)."""
        return sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))  # Sort by Y first, then X

    # Sort OCR output
    sorted_result = sort_ocr_results([line for res in result for line in res])

    # Extract structured text in original image format
    extracted_text = ""

    i = 0
    while i < len(sorted_result):
        left_col = sorted_result[i][1][0]  # Extract field name
        right_col = ""
        i += 1

        # Handle multi-line values
        while i < len(sorted_result) and sorted_result[i][0][0][0] > sorted_result[i - 1][0][1][0]:
            right_col += " " + sorted_result[i][1][0]
            i += 1

        extracted_text += f"{left_col.ljust(20)} {right_col.strip()}\n"

    # Display OCR extracted text in a formatted way
    st.subheader("📄 Extracted Text:")
    st.text_area("Text Output", value=extracted_text, height=300)

    # Download button for extracted text
    st.download_button("Download Extracted Text", extracted_text, file_name="ocr_text.txt")

