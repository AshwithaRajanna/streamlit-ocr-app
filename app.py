import streamlit as st
import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.title("OCR Text Extraction with EasyOCR")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    # Improve contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(sharpened)

    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(contrast_img, h=10, templateWindowSize=7, searchWindowSize=21)

    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])

    # Perform OCR
    results = reader.readtext(denoised)

    # Extract text and format it
    formatted_text = ""
    i = 0
    while i < len(results):
        left_col = results[i][1]
        right_col = ""
        i += 1
        while i < len(results) and results[i][0][0][0] > results[i - 1][0][1][0]:
            right_col += results[i][1] + " "
            i += 1
        formatted_text += f"{left_col:<20} {right_col.strip()}\n"

    # Display images
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display extracted text
    st.subheader("Extracted Text")
    st.text(formatted_text)
