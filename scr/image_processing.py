"""
Image Processing Module

This module provides utility functions for capturing, uploading, saving,
and preprocessing images for OCR and computer vision tasks.
"""

import cv2
import numpy as np
import tempfile
import os
from typing import Optional


def capture_image_from_camera() -> Optional[np.ndarray]:
    """
    Capture an image directly from the user's camera using Streamlit.

    Returns:
        Optional[np.ndarray]: Captured image in OpenCV format (BGR). 
                              Returns None if no image is captured.
    """
    import streamlit as st
    img_file_buffer = st.camera_input(" Capture a photo of the medicine package")

    if img_file_buffer is not None:
        # Convert image buffer to OpenCV format
        bytes_data = img_file_buffer.getvalue()
        img_array = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    return None


def upload_image() -> Optional[np.ndarray]:
    """
    Upload an image file from the local device using Streamlit.

    Returns:
        Optional[np.ndarray]: Uploaded image in OpenCV format (BGR).
                              Returns None if no file is uploaded.
    """
    import streamlit as st
    uploaded_file = st.file_uploader(" Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image

    return None


def save_image_temp(image: np.ndarray) -> str:
    """
    Save an image temporarily on disk for further processing.

    Args:
        image (np.ndarray): Image in OpenCV format.

    Returns:
        str: Path to the temporary saved image file.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.close()

    cv2.imwrite(temp_path, image)
    return temp_path


def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image to improve OCR accuracy.

    Steps:
        1. Convert to grayscale.
        2. Apply Otsu's thresholding.

    Args:
        image (np.ndarray): Input image in OpenCV format.

    Returns:
        np.ndarray: Preprocessed binary image ready for OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


if __name__ == "__main__":
    # Example test when running module directly
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Save dummy image temporarily
    path = save_image_temp(dummy_img)
    print(f" Temporary image saved at: {path}")

    # Preprocess dummy image
    processed = preprocess_image_for_ocr(dummy_img)
    print(" Preprocessing complete. Shape:", processed.shape)

    # Clean up
    if os.path.exists(path):
        os
