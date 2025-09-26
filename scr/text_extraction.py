"""
Text Extraction Module

This module provides utilities to:
1. Extract text from images using YOLO for region detection + EasyOCR for recognition.
2. Clean extracted texts for better downstream matching (e.g., drug name matching).
"""

import cv2
import easyocr
import re
import os
from typing import List
from scr.image_processing import save_image_temp


def extract_text_with_yolo(model, reader: easyocr.Reader, image, conf_threshold: float = 0.5) -> List[str]:
    """
    Extract text from an image using YOLO for object detection and EasyOCR for text recognition.

    Args:
        model: YOLO object detection model instance.
        reader (easyocr.Reader): Initialized EasyOCR reader.
        image (np.ndarray): Input image in OpenCV BGR format.
        conf_threshold (float, optional): Confidence threshold for YOLO detections. Defaults to 0.5.

    Returns:
        List[str]: List of recognized text strings extracted from the detected regions.
    """
    texts = []
    temp_path = save_image_temp(image)

    try:
        results = model.predict(temp_path, verbose=False)

        # Iterate over YOLO detection results
        for r in results:
            for box in r.boxes:
                conf = box.conf[0].cpu().numpy()

                if conf >= conf_threshold:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Crop the detected region from the image
                    crop = image[y1:y2, x1:x2]

                    # Run OCR on the cropped region
                    result = reader.readtext(crop, detail=0)
                    texts.extend(result)

    finally:
        # Ensure temporary file is removed
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    return texts


def clean_extracted_texts(texts: List[str]) -> List[str]:
    """
    Clean and normalize extracted text strings to improve matching accuracy.

    Cleaning steps:
        - Convert to lowercase.
        - Remove non-alphabetic characters.
        - Strip leading/trailing whitespace.
        - Keep words longer than 2 characters.

    Args:
        texts (List[str]): Raw OCR-extracted texts.

    Returns:
        List[str]: List of cleaned and normalized text strings.
    """
    cleaned = []
    for t in texts:
        t = t.lower()
        t = re.sub(r'[^a-z\s]', '', t)
        t = t.strip()
        if t and len(t) > 2:
            cleaned.append(t)
    return cleaned


if __name__ == "__main__":
    # Example test for cleaning (without YOLO/EasyOCR)
    sample_texts = ["Panadol® 500mg", "IBUPROFEN!!", "ok", "12"]
    cleaned = clean_extracted_texts(sample_texts)

    print("Original texts:", sample_texts)
    print("Cleaned texts:", cleaned)
