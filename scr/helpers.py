"""
Helper functions for loading models and displaying drug information
"""

import easyocr
from ultralytics import YOLO
import streamlit as st


def load_yolo_model(model_path: str) -> YOLO:
    """
    Load a YOLO model from the specified file path.

    Args:
        model_path (str): Path to the YOLO model (.pt file).

    Returns:
        YOLO: An initialized YOLO model ready for inference.
    """
    return YOLO(model_path)


def load_ocr_reader(languages: list[str]) -> easyocr.Reader:
    """
    Initialize an EasyOCR reader for the given languages.

    Args:
        languages (list[str]): List of language codes (e.g., ["en", "ar"]).

    Returns:
        easyocr.Reader: An OCR reader instance that can extract text from images.
    """
    return easyocr.Reader(languages)


def display_drug_info(drug_name: str, drug_info: dict | None) -> None:
    """
    Display drug information in Streamlit.

    Args:
        drug_name (str): The name of the drug being displayed.
        drug_info (dict | None): Dictionary of drug details. If None, a fallback message is shown.

    Returns:
        None
    """
    if drug_info:
        st.write(f"### Drug Information: **{drug_name}**")
        for key, value in drug_info.items():
            st.write(f"- **{key.capitalize()}**: {value}")
    else:
        st.write(f"⚠️ No information available for **{drug_name}**")
