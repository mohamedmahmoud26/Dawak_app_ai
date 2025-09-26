"""
Helper Functions
"""

import streamlit as st
from config import CONFIDENCE_THRESHOLD
from ultralytics import YOLO
import easyocr


@st.cache_resource
def load_yolo_model(model_path):
    """
    Load YOLO model from the given path.
    
    Args:
        model_path (str): Path to the YOLO model file.
    
    Returns:
        YOLO: Loaded YOLO model instance.
    """
    return YOLO(model_path)


@st.cache_resource
def load_ocr_reader(languages):
    """
    Initialize the EasyOCR reader with the given languages.
    
    Args:
        languages (list): List of language codes (e.g., ['en', 'ar']).
    
    Returns:
        easyocr.Reader: OCR reader instance.
    """
    return easyocr.Reader(languages)


def display_drug_info(drug_name, drug_info):
    """
    Display drug information using Streamlit.
    
    Args:
        drug_name (str): The name of the drug.
        drug_info (dict): Dictionary containing drug details.
    """
    if drug_info:
        st.subheader(f"Information about {drug_name}:")
        st.write(f"**Brand Name:** {drug_info['brand_name']}")
        st.write(f"**Generic Name:** {drug_info['generic_name']}")
        st.write(f"**Manufacturer:** {drug_info['manufacturer']}")
        st.write(f"**Purpose:** {drug_info['purpose']}")
        st.write(f"**Side Effects:** {drug_info['warnings']}")
        st.write(f"**Dosage:** {drug_info['dosage']}")
    else:
        st.warning(f"No information found for {drug_name} in the database.")
