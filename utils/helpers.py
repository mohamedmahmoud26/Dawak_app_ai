# -*- coding: utf-8 -*-
"""
أدوات مساعدة - Helper Functions
"""
import streamlit as st
from config import CONFIDENCE_THRESHOLD
from ultralytics import YOLO
import easyocr
@st.cache_resource
def load_yolo_model(model_path):
    """
    تحميل نموذج YOLO - Load YOLO model
    """
    return YOLO(model_path)

@st.cache_resource
def load_ocr_reader(languages):
    """
    تحميل قارئ OCR - Load OCR reader
    """
    return easyocr.Reader(languages)

def display_drug_info(drug_name, drug_info):
    """
    عرض معلومات الدواء - Display drug information
    """
    if drug_info:
        st.subheader(f"معلومات عن {drug_name}:")
        st.write(f"**الاسم التجاري:** {drug_info['brand_name']}")
        st.write(f"**المادة الفعالة:** {drug_info['generic_name']}")
        st.write(f"**الشركة المصنعة:** {drug_info['manufacturer']}")
        st.write(f"**الغرض:** {drug_info['purpose']}")
        st.write(f"**الآثار الجانبية:** {drug_info['warnings']}")
        st.write(f"**الجرعة:** {drug_info['dosage']}")
    else:
        st.warning(f"لم يتم العثور على معلومات لـ {drug_name} في قاعدة البيانات")