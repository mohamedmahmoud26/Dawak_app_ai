# -*- coding: utf-8 -*-
"""
معالجة الصور - Image Processing Module
"""
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

def capture_image_from_camera():
    """
    التقاط صورة من الكاميرا - Capture image from camera
    """
    import streamlit as st
    img_file_buffer = st.camera_input("التقاط صورة للعلبة الدوائية")
    
    if img_file_buffer is not None:
        # تحويل إلى صيغة OpenCV
        bytes_data = img_file_buffer.getvalue()
        img_array = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    return None

def upload_image():
    """
    رفع صورة من الجهاز - Upload image from device
    """
    import streamlit as st
    uploaded_file = st.file_uploader("اختر صورة", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # تحويل إلى صيغة OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    return None

def save_image_temp(image):
    """
    حفظ الصورة مؤقتاً للتنبؤ - Save image temporarily for prediction
    """
    # إنشاء ملف مؤقت
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.close()
    
    # حفظ الصورة
    cv2.imwrite(temp_path, image)
    return temp_path

def preprocess_image_for_ocr(image):
    """
    معالجة الصورة لتحسين دقة OCR - Preprocess image for better OCR accuracy
    """
    # تحويل الصورة إلى تدرجات الرمادي
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # تطبيق thresholding لتحسين النص
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh