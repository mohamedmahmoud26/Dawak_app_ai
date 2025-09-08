# -*- coding: utf-8 -*-
"""
الملف الرئيسي للتشغيل - Main Application File
"""
import streamlit as st
import cv2
import numpy as np

# استيراد الوحدات النمطية
from config import MODEL_PATH, OCR_LANGUAGES
from modules.image_processing import capture_image_from_camera, upload_image
from modules.text_extraction import extract_text_with_yolo, clean_extracted_texts
from modules.drug_matching import match_drug_names
from modules.api_handler import get_drug_info_from_api
from utils.helpers import load_yolo_model, load_ocr_reader, display_drug_info

def main():
    """
    الوظيفة الرئيسية للتطبيق - Main application function
    """
    st.title("نظام التعرف على الأدوية واستخراج معلوماتها")
    st.write("التقط صورة لعلبة الدواء أو ارفع صورة موجودة")
    
    # تحميل النماذج
    model = load_yolo_model(MODEL_PATH)
    reader = load_ocr_reader(OCR_LANGUAGES)
    
    # خيارات إدخال الصورة
    option = st.radio("اختر طريقة الإدخال:", ("الكاميرا", "رفع صورة"))
    
    image = None
    if option == "الكاميرا":
        image = capture_image_from_camera()
    else:
        image = upload_image()
    
    if image is not None:
        # عرض الصورة المختارة
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="الصورة المختارة", use_column_width=True)
        
        # معالجة الصورة
        with st.spinner("جاري معالجة الصورة..."):
            try:
                # استخراج النصوص
                texts = extract_text_with_yolo(model, reader, image)
                
                # تنظيف النصوص
                cleaned_texts = clean_extracted_texts(texts)
                
                # مطابقة الأدوية
                matched_drugs = match_drug_names(cleaned_texts)
                
                # عرض النتائج
                if matched_drugs:
                    st.success("تم التعرف على الأدوية التالية:")
                    
                    for word, match, score in matched_drugs:
                        st.write(f"**{match}** (مطابقة بنسبة {score}% للنص: '{word}')")
                        
                        # جلب معلومات الدواء
                        drug_info = get_drug_info_from_api(match)
                        display_drug_info(match, drug_info)
                else:
                    st.error("لم يتم التعرف على أي دواء في الصورة")
                    
                    # عرض النصوص المستخرجة للمساعدة في التصحيح
                    if cleaned_texts:
                        st.write("النصوص المستخرجة:", ", ".join(cleaned_texts))
            
            except Exception as e:
                st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")

if __name__ == "__main__":
    main()