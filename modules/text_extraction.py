# -*- coding: utf-8 -*-
"""
استخراج النصوص - Text Extraction Module
"""
import cv2
import easyocr
import re
import os
from modules.image_processing import save_image_temp

def extract_text_with_yolo(model, reader, image, conf_threshold=0.5):
    """
    استخراج النصوص باستخدام YOLO - Extract text using YOLO
    """
    texts = []
    
    # حفظ الصورة مؤقتاً للتنبؤ
    temp_path = save_image_temp(image)
    
    try:
        # التنبؤ باستخدام YOLO (باستخدام مسار الملف)
        results = model.predict(temp_path, verbose=False)
        
        for r in results:
            for box in r.boxes:
                # التحقق من مستوى الثقة
                conf = box.conf[0].cpu().numpy()
                
                if conf >= conf_threshold:
                    # إحداثيات المربع المحيط
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # قص النص من الصورة
                    crop = image[y1:y2, x1:x2]
                    
                    # استخراج النص باستخدام OCR
                    result = reader.readtext(crop, detail=0)
                    texts.extend(result)
    finally:
        # تنظيف الملف المؤقت
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return texts

def clean_extracted_texts(texts):
    """
    تنظيف النصوص المستخرجة - Clean extracted texts
    """
    cleaned = []
    for t in texts:
        t = t.lower()  # تحويل إلى أحرف صغيرة
        t = re.sub(r'[^a-z\s]', '', t)  # إزالة الرموز غير المرغوبة
        t = t.strip()  # إزالة المسافات الزائدة
        if t and len(t) > 2:  # تجاهل النصوص القصيرة جدًا
            cleaned.append(t)
    return cleaned