# -*- coding: utf-8 -*-
"""
إعدادات التطبيق - Configuration Settings
"""

# مسارات النماذج
MODEL_PATH = "C:/Users/digital/Downloads/best.pt"

# إعدادات OCR
OCR_LANGUAGES = ['en']

# إعدادات المطابقة
MATCH_THRESHOLD = 70
CONFIDENCE_THRESHOLD = 0.5

# APIs
FDA_API_URL = "https://api.fda.gov/drug/label.json"

# قاموس الأدوية
DRUG_DICTIONARY = [
    "paracetamol", "ibuprofen", "aspirin", "amoxicillin", 
    "omeprazole", "metformin", "panadol", "voltaren", 
    "augmentin", "zantac", "lipitor", "nexium"
]