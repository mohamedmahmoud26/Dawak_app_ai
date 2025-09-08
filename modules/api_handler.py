# -*- coding: utf-8 -*-
"""
التعامل مع APIs - API Handler Module
"""
import requests
from config import FDA_API_URL

def get_drug_info_from_api(drug_name):
    """
    جلب معلومات الدواء من API - Get drug information from API
    """
    try:
        # البحث في openFDA
        url = f"{FDA_API_URL}?search=generic_name:{drug_name}&limit=1"
        response = requests.get(url, timeout=10)
        data = response.json()

        if 'results' in data:
            drug_info = data['results'][0]
            return {
                'brand_name': drug_info.get('openfda', {}).get('brand_name', ['غير متوفر'])[0],
                'generic_name': drug_info.get('openfda', {}).get('generic_name', ['غير متوفر'])[0],
                'manufacturer': drug_info.get('openfda', {}).get('manufacturer_name', ['غير متوفر'])[0],
                'purpose': drug_info.get('purpose', ['غير متوفر'])[0],
                'warnings': drug_info.get('warnings', ['غير متوفر'])[0],
                'dosage': drug_info.get('dosage_and_administration', ['غير متوفر'])[0]
            }
    except Exception as e:
        print(f"Error fetching drug info: {e}")
    
    return None