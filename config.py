"""
Application Configuration Settings
"""

import os
import pandas as pd
from rapidfuzz import process, fuzz

# ===== Base Directory =====
BASE_DIR = "/mnt/c/Users/Mohamed Mahmoud/Dawak_vect"

# ===== Model Path =====
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# ===== OCR Settings =====
OCR_LANGUAGES = ['en', 'ar']  # Supported OCR languages

# ===== Matching Settings =====
MATCH_THRESHOLD = 70  # Minimum score threshold for text matching

# ===== CSV File Path =====
CSV_DRUG_PATH = os.path.join(BASE_DIR, "dataset", "durg.csv")  # Path to the drug dataset CSV

# ===== Load Drug Dictionary =====
DRUG_DICTIONARY = []

try:
    # Read CSV file into a DataFrame
    data = pd.read_csv(CSV_DRUG_PATH, low_memory=False)

    # Clean drug names (main column: drug_name)
    data['drug_name_lower'] = data['drug_name'].astype(str).str.lower().str.strip()

    # Convert all rows into a list of dictionaries
    DRUG_DICTIONARY = data.fillna(value=pd.NA).to_dict(orient="records")
    print(f"Loaded {len(DRUG_DICTIONARY)} drugs from CSV.")
except Exception as e:
    print(f"Error reading CSV file: {e}")


# ===== Function to Get Drug Info Using RapidFuzz =====
def get_drug_info(name: str):
    """
    Retrieve complete drug information (all columns) for the closest match.

    Args:
        name (str): Drug name to search for.

    Returns:
        dict | None: Dictionary with drug details and match score, or None if not found.
    """
    if not DRUG_DICTIONARY:
        return None

    # Normalize input name
    name_clean = name.lower().strip()
    drug_names = [d['drug_name_lower'] for d in DRUG_DICTIONARY]

    # Perform fuzzy matching
    result = process.extractOne(name_clean, drug_names, scorer=fuzz.ratio)
    if result:
        match_name, score, _ = result
        if score >= MATCH_THRESHOLD:
            # Retrieve the full record for the matched drug
            drug_info = next((d for d in DRUG_DICTIONARY if d['drug_name_lower'] == match_name), None)
            if drug_info:
                # Include the score in the returned dictionary
                return {"score": score, **drug_info}
    return None
