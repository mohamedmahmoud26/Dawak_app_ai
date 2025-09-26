"""
FastAPI deployment for Medicine Detection, OCR, and Drug Matching.
This service detects text from prescription images, applies OCR, and 
matches the extracted text against a structured drug dataset.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import sys
import pandas as pd
from ultralytics import YOLO
import easyocr
from rapidfuzz import process, fuzz

# Configure paths for the "scr" module and project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Current "api" folder
ROOT_DIR = os.path.dirname(BASE_DIR)                  # Project root
SCR_DIR = os.path.join(ROOT_DIR, "scr")

for path in [SCR_DIR, ROOT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from scr.text_extraction import extract_text_with_yolo, clean_extracted_texts

# Load YOLO detection model
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best.pt")
det_model = YOLO(MODEL_PATH)

# Initialize OCR reader (supports English and Arabic)
ocr_reader = easyocr.Reader(["en", "ar"])

# Load drug dataset (CSV)
CSV_PATH = os.path.join(ROOT_DIR, "dataset", "durg.csv")
data = pd.read_csv(CSV_PATH, on_bad_lines='skip')  # Skip problematic rows
data = data.fillna("")  # Replace NaN values with empty strings

# Convert dataset to dictionary format for easier lookup
DRUG_DICTIONARY = data.to_dict(orient="records")

# Threshold for drug name matching accuracy
MATCH_THRESHOLD = 60


def match_drug_names(cleaned_texts, dictionary=DRUG_DICTIONARY, threshold=MATCH_THRESHOLD):
    """
    Match OCR-extracted texts against the drug dictionary.

    Args:
        cleaned_texts (list[str]): List of cleaned OCR text strings.
        dictionary (list[dict]): List of drug records (converted from CSV).
        threshold (int): Minimum similarity score required for a match.

    Returns:
        list[dict]: List of matched drug records with details.
    """
    if not dictionary:
        return []

    # Build a lookup dictionary: lowercase name → drug record
    drug_lookup = {}
    for drug in dictionary:
        if "drug_name" in drug and drug["drug_name"]:
            drug_lookup[drug["drug_name"].lower()] = drug

        # Include substitute names if available
        for k, v in drug.items():
            if k.lower().startswith("substitute") and v:
                drug_lookup[v.lower()] = drug

    search_space = list(drug_lookup.keys())

    # Perform fuzzy matching on OCR results
    matches = []
    for word in cleaned_texts:
        word_lower = word.lower().strip()
        result = process.extractOne(word_lower, search_space, scorer=fuzz.token_sort_ratio)
        if result:
            match_name, score, _ = result
            if score >= threshold:
                drug_info = drug_lookup[match_name]
                matches.append({
                    "extracted_word": word,
                    "matched_name": match_name,
                    "score": score,
                    "details": map_to_final_schema(drug_info)
                })

    # Remove duplicates (keep the first occurrence)
    unique_matches = []
    seen = set()
    for m in matches:
        if m["matched_name"] not in seen:
            unique_matches.append(m)
            seen.add(m["matched_name"])

    return unique_matches


def map_to_final_schema(details):
    """
    Map a raw drug dictionary row to the standardized JSON schema.

    Args:
        details (dict): A single drug record from the dataset.

    Returns:
        dict: Drug details structured in the final response schema.
    """
    return {
        "drug_name": details.get("drug_name", None),
        "dosage": details.get("dosage", None),
        "frequency": details.get("frequency", None),
        "use": details.get("use", None),
        "side_effects": details.get("side_effects", None),
        "substitutes": details.get("substitutes", None),
        "chemical_class": details.get("chemical_class", None),
        "therapeutic_class": details.get("therapeutic_class", None),
        "habit_forming": details.get("habit_forming", None),
        "warnings": details.get("warnings", None)
    }


# Initialize FastAPI application
app = FastAPI(title="Medicine Detection + OCR + Matcher")


@app.post("/predict_medicine")
async def predict_medicine(file: UploadFile = File(...)):
    """
    Predict medicines from a prescription image.

    Steps:
        1. Decode the uploaded image.
        2. Detect text using YOLO + OCR.
        3. Clean and normalize extracted texts.
        4. Match texts against drug dictionary.
        5. Return structured results in JSON format.

    Args:
        file (UploadFile): Uploaded prescription image.

    Returns:
        JSONResponse: OCR text results and matched drug information.
    """
    contents = await file.read()

    # Decode image into OpenCV format
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Extract texts using YOLO detection and OCR
    texts = extract_text_with_yolo(det_model, ocr_reader, img, conf_threshold=0.5)
    cleaned_texts = clean_extracted_texts(texts)

    # Match extracted texts against drug dataset
    matches = match_drug_names(cleaned_texts)

    # Return results as JSON
    return JSONResponse(content={
        "ocr_texts": cleaned_texts,
        "matches": matches
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Deploy_fastapi:app", host="0.0.0.0", port=8000, reload=True)
