from rapidfuzz import process, fuzz
from config import DRUG_DICTIONARY, MATCH_THRESHOLD


def match_drug_names(cleaned_texts, dictionary=DRUG_DICTIONARY, threshold=MATCH_THRESHOLD):
    """
    Match OCR-extracted text tokens against a drug dictionary.

    This function attempts to find approximate string matches between words
    extracted via OCR and a reference drug dictionary (loaded from a CSV file).

    Args:
        cleaned_texts (list[str]): List of OCR-extracted text tokens to match.
        dictionary (list[dict], optional): Drug dictionary, where each entry is a dict
            representing a drug and its details. Defaults to DRUG_DICTIONARY.
        threshold (int, optional): Minimum similarity score required to accept a match.
            Defaults to MATCH_THRESHOLD.

    Returns:
        list[dict]: A list of match results, where each element contains:
            - extracted_word (str): The original OCR-extracted word.
            - matched_name (str): The best-matching drug name or substitute.
            - score (float): The similarity score between extracted word and matched name.
            - details (dict): Full drug row (all details) from the dictionary.
    """
    if not dictionary:
        return []

    # --- Step 1: Build a lookup table for searchable names ---
    # Each drug name and its substitutes are mapped (in lowercase) to the original drug row
    drug_lookup = {}
    for drug in dictionary:
        if "drug_name" in drug and drug["drug_name"]:
            drug_lookup[drug["drug_name"].lower()] = drug

        # Include substitutes (if any). Some may be comma-separated.
        substitutes = drug.get("substitutes")
        if substitutes:
            for sub in str(substitutes).split(","):
                sub = sub.strip().lower()
                if sub:
                    drug_lookup[sub] = drug

    search_space = list(drug_lookup.keys())

    # --- Step 2: Match OCR-extracted words against the search space ---
    matches = []
    for word in cleaned_texts:
        word_lower = word.lower().strip()

        # Use token_sort_ratio to handle spacing and word order variations
        result = process.extractOne(word_lower, search_space, scorer=fuzz.token_sort_ratio)

        if result:
            match_name, score, _ = result
            if score >= threshold:
                drug_info = drug_lookup[match_name]
                matches.append({
                    "extracted_word": word,
                    "matched_name": match_name,
                    "score": score,
                    "details": drug_info
                })

    # --- Step 3: Deduplicate results by matched_name ---
    unique_matches = []
    seen = set()
    for match in matches:
        if match["matched_name"] not in seen:
            unique_matches.append(match)
            seen.add(match["matched_name"])

    return unique_matches
