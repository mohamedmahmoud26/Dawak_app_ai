# -*- coding: utf-8 -*-
"""
مطابقة الأدوية - Drug Matching Module
"""
from rapidfuzz import process, fuzz
from config import DRUG_DICTIONARY, MATCH_THRESHOLD

def match_drug_names(cleaned_texts, dictionary=DRUG_DICTIONARY, threshold=MATCH_THRESHOLD):
    """
    مطابقة النصوص مع قاموس الأدوية - Match texts with drug dictionary
    """
    matches = []
    for word in cleaned_texts:
        if ' ' in word:  # إذا كانت الكلمة مركبة
            for subword in word.split():
                match, score, _ = process.extractOne(subword, dictionary, scorer=fuzz.ratio)
                if score >= threshold:
                    matches.append((subword, match, score))
        else:
            match, score, _ = process.extractOne(word, dictionary, scorer=fuzz.ratio)
            if score >= threshold:
                matches.append((word, match, score))
    
    # إزالة التكرارات
    unique_matches = []
    seen = set()
    for match in matches:
        if match[1] not in seen:
            unique_matches.append(match)
            seen.add(match[1])
    
    return unique_matches