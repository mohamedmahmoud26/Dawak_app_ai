"""
API Handler Module
------------------
This module is responsible for interacting with external APIs, specifically
the openFDA API, to fetch structured drug information.
"""

import requests
from config import FDA_API_URL


def get_drug_info_from_api(drug_name: str):
    """
    Query the openFDA API for information about a specific drug.

    Args:
        drug_name (str): The generic name of the drug to search for.

    Returns:
        dict | None: A dictionary containing drug details if found, otherwise None.
            The dictionary includes:
                - brand_name (str): Commercial brand name of the drug.
                - generic_name (str): Generic name of the drug.
                - manufacturer (str): Manufacturer of the drug.
                - purpose (str): Indicated purpose or use of the drug.
                - warnings (str): Associated warnings.
                - dosage (str): Dosage and administration instructions.
    """
    try:
        # Build the API request URL with search query and limit
        url = f"{FDA_API_URL}?search=generic_name:{drug_name}&limit=1"
        response = requests.get(url, timeout=10)
        data = response.json()

        # Verify that results are returned and extract the first entry
        if "results" in data:
            drug_info = data["results"][0]
            return {
                "brand_name": drug_info.get("openfda", {}).get("brand_name", ["Not Available"])[0],
                "generic_name": drug_info.get("openfda", {}).get("generic_name", ["Not Available"])[0],
                "manufacturer": drug_info.get("openfda", {}).get("manufacturer_name", ["Not Available"])[0],
                "purpose": drug_info.get("purpose", ["Not Available"])[0],
                "warnings": drug_info.get("warnings", ["Not Available"])[0],
                "dosage": drug_info.get("dosage_and_administration", ["Not Available"])[0],
            }

    except Exception as e:
        # Log error to console; can be replaced with logging in production
        print(f"Error fetching drug info: {e}")

    return None
