import os
import tempfile
import json
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Load environment variables from the .env file
load_dotenv()


class MedPrescriptionAssistant:
    """
    A helper class for interacting with Google's Generative AI model
    to analyze prescription images and extract relevant medical details.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the assistant with a specific model.

        Args:
            model_name (str): The name of the Gemini model to use.
        Raises:
            ValueError: If the API key is missing in environment variables.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key is missing. Please set GEMINI_API_KEY in your .env file.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def get_response(self, file_path: str, prompt: str) -> str:
        """
        Upload a file and send it to the model along with a prompt.

        Args:
            file_path (str): Path to the file to be uploaded.
            prompt (str): Instructional text for the model.

        Returns:
            str: The generated response text from the model.
        """
        uploaded_file = genai.upload_file(file_path)
        try:
            response = self.model.generate_content([uploaded_file, prompt])
            return response.text.strip()
        finally:
            # Attempt to clean up the uploaded file on Google's side
            try:
                genai.delete_file(uploaded_file.name)
            except Exception as delete_err:
                print("Failed to delete file:", delete_err)


# Initialize FastAPI application
app = FastAPI(title="Gemini Prescription API")
assistant = MedPrescriptionAssistant(model_name="gemini-2.5-flash")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analyze a prescription image and extract structured medical details.

    Args:
        file (UploadFile): An uploaded image file of the prescription.

    Returns:
        JSONResponse: A response containing extracted details in JSON format,
                      or an error message if the response is invalid.
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        # Instructional prompt for the model
        prompt_text = (
            "You are a medical prescription analyzer. "
            "Analyze the uploaded prescription image and extract medicine details. "
            "If the image only shows the drug name, use your medical knowledge to fill in the other fields. "
            "Return ONLY a valid JSON object in this exact format:\n\n"
            "{\n"
            '  "drug_name": "string or null",\n'
            '  "dosage": "string or null",\n'
            '  "frequency": "string or null",\n'
            '  "instructions": "string or null",\n'
            '  "contraindications": ["list of conditions or empty list"],\n'
            '  "side_effects": ["list or empty list"],\n'
            '  "substitutes": ["list or empty list"],\n'
            '  "therapeutic_class": "string or null",\n'
            '  "chemical_class": "string or null",\n'
            '  "habit_forming": "Yes/No or null",\n'
            '  "warnings": ["list of warnings or empty list"]\n'
            "}\n\n"
            "IMPORTANT: Return ONLY raw JSON (no markdown, no explanation, no code block)."
        )

        # Get model output
        raw_result = assistant.get_response(tmp_path, prompt_text)

        # Clean response if wrapped with code block markers
        cleaned = raw_result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        # Validate the JSON structure
        try:
            parsed = json.loads(cleaned)
        except Exception:
            return JSONResponse(
                {"error": "Invalid JSON returned from model", "raw_response": raw_result},
                status_code=500
            )

        return JSONResponse({"response": parsed})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        # Ensure temporary file is always removed
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_fast:app", host="0.0.0.0", port=8000, reload=True)
