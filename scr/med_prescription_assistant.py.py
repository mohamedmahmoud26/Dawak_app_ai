import google.generativeai as genai
import os

class MedPrescriptionAssistant:
    def __init__(self, model_name="gemini-2.5-flash"):
        api_key = "AIzaSyAOETwiYrKecKX5u0iqmMGhPPSJMIrk0sk" # Replace with your actual API key
        if not api_key:
             raise ValueError("API key is empty. Please provide your actual Gemini API key.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def upload_file(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return genai.upload_file(file_path)

    def get_response(self, file_path, prompt="What is written in the prescription"):
        uploaded_file = self.upload_file(file_path)
        print(" Uploaded file ID:", uploaded_file.name)

        try:
            response = self.model.generate_content([
                {"file_data": uploaded_file},
                prompt
            ])
            return response.text
        finally:
            try:
                genai.delete_file(uploaded_file.name)
                print("Deleted uploaded file:", uploaded_file.name)
            except Exception as delete_err:
                print("Failed to delete file:", delete_err)

image_file_path = "C:\\Users\\Mohamed Mahmoud\\Downloads\\Panadol_advance_48.jpg"
prompt_text = "What is written in the prescription? Provide the response in Arabic and include all information about the medicine drug, what it is for, how to use it, when it cannot be taken, side effects, and any other relevant details shown in the image."
model_name = "gemini-2.5-flash"

try:
    assistant = MedPrescriptionAssistant(model_name=model_name)

    response = assistant.get_response(image_file_path, prompt=prompt_text)

    print("\n Response:\n")
    print(response)

except FileNotFoundError as e:
    print(f"\n Error: {e}")
except ValueError as e:
    print(f"\n Error: {e}")
except Exception as e:
    print(f"\n An unexpected error occurred: {e}")

