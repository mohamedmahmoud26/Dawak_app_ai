from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# استدعاء كود الـ OCR بتاعك
from modules.text_extraction import extract_text_with_yolo, clean_extracted_texts

# تحميل موديل الـ Object Detection
det_model = YOLO("models/best.pt")

# تحميل OCR reader (إنجليزي + عربي)
ocr_reader = easyocr.Reader(["en", "ar"])

app = FastAPI(title="Medicine Detection + OCR API")

@app.post("/predict_medicine")
async def predict_medicine(file: UploadFile = File(...)):
    # تحويل الصورة من bytes لـ OpenCV
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # تشغيل الـ OCR مع YOLO
    texts = extract_text_with_yolo(det_model, ocr_reader, img, conf_threshold=0.5)
    cleaned_texts = clean_extracted_texts(texts)

    return JSONResponse(content={"texts": cleaned_texts})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
