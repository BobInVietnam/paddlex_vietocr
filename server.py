import os
import shutil
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Import your custom class from model.py
from ocrmodel import OCRPipeline

# Initialize FastAPI
app = FastAPI(
    title="PaddleX x VietOCR API",
    description="For Document Unwarping, Detection, and Vietnamese Recognition. For use with my Dyslexia app.",
    version="1.0.0"
)

# Global OCR Pipeline instance
# This ensures models are loaded into VRAM only once at startup
print("Starting server and loading models...")
ocr_model = OCRPipeline(debug=False)

# Define the response structure
class OCRResponse(BaseModel):
    filename: str
    text_lines: List[str]
    count: int

@app.post("/predict", response_model=OCRResponse)
async def predict_ocr(file: UploadFile = File(...)):
    """
    Receives an image file, saves it temporarily, runs the OCR pipeline, 
    and returns the extracted text lines.
    """
    # 1. Basic validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # 2. Create a unique temporary filename to prevent collisions between users
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    temp_path = os.path.join(temp_dir, unique_filename)

    try:
        # 3. Save the uploaded file to disk
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 4. Run the OCR Pipeline
        # Note: model.py handles unwarping, detection, and recognition
        extracted_text = ocr_model.predict(temp_path)

        # 5. Return the result
        return OCRResponse(
            filename=file.filename,
            text_lines=extracted_text,
            count=len(extracted_text)
        )

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR Processing failed: {str(e)}")

    finally:
        # 6. Cleanup: Always remove the original uploaded file from the temp folder
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
async def root():
    """
    Root endpoint providing links to the documentation.
    """
    return {
        "message": "Vietnamese OCR Server is active.",
        "documentation": {
            "SwaggerUI": "/docs",
            "ReDoc": "/redoc"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Run server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)