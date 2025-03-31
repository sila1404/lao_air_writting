from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import cv2
from utils import CharacterRecognitionModel, OCRProcessor

# Global variables for model instances
model = None
ocr_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models when the application starts
    global model, ocr_processor
    try:
        model = CharacterRecognitionModel()
        model.load_model()
        ocr_processor = OCRProcessor()
        print("Models initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {e}")

    yield  # Server is running and handling requests

    # Cleanup: Release resources when the application is shutting down
    model = None
    ocr_processor = None


# Pass the lifespan context manager to FastAPI
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_character(image: UploadFile = File(...)):
    if not model or not ocr_processor:
        raise HTTPException(
            status_code=503, detail="Model not initialized. Please try again later."
        )

    # Read the image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image using your existing OCR processor
    predicted_text = ocr_processor.recognize_text(img)

    return {"prediction": predicted_text}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "ocr_processor_loaded": ocr_processor is not None,
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", reload=True)