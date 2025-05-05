from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import cv2
import logging
from utils import OCRProcessor
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables for model instances
ocr_processor = None

# Constants
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FORMATS = ["image/jpeg", "image/png", "image/bmp", "image/tiff"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models when the application starts
    global ocr_processor
    try:
        logger.info("Initializing OCR model...")
        ocr_processor = OCRProcessor()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        # Even though initialization failed, we still start the app
        # The health endpoint will report the failure

    yield  # Server is running and handling requests

    # Cleanup: Release resources when the application is shutting down
    logger.info("Shutting down and releasing resources...")
    ocr_processor = None


# Pass the lifespan context manager to FastAPI
app = FastAPI(
    title="OCR API",
    description="API for optical character recognition",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image(file: UploadFile) -> bool:
    # Check file size
    if file.size and file.size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size allowed is {MAX_IMAGE_SIZE / (1024 * 1024)}MB",
        )

    # Check content type
    if file.content_type not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_FORMATS)}",
        )

    return True


def log_request(image_name: str, prediction: str, processing_time: float):
    """Log request details for monitoring purposes"""
    logger.info(
        f"Processed image '{image_name}' in {processing_time:.2f}s with result: {prediction[:50]}..."
    )


@app.post("/predict")
async def predict_character(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    return_bbox: bool = Query(False, description="Whether to return bounding boxes"),
    min_confidence: float = Query(
        0.1, description="Minimum confidence threshold", ge=0.0, le=1.0
    ),
):
    start_time = time.time()

    # Check if model is loaded
    if not ocr_processor:
        raise HTTPException(
            status_code=503,
            detail="OCR model not initialized. Please check health endpoint for status.",
        )

    # Validate image
    try:
        validate_image(image)
    except HTTPException as e:
        logger.warning(f"Validation error: {e.detail}")
        raise

    try:
        # Read the image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check if image was successfully decoded
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please ensure it's a valid image file.",
            )

        # Process the image with proper return value handling
        if return_bbox:
            text, bboxes, confidence_scores, has_content = ocr_processor.recognize_text(
                img, return_bbox=True, min_confidence=min_confidence
            )
            # Convert bounding boxes to a serializable format
            bboxes_serializable = [
                {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                for x, y, w, h in bboxes
            ]
            result = {
                "text": text,
                "bounding_boxes": bboxes_serializable,
                "confidence_scores": [float(c) for c in confidence_scores],
                "has_content": has_content,
            }
        else:
            text, confidence_scores, has_content = ocr_processor.recognize_text(
                img, return_bbox=False, min_confidence=min_confidence
            )
            result = {
                "text": text,
                "confidence_scores": [float(c) for c in confidence_scores],
                "has_content": has_content,
            }

        # Calculate processing time
        processing_time = time.time() - start_time

        # Log request details in the background
        background_tasks.add_task(log_request, image.filename, text, processing_time)

        return {
            "success": True,
            "result": result,
            "processing_time_seconds": processing_time,
        }

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Generic error to avoid exposing system details
        raise HTTPException(
            status_code=500,
            detail="Error processing image. Please try again or contact support if the issue persists.",
        )
    finally:
        # Reset file pointer in case we need to read it again
        await image.seek(0)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    status = "healthy" if ocr_processor is not None else "degraded"

    return {
        "status": status,
        "ocr_processor_loaded": ocr_processor is not None,
        "version": "1.0.0",
    }
