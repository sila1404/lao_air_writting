from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Query,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import cv2
import logging
import asyncio
import time
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Environment Variables for MongoDB ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "laoAirWritingDB")
MONGO_FEEDBACK_COLLECTION = os.getenv("MONGO_FEEDBACK_COLLECTION", "feedback")

# Global variables for model instances and DB client
ocr_processor = None
db_client: Optional[MongoClient] = None
db = None

# Constants
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FORMATS = ["image/jpeg", "image/png", "image/bmp", "image/tiff"]


# --- Pydantic Models ---
class OCRResult(BaseModel):
    text: str
    bounding_boxes: Optional[List[dict]] = None
    confidence_scores: List[float]
    has_content: bool


class PredictResponse(BaseModel):
    success: bool
    result: OCRResult
    processing_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    ocr_processor_loaded: bool
    mongodb_connected: bool
    version: str


class FeedbackData(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    email: Optional[EmailStr] = Field(None, max_length=100)
    rating: Optional[int] = Field(None, ge=1, le=5)
    category: Optional[str] = Field(None, max_length=50)
    comments: str = Field(..., min_length=10, max_length=2000)


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models and DB connection
    global ocr_processor, db_client, db

    # --- OCR Model Loading ---
    async def load_model_background():
        global ocr_processor
        try:
            logger.info("Importing OCR components...")
            from utils import OCRProcessor

            logger.info("Initializing OCR model...")
            ocr_processor_instance = await asyncio.to_thread(OCRProcessor)
            ocr_processor = ocr_processor_instance
            logger.info("OCR Model initialized successfully")
        except ImportError:
            logger.error("Failed to import OCRProcessor from utils.")
            ocr_processor = None
        except Exception as e:
            logger.error(f"Error initializing OCR model: {e}")
            ocr_processor = None

    asyncio.create_task(load_model_background())

    # --- MongoDB Connection ---
    logger.info("Attempting to connect to MongoDB...")
    try:
        db_client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
        # Send a ping to confirm a successful connection
        db_client.admin.command("ping")
        db = db_client[MONGO_DB_NAME]
        logger.info("Successfully connected to MongoDB!")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        db_client = None
        db = None

    yield  # Server is running

    # Cleanup: Release resources
    logger.info("Shutting down and releasing resources...")
    ocr_processor = None

    if db_client:
        logger.info("Closing MongoDB connection...")
        db_client.close()
        logger.info("MongoDB connection closed.")


# Pass the lifespan context manager to FastAPI
app = FastAPI(
    title="Lao Air Writing OCR & Feedback API",
    description="API for Lao air writing optical character recognition and user feedback.",
    version="0.0.3",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def validate_image(file: UploadFile) -> bool:
    if file.size and file.size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size allowed is {MAX_IMAGE_SIZE / (1024 * 1024)}MB",
        )
    if file.content_type not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_FORMATS)}",
        )
    return True


def log_request_details(
    image_name: Optional[str], prediction_text: str, processing_time: float
):
    log_image_name = image_name if image_name else "N/A"
    logger.info(
        f"Processed image '{log_image_name}' in {processing_time:.4f}s. Prediction: {prediction_text[:100]}..."
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict_character(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    return_bbox: bool = Query(False, description="Whether to return bounding boxes"),
    min_confidence: float = Query(
        0.1, description="Minimum confidence threshold for OCR", ge=0.0, le=1.0
    ),
):
    start_time = time.time()

    if ocr_processor is None:
        logger.error("OCR model not available for /api/predict endpoint.")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable: OCR model is not initialized. Please try again shortly.",
        )

    validate_image(image)

    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning(f"Could not decode image: {image.filename}")
            raise HTTPException(
                status_code=400, detail="Invalid image format or corrupted file."
            )

        # Run synchronous OCR processing in a separate thread
        recognition_result = await asyncio.to_thread(
            ocr_processor.recognize_text,
            img,
            return_bbox=return_bbox,
            min_confidence=min_confidence,
        )

        if return_bbox:
            text, bboxes, confidence_scores, has_content = recognition_result
            bboxes_serializable = [
                {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                for x, y, w, h in bboxes
            ]
            result_data = {
                "text": text,
                "bounding_boxes": bboxes_serializable,
                "confidence_scores": [float(c) for c in confidence_scores],
                "has_content": has_content,
            }
        else:
            text, confidence_scores, has_content = recognition_result
            result_data = {
                "text": text,
                "confidence_scores": [float(c) for c in confidence_scores],
                "has_content": has_content,
            }

        ocr_result_obj = OCRResult(**result_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Critical error processing image {image.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during image processing.",
        )
    finally:
        await image.close()

    processing_time = time.time() - start_time
    background_tasks.add_task(
        log_request_details, image.filename, ocr_result_obj.text, processing_time
    )

    return PredictResponse(
        success=True,
        result=ocr_result_obj,
        processing_time_seconds=round(processing_time, 4),
    )


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackData = Body(...)):
    global db  # Access the global db instance
    if db is None or db_client is None:  # Check if MongoDB connection was successful
        logger.error("Failed to submit feedback: MongoDB is not connected.")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable: Cannot store feedback at the moment.",
        )

    try:
        feedback_dict = feedback.model_dump(exclude_none=True)
        feedback_dict["submitted_at"] = datetime.now(
            timezone.utc
        )  # Add a UTC timestamp

        # Get the feedback collection
        feedback_collection = db[MONGO_FEEDBACK_COLLECTION]

        # Insert the feedback document
        insert_result = await asyncio.to_thread(
            feedback_collection.insert_one, feedback_dict
        )

        feedback_id = str(insert_result.inserted_id)
        logger.info(f"Feedback stored successfully with ID: {feedback_id}")

        return FeedbackResponse(
            success=True,
            message="Feedback submitted successfully! Thank you.",
            feedback_id=feedback_id,
        )
    except pymongo.errors.ConnectionFailure as e:
        logger.error(f"MongoDB ConnectionFailure while storing feedback: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database connection error. Could not store feedback.",
        )
    except Exception as e:
        logger.exception(f"Failed to store feedback in MongoDB: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not store feedback due to an internal server error.",
        )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    global db_client
    ocr_is_ready = ocr_processor is not None
    mongodb_is_connected = False
    if db_client:
        try:
            db_client.admin.command("ping")
            mongodb_is_connected = True
        except pymongo.errors.ConnectionFailure:
            logger.warning("Health check: MongoDB connection ping failed.")
            mongodb_is_connected = False
        except Exception as e:
            logger.warning(f"Health check: MongoDB status check error: {e}")
            mongodb_is_connected = False

    current_status = "healthy"
    if not ocr_is_ready:
        current_status = "degraded"
        logger.warning("Health check: OCR processor not loaded.")
    if not mongodb_is_connected:
        current_status = (
            "degraded" if current_status == "healthy" else current_status
        )  # Keep degraded if already set
        logger.warning("Health check: MongoDB not connected.")

    return HealthResponse(
        status=current_status,
        ocr_processor_loaded=ocr_is_ready,
        mongodb_connected=mongodb_is_connected,
        version=app.version,
    )
