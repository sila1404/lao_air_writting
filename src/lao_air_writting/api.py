from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Query,
    Body,
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import cv2
import logging
import io
import asyncio
import time
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from datetime import datetime, timezone
from pyngrok import ngrok
from transformers import AutoTokenizer, VitsModel
import torch
import scipy.io.wavfile
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
tts_models = {}
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


class TTSRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=500, examples=["ສະບາຍດີ."]
    )


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
    global ocr_processor, tts_models, db_client, db

    # --- NGROK SETUP ---
    # Set auth token from environment variable
    ngrok_auth_token = os.getenv("NGROK_TOKEN")
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)

    # Start the ngrok tunnel
    # The port (8000) should match the port uvicorn is running on
    public_url = ngrok.connect(8000).public_url
    logger.info(f"Ngrok tunnel established at: {public_url}")
    app.state.public_url = public_url
    # -------------------

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

    # --- Loading TTS model ---
    logger.info("Application startup: Loading TTS model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_models["tokenizer"] = AutoTokenizer.from_pretrained("facebook/mms-tts-lao")
        tts_models["model"] = VitsModel.from_pretrained("facebook/mms-tts-lao").to(
            device
        )
        tts_models["device"] = device
        logger.info(f"Model loaded successfully on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")

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

    logger.info("TTS resources cleared.")
    tts_models.clear()

    if db_client:
        logger.info("Closing MongoDB connection...")
        db_client.close()
        logger.info("MongoDB connection closed.")

    ngrok.disconnect(public_url)
    logger.info("Ngrok tunnel disconnected.")


# Pass the lifespan context manager to FastAPI
app = FastAPI(
    title="Lao Air Writing OCR & Text-to-Speech API",
    description="API for Lao air writing optical character recognition and Lao text to speech.",
    version="0.1.0",
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


@app.post("/api/tts", response_class=StreamingResponse)
async def generate_tts(request: TTSRequest):
    """
    Accepts Lao text and returns the generated audio in WAV format.
    """
    logger.info(f"Received TTS request for text: '{request.text[:30]}...'")

    if "model" not in tts_models or "tokenizer" not in tts_models:
        logger.error("Model is not loaded. Cannot process request.")
        raise HTTPException(
            status_code=503, detail="Model is not available. Please check server logs."
        )

    try:
        # --- Tokenization ---
        inputs = tts_models["tokenizer"](request.text, return_tensors="pt").to(
            tts_models["device"]
        )

        # Define the synchronous inference function
        def run_inference():
            with torch.no_grad():
                # This is the blocking part that needs to be in a separate thread
                return tts_models["model"](**inputs).waveform

        # --- Run synchronous inference in a separate thread ---
        output = await asyncio.to_thread(run_inference)

        # --- Prepare Audio Data ---
        sampling_rate = tts_models["model"].config.sampling_rate
        audio_numpy = output.squeeze().cpu().numpy()

        # --- AMPLIFY AND NORMALIZE AUDIO ---
        max_val = np.max(np.abs(audio_numpy))
        if max_val > 0:
            # Normalize to the range [-1.0, 1.0]
            normalized_audio = audio_numpy / max_val
            # Scale to 16-bit integer range and convert type
            # We scale to 95% of max value to leave a little headroom and prevent clipping
            audio_amplified = np.int16(normalized_audio * 32767 * 0.95)
        else:
            # The audio is silent, just ensure it's the correct type
            audio_amplified = audio_numpy.astype(np.int16)

        logger.info(
            f"Successfully generated and amplified audio waveform of length {len(audio_amplified)}."
        )

        # --- Save to In-Memory Buffer ---
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, rate=sampling_rate, data=audio_amplified)
        buffer.seek(0)

        # --- Return Streaming Response ---
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        logger.error(f"An error occurred during TTS generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate audio.")


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
