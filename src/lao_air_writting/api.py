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
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker, Session
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

# --- Environment Variables for the feedback database ---
# Defaults to a local SQLite file. Set DATABASE_URL to point at Postgres
# instead, e.g. postgresql+psycopg2://user:password@host:5432/dbname
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./lao_air_writing.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")
FEEDBACK_TABLE_NAME = os.getenv("FEEDBACK_TABLE_NAME", "feedback")

Base = declarative_base()


class FeedbackRecord(Base):
    __tablename__ = FEEDBACK_TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=True)
    email = Column(String(100), nullable=True)
    rating = Column(Integer, nullable=True)
    category = Column(String(50), nullable=True)
    comments = Column(Text, nullable=False)
    submitted_at = Column(DateTime(timezone=True), nullable=False)


# Global variables for model instances and DB engine/session factory
ocr_processor = None
db_engine: Optional[Engine] = None
db_session_factory: Optional[sessionmaker] = None

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
    database_connected: bool
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


class FeedbackItem(BaseModel):
    id: int
    name: Optional[str] = None
    email: Optional[str] = None
    rating: Optional[int] = None
    category: Optional[str] = None
    comments: str
    submitted_at: datetime

    model_config = {"from_attributes": True}


class FeedbackListResponse(BaseModel):
    count: int
    items: List[FeedbackItem]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models and DB connection
    global ocr_processor, db_engine, db_session_factory

    # --- OCR Model Loading ---
    async def load_model_background():
        global ocr_processor
        try:
            logger.info("Importing OCR components...")
            from utils.ocr import OCRProcessor

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

    # --- Database Connection (SQLite by default, Postgres if DATABASE_URL is set) ---
    safe_db_url = make_url(DATABASE_URL).render_as_string(hide_password=True)
    logger.info(f"Attempting to connect to database: {safe_db_url}")
    try:
        connect_args = (
            {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
        )
        engine = create_engine(DATABASE_URL, connect_args=connect_args)

        def init_db():
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            Base.metadata.create_all(bind=engine)

        await asyncio.to_thread(init_db)

        db_engine = engine
        db_session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        logger.info("Successfully connected to the database!")
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        db_engine = None
        db_session_factory = None

    yield  # Server is running

    # Cleanup: Release resources
    logger.info("Shutting down and releasing resources...")
    ocr_processor = None

    if db_engine:
        logger.info("Closing database connection...")
        db_engine.dispose()
        logger.info("Database connection closed.")


# Pass the lifespan context manager to FastAPI
app = FastAPI(
    title="Lao Air Writing OCR API",
    description="API for Lao air writing optical character recognition.",
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


def _insert_feedback(session_factory: sessionmaker, feedback_dict: dict) -> int:
    session: Session = session_factory()
    try:
        record = FeedbackRecord(**feedback_dict)
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id
    finally:
        session.close()


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackData = Body(...)):
    if db_session_factory is None:  # Check if the database connection was successful
        logger.error("Failed to submit feedback: database is not connected.")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable: Cannot store feedback at the moment.",
        )

    try:
        feedback_dict = feedback.model_dump(exclude_none=True)
        feedback_dict["submitted_at"] = datetime.now(
            timezone.utc
        )  # Add a UTC timestamp

        feedback_id = await asyncio.to_thread(
            _insert_feedback, db_session_factory, feedback_dict
        )
        logger.info(f"Feedback stored successfully with ID: {feedback_id}")

        return FeedbackResponse(
            success=True,
            message="Feedback submitted successfully! Thank you.",
            feedback_id=str(feedback_id),
        )
    except SQLAlchemyError as e:
        logger.error(f"Database error while storing feedback: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database connection error. Could not store feedback.",
        )
    except Exception as e:
        logger.exception(f"Failed to store feedback in the database: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not store feedback due to an internal server error.",
        )


def _list_feedback(session_factory: sessionmaker, limit: int, offset: int):
    session: Session = session_factory()
    try:
        query = session.query(FeedbackRecord).order_by(FeedbackRecord.submitted_at.desc())
        total = query.count()
        items = query.offset(offset).limit(limit).all()
        return total, items
    finally:
        session.close()


@app.get("/api/feedback", response_model=FeedbackListResponse)
async def get_feedback(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    if db_session_factory is None:
        logger.error("Failed to fetch feedback: database is not connected.")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable: Cannot fetch feedback at the moment.",
        )

    try:
        total, items = await asyncio.to_thread(
            _list_feedback, db_session_factory, limit, offset
        )
        return FeedbackListResponse(count=total, items=items)
    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching feedback: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database connection error. Could not fetch feedback.",
        )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    ocr_is_ready = ocr_processor is not None
    database_is_connected = False
    engine = db_engine
    if engine:
        try:
            def ping():
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

            await asyncio.to_thread(ping)
            database_is_connected = True
        except SQLAlchemyError:
            logger.warning("Health check: database connection ping failed.")
            database_is_connected = False
        except Exception as e:
            logger.warning(f"Health check: database status check error: {e}")
            database_is_connected = False

    current_status = "healthy"
    if not ocr_is_ready:
        current_status = "degraded"
        logger.warning("Health check: OCR processor not loaded.")
    if not database_is_connected:
        current_status = (
            "degraded" if current_status == "healthy" else current_status
        )  # Keep degraded if already set
        logger.warning("Health check: database not connected.")

    return HealthResponse(
        status=current_status,
        ocr_processor_loaded=ocr_is_ready,
        database_connected=database_is_connected,
        version=app.version,
    )
