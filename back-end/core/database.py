import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Float, Text, DateTime, create_engine, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from core.settings import get_settings

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(20), default="queued", nullable=False, index=True)
    mode = Column(String(10), nullable=False)
    image_s3_key = Column(String(500))
    image_url = Column(String(2000))

    # Classification results
    class_name = Column(String(100))
    confidence = Column(Float)
    predictions = Column(JSON)

    # Nutrition results
    nutrition = Column(JSON)
    nutrition_source = Column(String(50))

    # Ingredient detection results
    ingredients = Column(JSON)
    overlay_s3_key = Column(String(500))

    # Portion estimation results
    portion = Column(JSON)
    depth_map_s3_key = Column(String(500))

    # Error tracking
    error = Column(Text)

    # Request parameters
    models = Column(JSON)
    box_threshold = Column(Float, default=0.3)
    reference_height_cm = Column(Float)

    # Progress tracking
    progress = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(20), default="user", nullable=False)  # "user" or "admin"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Email Verification & Reset Password Fields
    email = Column(String(150), unique=True, nullable=True, index=True)
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_code = Column(String(10), nullable=True)
    verification_code_expires_at = Column(DateTime, nullable=True)


class UserSession(Base):
    __tablename__ = "user_sessions"

    session_token = Column(String(64), primary_key=True)
    user_id = Column(String(36), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)



_engine = None
_SessionLocal = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        db_url = settings.database_url
        # SQLite fallback for local dev without PostgreSQL
        if db_url.startswith("sqlite"):
            _engine = create_engine(db_url, connect_args={"check_same_thread": False})
            _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=True, expire_on_commit=False)
        else:
            try:
                temp_engine = create_engine(db_url, pool_size=10, max_overflow=20)
                # Test connection immediately
                with temp_engine.connect() as conn:
                    pass
                _engine = temp_engine
                _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=True, expire_on_commit=False)
                print("Database: Connected to PostgreSQL database successfully.")
            except Exception as e:
                print(f"Database: PostgreSQL connection failed: {e}. Falling back to SQLite backup database!")
                backup_url = "sqlite:///vnfood_backup.db"
                _engine = create_engine(backup_url, connect_args={"check_same_thread": False})
                _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=True, expire_on_commit=False)
    return _engine


def _get_session_local():
    _get_engine()
    global _SessionLocal
    return _SessionLocal


def init_db():
    """Create all tables if they don't exist and seed default admin."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    
    # Seed default admin user if users table is empty
    import hashlib
    import os
    
    session_local = _get_session_local()
    session = session_local()
    try:
        admin_exists = session.query(User).filter(User.username == "admin").first()
        if not admin_exists:
            # Hash password using PBKDF2-HMAC-SHA256
            salt = os.urandom(16)
            key = hashlib.pbkdf2_hmac('sha256', b"admin123", salt, 100000)
            hashed = salt.hex() + ":" + key.hex()
            
            admin_user = User(
                username="admin",
                hashed_password=hashed,
                role="admin",
                is_active=True
            )
            session.add(admin_user)
            session.commit()
            print("Successfully seeded default admin user (admin/admin123).")
    except Exception as e:
        session.rollback()
        print("Failed to seed default admin:", e)
    finally:
        session.close()



def get_session() -> Session:
    """Get a database session. Caller must close it."""
    SessionLocal = _get_session_local()
    session = SessionLocal()
    try:
        return session
    except Exception:
        session.close()
        raise


def create_job(mode: str, image_url: str = None, image_s3_key: str = None,
               models: list = None, box_threshold: float = 0.3,
               reference_height_cm: float = None) -> Job:
    """Create a new job in the database."""
    session = get_session()
    try:
        job = Job(
            mode=mode,
            image_url=image_url,
            image_s3_key=image_s3_key,
            models=models,
            box_threshold=box_threshold,
            reference_height_cm=reference_height_cm,
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        return job
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_job(job_id: str) -> Optional[Job]:
    """Get a job by ID."""
    session = get_session()
    try:
        job = session.query(Job).filter(Job.id == job_id).first()
        return job
    finally:
        session.close()


def update_job(job_id: str, **kwargs) -> Optional[Job]:
    """Update a job's fields."""
    session = get_session()
    try:
        job = session.query(Job).filter(Job.id == job_id).first()
        if job:
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            session.commit()
            session.refresh(job)
            
            # Publish update to Redis Pub/Sub channel
            try:
                from core.cache import _get_cache_mode, get_redis
                if _get_cache_mode() == "redis":
                    redis_client = get_redis()
                    job_dict = {
                        "id": str(job.id),
                        "status": job.status,
                        "mode": job.mode,
                        "image_s3_key": job.image_s3_key,
                        "image_url": job.image_url,
                        "class_name": job.class_name,
                        "confidence": job.confidence,
                        "predictions": job.predictions,
                        "nutrition": job.nutrition,
                        "nutrition_source": job.nutrition_source,
                        "ingredients": job.ingredients,
                        "overlay_s3_key": job.overlay_s3_key,
                        "portion": job.portion,
                        "depth_map_s3_key": job.depth_map_s3_key,
                        "error": job.error,
                        "models": job.models,
                        "box_threshold": job.box_threshold,
                        "reference_height_cm": job.reference_height_cm,
                        "progress": job.progress,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "started_at": job.started_at.isoformat() if job.started_at else None,
                        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    }
                    import json
                    redis_client.publish(f"job_updates:{job_id}", json.dumps(job_dict))
            except Exception as pe:
                print(f"Failed to publish job update to Redis: {pe}")
                
        return job
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()