from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, field_validator


class AnalyzeRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    mode: Literal["fast", "accurate"] = "fast"
    models: Optional[list[str]] = None
    box_threshold: float = 0.3
    reference_height_cm: Optional[float] = None

    @field_validator("image_url", "image_base64")
    @classmethod
    def check_image_source(cls, v, info):
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in ("fast", "accurate"):
            raise ValueError("mode must be 'fast' or 'accurate'")
        return v


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str
    mode: str
    created_at: datetime


class PredictionItem(BaseModel):
    rank: int
    class_name: str
    probability: float


class NutritionItem(BaseModel):
    name: str
    value: float
    unit: str


class IngredientResult(BaseModel):
    label: str
    confidence: float
    bbox: list[float]
    mask_pixel_count: int
    mask_area_ratio: Optional[float] = None


class PortionResult(BaseModel):
    estimated_weight_grams: float
    estimated_volume_ml: float
    density_used: float
    scaling_method: str
    nutrient_multiplier: float
    typical_portion_grams: int
    area_ratio: Optional[float] = None
    depth_map_s3_key: Optional[str] = None


class AnalysisResult(BaseModel):
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    predictions: Optional[dict] = None
    nutrition: Optional[dict] = None
    nutrition_source: Optional[str] = None
    ingredients: Optional[list[dict]] = None
    overlay_url: Optional[str] = None
    portion: Optional[dict] = None
    depth_map_url: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    mode: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[dict] = None
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    path: str
    type: str  # "pytorch" or "onnx"


class HealthResponse(BaseModel):
    status: str
    redis: str
    database: str
    models_loaded: list[str]


# --- Admin & Auth Schemas ---

class UserRegisterRequest(BaseModel):
    username: str
    password: str


class UserLoginRequest(BaseModel):
    username: str
    password: str


class UserLoginResponse(BaseModel):
    session_token: str
    username: str
    role: str


class UserResponse(BaseModel):
    id: str
    username: str
    role: str
    is_active: bool
    created_at: datetime


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class UserUpdateRequest(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class AdminStatsResponse(BaseModel):
    total_users: int
    active_sessions: int
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    db_status: str