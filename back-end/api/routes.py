import base64
import io
from datetime import datetime
from typing import Optional

import requests as http_requests
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Header

from api.schemas import (
    AnalyzeRequest, AnalyzeResponse, JobStatusResponse,
    AnalysisResult, ModelInfo, HealthResponse,
    UserRegisterRequest, UserLoginRequest, UserLoginResponse,
    UserResponse, UserCreateRequest, UserUpdateRequest, AdminStatsResponse,
    RecommendRequest, VerifyEmailRequest, ForgotPasswordRequest, ResetPasswordRequest
)
from core.database import create_job, get_job, update_job, init_db, get_session, User, UserSession, Job
from sqlalchemy import text
from core.storage import upload_image, get_presigned_url, download_bytes
from core.model_registry import ModelRegistry
from core.settings import get_settings
from api.auth import get_current_user, get_current_admin, hash_password, verify_password, create_session, destroy_session

router = APIRouter()
settings = get_settings()


def run_in_process_fallback(job_id: str, image_s3_key: str, request_models: list, request_mode: str, box_threshold: float, reference_height_cm: float):
    try:
        from workers.classification_worker import classify_food
        from workers.nutrition_worker import lookup_nutrition_task
        from workers.detection_worker import detect_ingredients_task
        from workers.portion_worker import estimate_portion_task
        from workers.aggregator_worker import aggregate_results

        # Update status to started
        update_job(job_id, started_at=datetime.utcnow())
        
        # 1. Classify
        classify_res = classify_food.run(job_id, image_s3_key, request_models)
        class_name = classify_res.get("class_name")
        
        # 2. Nutrition
        if class_name:
            lookup_nutrition_task.run(job_id, class_name)
            
        # 3. Always run detection and portion estimation (Accurate pipeline)
        try:
            params = {
                "box_threshold": box_threshold,
                "reference_height_cm": reference_height_cm,
            }
            
            detect_res = detect_ingredients_task.run(job_id, class_name, image_s3_key, params)
            estimate_portion_task.run(job_id, class_name, image_s3_key, detect_res, params)
        except Exception as e:
            # Log error but don't fail the whole job
            print(f"Skipping accurate models (DINO/SAM2/Depth) in fallback: {e}")
                
        # 4. Aggregate
        aggregate_results.run(job_id)
    except Exception as e:
        update_job(job_id, status="failed", error=f"In-process fallback failed: {e}")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Submit an image for analysis. Returns a job_id immediately."""
    # Validate image source
    if not request.image_url and not request.image_base64:
        raise HTTPException(status_code=400, detail="Provide image_url or image_base64")

    # Download or decode image
    if request.image_url:
        try:
            resp = http_requests.get(request.image_url, timeout=15)
            resp.raise_for_status()
            image_bytes = resp.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")
    else:
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    # Upload to S3
    image_s3_key = upload_image("pending", image_bytes)

    # Create job with forced mode='accurate'
    models = request.models or list(ModelRegistry().list_classification_models().keys())
    job = create_job(
        mode="accurate",
        image_s3_key=image_s3_key,
        image_url=request.image_url,
        models=models,
        box_threshold=request.box_threshold,
        reference_height_cm=request.reference_height_cm,
    )

    # Update S3 key with actual job ID
    image_s3_key = f"images/{job.id}/original.jpg"
    update_job(str(job.id), image_s3_key=image_s3_key)
    upload_image(str(job.id), image_bytes)

    # Build task pipeline
    params = {
        "box_threshold": request.box_threshold,
        "reference_height_cm": request.reference_height_cm,
    }

    job_id = str(job.id)

    # Check cache mode (memory vs redis)
    from core.cache import _get_cache_mode
    if _get_cache_mode() == "memory":
        # Run in-process fallback using FastAPI's BackgroundTasks
        background_tasks.add_task(
            run_in_process_fallback,
            job_id,
            image_s3_key,
            request.models,
            "accurate",
            request.box_threshold,
            request.reference_height_cm
        )
    else:
        # Try to dispatch Celery task; if broker is down, fall back to in-process background task
        try:
            from celery import chain
            from workers.celery_app import configure_celery
            from workers.classification_worker import classify_food
            from workers.nutrition_worker import lookup_nutrition_task
            from workers.detection_worker import detect_ingredients_task
            from workers.portion_worker import estimate_portion_task
            from workers.aggregator_worker import aggregate_results

            configure_celery()

            workflow = chain(
                classify_food.si(job_id, image_s3_key, request.models),
                lookup_nutrition_task.si(job_id),
                detect_ingredients_task.si(job_id, None, image_s3_key, params),
                estimate_portion_task.si(job_id, None, image_s3_key, None, params),
                aggregate_results.si(job_id),
            )

            workflow.apply_async()
        except Exception as e:
            # Fall back to in-process background task
            background_tasks.add_task(
                run_in_process_fallback,
                job_id,
                image_s3_key,
                request.models,
                "accurate",
                request.box_threshold,
                request.reference_height_cm
            )

    return AnalyzeResponse(
        job_id=job_id,
        status="queued",
        mode="accurate",
        created_at=job.created_at,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll job status and results."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = None
    if job.status == "completed":
        overlay_url = get_presigned_url(job.overlay_s3_key) if job.overlay_s3_key else None
        depth_map_url = get_presigned_url(job.depth_map_s3_key) if job.depth_map_s3_key else None

        result = AnalysisResult(
            class_name=job.class_name,
            confidence=job.confidence,
            predictions=job.predictions,
            nutrition=job.nutrition,
            nutrition_source=job.nutrition_source,
            ingredients=job.ingredients,
            overlay_url=overlay_url,
            portion=job.portion,
            depth_map_url=depth_map_url,
        )

    return JobStatusResponse(
        job_id=str(job.id),
        status=job.status,
        mode=job.mode,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=job.progress,
        result=result,
        error=job.error,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Redis, database, and model availability."""
    redis_ok = "ok"
    try:
        from core.cache import _get_cache_mode
        mode = _get_cache_mode()
        redis_ok = mode  # "redis" or "memory"
    except Exception as e:
        redis_ok = f"error: {e}"

    db_ok = "ok"
    try:
        init_db()
    except Exception as e:
        db_ok = f"error: {e}"

    models = list(ModelRegistry().list_classification_models().keys())

    return HealthResponse(
        status="ok" if redis_ok in ["ok", "redis"] and db_ok == "ok" else "degraded",
        redis=redis_ok,
        database=db_ok,
        models_loaded=models,
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    """List available classification models."""
    registry = ModelRegistry()
    available = registry.list_classification_models()
    models = [
        ModelInfo(
            name=name,
            path=path,
            type="onnx" if path.endswith(".onnx") else "pytorch",
        )
        for name, path in available.items()
    ]
    
    # Always include Google Gemini option
    models.append(ModelInfo(name="gemini:gemini-1.5-flash", path="cloud", type="gemini"))
    
    # Dynamically query local Ollama instance for pulled models
    import requests
    from core.settings import get_settings
    settings = get_settings()
    
    try:
        ollama_url = settings.ollama_host
        if "localhost" in ollama_url or "127.0.0.1" in ollama_url:
            ollama_url = ollama_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
            
        r = requests.get(f"{ollama_url}/api/tags", timeout=1.0)
        if r.status_code == 200:
            ollama_data = r.json()
            local_models = ollama_data.get("models", [])
            for m in local_models:
                m_name = m.get("name", "")
                if m_name:
                    models.append(ModelInfo(name=f"ollama:{m_name}", path="local", type="ollama"))
    except Exception as e:
        print(f"Could not reach Ollama at {settings.ollama_host} to list tags: {e}")
        
    # Fallback to templates if Ollama is empty/offline
    if not any(m.name.startswith("ollama:") for m in models):
        models.append(ModelInfo(name="ollama:llava", path="local", type="ollama"))
        models.append(ModelInfo(name="ollama:llava-phi3", path="local", type="ollama"))
        models.append(ModelInfo(name="ollama:bakllava", path="local", type="ollama"))
        
    return models


@router.get("/explore")
async def explore(
    calories: Optional[int] = None,
    protein: Optional[int] = None,
    carbs: Optional[int] = None,
    fat: Optional[int] = None
):
    """Retrieve scraped healthy food options for exploring new dishes."""
    from api.explore_scraper import get_explore_dishes
    return get_explore_dishes(
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat
    )


@router.get("/explore/generate")
async def explore_generate(
    ingredients: str,
    calories: Optional[int] = None,
    protein: Optional[int] = None,
    carbs: Optional[int] = None,
    fat: Optional[int] = None
):
    """Generate healthy recipes based on provided ingredients and target macros."""
    from api.explore_scraper import generate_recipes_from_ingredients
    return generate_recipes_from_ingredients(
        ingredients=ingredients,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat
    )


@router.post("/explore/recommend")
async def explore_recommend(request: RecommendRequest):
    """Rank and filter explore dishes using personalized Flavor AI recommendation engine."""
    from api.explore_scraper import get_explore_dishes
    from api.recommendation_engine import recommend_dishes
    
    profile_dict = request.user_profile.model_dump()
    recent_meals_dicts = [meal.model_dump() for meal in request.recent_meals]
    
    candidates = get_explore_dishes(
        calories=profile_dict.get("target_calories"),
        protein=profile_dict.get("target_protein"),
        carbs=profile_dict.get("target_carbs"),
        fat=profile_dict.get("target_fat")
    )
    
    ranked_dishes = recommend_dishes(
        user_profile=profile_dict,
        recent_meals=recent_meals_dicts,
        candidate_dishes=candidates
    )
    return ranked_dishes


@router.websocket("/jobs/{job_id}/stream")
async def websocket_jobs_stream(websocket: WebSocket, job_id: str):
    """Stream job status updates and progress in real-time."""
    from fastapi import WebSocket, WebSocketDisconnect
    import asyncio
    from core.database import get_job
    from core.storage import get_presigned_url

    await websocket.accept()

    last_progress = None
    last_status = None

    try:
        while True:
            job = get_job(job_id)
            if not job:
                await websocket.send_json({"error": "Job not found"})
                break

            current_progress = job.progress
            current_status = job.status

            # Send update if state has changed
            if current_progress != last_progress or current_status != last_status:
                last_progress = current_progress
                last_status = current_status

                result = None
                if current_status == "completed":
                    overlay_url = get_presigned_url(job.overlay_s3_key) if job.overlay_s3_key else None
                    depth_map_url = get_presigned_url(job.depth_map_s3_key) if job.depth_map_s3_key else None

                    result = {
                        "class_name": job.class_name,
                        "confidence": job.confidence,
                        "predictions": job.predictions,
                        "nutrition": job.nutrition,
                        "nutrition_source": job.nutrition_source,
                        "ingredients": job.ingredients,
                        "overlay_url": overlay_url,
                        "portion": job.portion,
                        "depth_map_url": depth_map_url,
                    }

                await websocket.send_json({
                    "job_id": str(job.id),
                    "status": job.status,
                    "mode": job.mode,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "progress": job.progress,
                    "result": result,
                    "error": job.error,
                })

            if current_status in ["completed", "failed"]:
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        # Client disconnected cleanly
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": f"Internal server error: {e}"})
        except Exception:
            pass


# --- Authentication Endpoints ---

@router.post("/auth/register", response_model=UserResponse)
async def register(request: UserRegisterRequest):
    db = get_session()
    try:
        existing = db.query(User).filter(User.username == request.username).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
            
        if request.email:
            existing_email = db.query(User).filter(User.email == request.email).first()
            if existing_email:
                raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed = hash_password(request.password)
        new_user = User(
            username=request.username,
            email=request.email,
            hashed_password=hashed,
            is_verified=False,
            role="user",
            is_active=True
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Send verification code if email is provided
        if request.email:
            import random
            from datetime import timedelta
            from core.email import send_email
            
            code = f"{random.randint(100000, 999999)}"
            new_user.verification_code = code
            new_user.verification_code_expires_at = datetime.utcnow() + timedelta(minutes=15)
            db.commit()
            
            send_email(
                request.email,
                "Munchin' - Xac thuc tai khoan",
                f"<h3>Xác thực tài khoản</h3><p>Mã xác thực tài khoản Munchin' của bạn là: <strong>{code}</strong>. Mã này có hiệu lực trong 15 phút.</p>"
            )
            
        return new_user
    finally:
        db.close()


@router.post("/auth/login", response_model=UserLoginResponse)
async def login(request: UserLoginRequest):
    db = get_session()
    try:
        # Search by username OR email
        user = db.query(User).filter(
            (User.username == request.username) | (User.email == request.username)
        ).first()
        
        if not user or not verify_password(request.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Incorrect username/email or password")
            
        if not user.is_active:
            raise HTTPException(status_code=403, detail="User account is deactivated")
            
        token = create_session(user.id)
        return UserLoginResponse(
            session_token=token,
            username=user.username,
            role=user.role
        )
    finally:
        db.close()


@router.post("/auth/logout")
async def logout(authorization: str = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        destroy_session(token)
    return {"message": "Successfully logged out"}


@router.post("/auth/send-verification")
async def send_verification(current_user: User = Depends(get_current_user)):
    if not current_user.email:
        raise HTTPException(status_code=400, detail="User does not have an email registered")
    
    db = get_session()
    try:
        db_user = db.query(User).filter(User.id == current_user.id).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
        if db_user.is_verified:
            return {"message": "Email is already verified"}
            
        import random
        from datetime import timedelta
        from core.email import send_email
        
        code = f"{random.randint(100000, 999999)}"
        db_user.verification_code = code
        db_user.verification_code_expires_at = datetime.utcnow() + timedelta(minutes=15)
        db.commit()
        
        send_email(
            db_user.email,
            "Munchin' - Xac thuc tai khoan",
            f"<h3>Xác thực tài khoản</h3><p>Mã xác thực tài khoản Munchin' của bạn là: <strong>{code}</strong>. Mã này có hiệu lực trong 15 phút.</p>"
        )
        return {"message": "Verification email sent"}
    finally:
        db.close()


@router.post("/auth/verify-email")
async def verify_email(req: VerifyEmailRequest, current_user: User = Depends(get_current_user)):
    db = get_session()
    try:
        db_user = db.query(User).filter(User.id == current_user.id).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
        if not db_user.verification_code or db_user.verification_code != req.code:
            raise HTTPException(status_code=400, detail="Invalid verification code")
        if db_user.verification_code_expires_at < datetime.utcnow():
            raise HTTPException(status_code=400, detail="Verification code has expired")
            
        db_user.is_verified = True
        db_user.verification_code = None
        db_user.verification_code_expires_at = None
        db.commit()
        return {"message": "Email successfully verified"}
    finally:
        db.close()


@router.post("/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    db = get_session()
    try:
        db_user = db.query(User).filter(User.email == req.email).first()
        if not db_user:
            return {"message": "If the email exists, a password reset code has been sent"}
            
        import random
        from datetime import timedelta
        from core.email import send_email
        
        code = f"{random.randint(100000, 999999)}"
        db_user.verification_code = code
        db_user.verification_code_expires_at = datetime.utcnow() + timedelta(minutes=15)
        db.commit()
        
        send_email(
            db_user.email,
            "Munchin' - Yeu cau dat lai mat khau",
            f"<h3>Đặt lại mật khẩu</h3><p>Mã đặt lại mật khẩu của bạn là: <strong>{code}</strong>. Mã này có hiệu lực trong 15 phút.</p>"
        )
        return {"message": "Password reset code sent"}
    finally:
        db.close()


@router.post("/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    db = get_session()
    try:
        db_user = db.query(User).filter(User.email == req.email).first()
        if not db_user or not db_user.verification_code or db_user.verification_code != req.code:
            raise HTTPException(status_code=400, detail="Invalid code or email")
        if db_user.verification_code_expires_at < datetime.utcnow():
            raise HTTPException(status_code=400, detail="Reset code has expired")
            
        db_user.hashed_password = hash_password(req.new_password)
        db_user.verification_code = None
        db_user.verification_code_expires_at = None
        
        from core.database import UserSession
        db.query(UserSession).filter(UserSession.user_id == db_user.id).delete()
        
        db.commit()
        return {"message": "Password successfully reset"}
    finally:
        db.close()


# --- Admin User Management Endpoints ---

@router.get("/admin/users", response_model=list[UserResponse])
async def list_users(admin: User = Depends(get_current_admin)):
    db = get_session()
    try:
        users = db.query(User).order_by(User.created_at.desc()).all()
        return users
    finally:
        db.close()


@router.post("/admin/users", response_model=UserResponse)
async def create_user(request: UserCreateRequest, admin: User = Depends(get_current_admin)):
    db = get_session()
    try:
        existing = db.query(User).filter(User.username == request.username).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
            
        hashed = hash_password(request.password)
        new_user = User(
            username=request.username,
            hashed_password=hashed,
            role=request.role,
            is_active=True
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    finally:
        db.close()


@router.put("/admin/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, request: UserUpdateRequest, admin: User = Depends(get_current_admin)):
    db = get_session()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        if request.username is not None:
            if request.username != user.username:
                existing = db.query(User).filter(User.username == request.username).first()
                if existing:
                    raise HTTPException(status_code=400, detail="Username already exists")
            user.username = request.username
            
        if request.password is not None and request.password != "":
            user.hashed_password = hash_password(request.password)
            
        if request.role is not None:
            user.role = request.role
            
        if request.is_active is not None:
            user.is_active = request.is_active
            if not request.is_active:
                db.query(UserSession).filter(UserSession.user_id == user_id).delete()
                
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


@router.delete("/admin/users/{user_id}")
async def delete_user(user_id: str, admin: User = Depends(get_current_admin)):
    db = get_session()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        if user.username == "admin" or user.id == admin.id:
            raise HTTPException(status_code=400, detail="Cannot delete default admin or yourself")
            
        db.query(UserSession).filter(UserSession.user_id == user_id).delete()
        db.delete(user)
        db.commit()
        return {"message": "User successfully deleted"}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/toggle-status", response_model=UserResponse)
async def toggle_user_status(user_id: str, admin: User = Depends(get_current_admin)):
    db = get_session()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        if user.id == admin.id:
            raise HTTPException(status_code=400, detail="Cannot block yourself")
            
        user.is_active = not user.is_active
        if not user.is_active:
            db.query(UserSession).filter(UserSession.user_id == user_id).delete()
            
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


@router.get("/admin/stats", response_model=AdminStatsResponse)
async def get_admin_stats(admin: User = Depends(get_current_admin)):
    db = get_session()
    try:
        total_users = db.query(User).count()
        active_sessions = db.query(UserSession).count()
        total_jobs = db.query(Job).count()
        completed_jobs = db.query(Job).filter(Job.status == "completed").count()
        failed_jobs = db.query(Job).filter(Job.status == "failed").count()
        
        return AdminStatsResponse(
            total_users=total_users,
            active_sessions=active_sessions,
            total_jobs=total_jobs,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            db_status="healthy"
        )
    finally:
        db.close()