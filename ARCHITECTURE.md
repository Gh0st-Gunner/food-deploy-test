# Project Status & Architecture Guide

## What Was Built

### Original Streamlit App (unchanged, still working)
- **URL**: http://localhost:8502
- **Entry**: `app.py` → 3 tabs: Classification & Analysis, Model Comparison, Information
- **Pipeline**: Upload image → classify (EfficientNet-B0 + ResNet-50) → USDA/FatSecret nutrition → Grounding DINO + SAM 2 ingredient detection → area-ratio portion estimation → Depth Anything V2 visualization
- **USDA API key**: configured in `.streamlit/secrets.toml`
- **Models**: `models/eff_b0.pth`, `models/resnet50_old.pth`, plus HuggingFace downloads (Grounding DINO ~660MB, SAM 2.1 ~184MB, Depth Anything V2 ~100MB)

### New FastAPI + Celery Backend
- **URL**: http://127.0.0.1:10800 (Swagger docs at `/docs`)
- **Endpoints**:
  - `POST /api/v1/analyze` — submit image (URL or base64) + mode (fast/accurate), returns job_id
  - `GET /api/v1/jobs/{id}` — poll job status and results
  - `GET /api/v1/health` — Redis (memory/redis), DB (ok/error), models list
  - `GET /api/v1/models` — list available .pth/.onnx models

### New Files Created

```
core/
  settings.py       — Pydantic Settings (VNFOOD_ env prefix)
  database.py       — SQLAlchemy Job model, SQLite fallback, CRUD
  cache.py           — Redis or in-memory dict caching (auto-fallback)
  storage.py         — S3/MinIO or local filesystem (auto-fallback)
  model_registry.py  — Singleton lazy model loading (replaces @st.cache_resource)

api/
  main.py            — FastAPI app, CORS, lifespan
  routes.py           — 4 endpoints, Celery task dispatch
  schemas.py          — Pydantic request/response models
  dependencies.py     — Redis/session helpers

workers/
  celery_app.py           — Celery config, task routes, preload signal
  classification_worker.py — classify_food task
  nutrition_worker.py       — lookup_nutrition + ingredient_nutrition tasks
  detection_worker.py       — DINO + SAM2 ingredient detection task
  portion_worker.py         — depth + area-ratio portion estimation task
  aggregator_worker.py      — finalize job after all tasks complete

nutrition/
  nutrition_cache.py  — @st.cache_data wrappers for Streamlit (new)
  usda_client.py       — removed @st.cache_data, fixed nutrient matching (nutrientNumber vs nutrientId)
  fatsecret_client.py  — removed @st.cache_data and import streamlit
  nutrition_provider.py — removed @st.cache_data and import streamlit

classification/model_loader.py — removed @st.cache_resource, added module-level dict caching
segmentation/grounding_dino_loader.py — removed @st.cache_resource, added module-level caching
segmentation/sam2_loader.py — same
depth/depth_loader.py — same

Dockerfile                  — Python 3.11 slim, installs requirements
docker-compose.yml          — API, 4 worker queues, Redis, PostgreSQL, MinIO
.env.example                — Template for all config vars
scripts/init.sh             — MinIO bucket creation
```

### Key Bug Fixes During Development
1. **USDA nutrient matching**: `get_nutrients()` was comparing `nutrientId` (1003, 1004...) against `NUTRIENT_IDS` values that were actually `nutrientNumber` (203, 204...). Fixed by matching on `nutrientNumber` as strings.
2. **USDA dataType 400 errors**: `requests` was encoding `dataType` as a list, causing random 400s from USDA's nginx. Fixed by using explicit tuple params with repeated keys.
3. **USDA empty nutrient results**: `search_food()` returned the first result which sometimes had 0 nutrients. Added `find_food_with_nutrients()` that scans all results for the first with actual nutrient data.
4. **Ingredient nutrition "No data found"**: Cascading from the above fixes — all three test ingredients (jalapeno pepper, baguette bread, carrot) now return 11 nutrients each.
5. **Portion estimation accuracy**: Replaced depth-based volume estimation with SAM 2 area-ratio method. Depth map is now visualization-only.
6. **FatSecret IP restriction**: Basic tier blocks non-US IPs. User's IP (42.115.193.17) is Vietnamese. Needs Premier Free tier.
7. **Model loader Streamlit decoupling**: Removed `@st.cache_resource` from all 4 model loaders, replaced with module-level dict caching. Added `nutrition_cache.py` with `@st.cache_data` wrappers for Streamlit compatibility.

### Local Dev Fallbacks (no Docker needed)
| Component | Production | Local Dev Fallback |
|-----------|-----------|-------------------|
| Database | PostgreSQL | SQLite (`vnfood.db`) |
| Cache | Redis | In-memory dict with TTL |
| Storage | S3/MinIO | Local `storage/` directory |
| Task Queue | Celery + Redis | Memory broker (`memory://`) — jobs created but not processed |

## Next Steps & Integration Plan

### 1. Docker Deployment (requires Docker Desktop)
```bash
# Start infrastructure
docker-compose up -d db redis minio

# Create .env from template
cp .env.example .env
# Edit .env with your values

# Start API
docker-compose up -d api

# Start workers (1 GPU machine, combined detection+portion)
docker-compose up -d worker-classification worker-detection worker-nutrition worker-default
```

### 2. Celery Worker Testing
Once Redis is running:
```bash
# Classification + nutrition workers (CPU)
celery -A workers.celery_app worker -Q classification,nutrition -c 2 --loglevel=info

# Detection + portion worker (GPU)
celery -A workers.celery_app worker -Q detection -c 1 --loglevel=info

# Default/aggregator worker (CPU)
celery -A workers.celery_app worker -Q default -c 2 --loglevel=info
```

### 3. Remaining Work
- **WebSocket updates**: Add `GET /api/v1/jobs/{id}/stream` for real-time progress (currently polling only)
- **Fast/Accurate mode in workers**: The `classify_food` task returns `class_name` but downstream tasks (nutrition, detection) need it as input. Verify the Celery chain passes results correctly.
- **Mask serialization**: Ingredient masks are numpy arrays that can't be JSON-serialized. Currently storing area stats only. For full mask data, store as PNG in S3 and reference by key.
- **Rate limiting**: Add rate limiting middleware to FastAPI (`slowapi` or similar)
- **Authentication**: Add API key auth for the `/analyze` endpoint
- **FatSecret Premier Free tier**: Apply at https://platform.fatsecret.com/upgrade-account?type=1 to remove IP restriction
- **Rotate credentials**: FatSecret Client ID `5ba44ee862f44eb7a0767ac3a3b0e0ae` and Secret `760da0ab93ee48d3b8828faed154f66c` were exposed in chat — regenerate them

### 4. Architecture Decisions
- **Celery over Temporal**: Simpler setup, sufficient for MVP. Temporal can be added later for durable workflows.
- **SQLite fallback**: `core/database.py` uses SQLAlchemy with connection-level `check_same_thread=False` for SQLite. Switches to PostgreSQL when `VNFOOD_DATABASE_URL` points to one.
- **Memory cache fallback**: `core/cache.py` auto-detects Redis availability. Falls back to a dict with TTL expiry. Not shared across workers — Redis required for multi-worker caching.
- **Local file storage fallback**: `core/storage.py` stores images in `storage/` directory. MinIO/S3 required for distributed workers.
- **Streamlit still works**: All model loaders and nutrition code work with or without Streamlit. The `nutrition_cache.py` provides `@st.cache_data` wrappers for the Streamlit UI.

### 5. Key File Locations
| Purpose | File |
|---------|------|
| Streamlit entry point | `app.py` |
| FastAPI entry point | `api/main.py` |
| Celery config | `workers/celery_app.py` |
| All config (backend) | `core/settings.py` (env vars with `VNFOOD_` prefix) |
| All config (Streamlit) | `config.py` |
| USDA API key | `.streamlit/secrets.toml` |
| USDA client (with fixes) | `nutrition/usda_client.py` |
| Portion estimator (area-ratio) | `depth/portion_estimator.py` |
| Model loading (no Streamlit) | `classification/model_loader.py`, `core/model_registry.py` |
| Docker infra | `docker-compose.yml`, `Dockerfile` |
| Job DB schema | `core/database.py` (Job model) |

### 6. API Request Examples
```bash
# Fast mode (classification + nutrition only)
curl -X POST http://localhost:10800/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/pho.jpg", "mode": "fast"}'

# Accurate mode (full pipeline)
curl -X POST http://localhost:10800/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/pho.jpg", "mode": "accurate"}'

# Base64 image
curl -X POST http://localhost:10800/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64>", "mode": "fast"}'

# Poll job status
curl http://localhost:10800/api/v1/jobs/<job_id>

# Health check
curl http://localhost:10800/api/v1/health

# List models
curl http://localhost:10800/api/v1/models
```

### 7. Ports
| Service | Port |
|---------|------|
| Streamlit UI | 8502 |
| FastAPI API | 10800 |
| PostgreSQL | 5432 (Docker) |
| Redis | 6379 (Docker) |
| MinIO API | 9000 (Docker) |
| MinIO Console | 9001 (Docker) |