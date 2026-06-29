import sys
import os

# Add back-end path to sys.path so imports remain unbroken
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.database import init_db
from core.model_registry import ModelRegistry


def preload_all_models():
    try:
        registry = ModelRegistry()
        registry.preload("classification", "dino", "sam2", "depth")
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error preloading models: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables
    init_db()
    
    # Preload models in a background thread to prevent blocking startup
    threading.Thread(target=preload_all_models, daemon=True).start()
    
    yield


app = FastAPI(
    title="Vietnamese Food Classifier API",
    description="Scalable API for Vietnamese food classification, nutrition lookup, ingredient detection, and portion estimation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


from fastapi.staticfiles import StaticFiles

# Mount front-end/static safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
static_dir = os.path.join(BASE_DIR, "front-end", "static")

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
