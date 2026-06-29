from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT, settings
from app.database import Base, engine
from app.ml.predictor import predictor
from app.ml.explainer import explainer
from app.routers import health, orders, predict

_STATIC = PROJECT_ROOT / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    try:
        predictor.load()
        explainer.build(predictor.model)
        print(f"[returnshield] Model loaded: {predictor.meta['name']} v{predictor.meta['version']}")
    except FileNotFoundError as exc:
        print(f"[returnshield] WARNING: {exc}")
        print("[returnshield] Prediction endpoints need a trained model — run python train_model.py")
    yield


app = FastAPI(
    title="ReturnShield",
    description=(
        "Intelligent e-commerce returns decision engine. "
        "Predicts return probability, surfaces SHAP-based risk factors, "
        "estimates expected return cost, and generates actionable recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")
app.include_router(orders.router, prefix="/api/v1")

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(_STATIC / "index.html")
