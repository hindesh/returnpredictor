from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db
from app.ml.predictor import predictor

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check(db: Session = Depends(get_db)):
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    model_ok = predictor._loaded

    return {
        "status": "ok" if (db_ok and model_ok) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "database": "ok" if db_ok else "unavailable",
            "model": "loaded" if model_ok else "not loaded",
        },
        "model_info": predictor.meta if model_ok else None,
    }
