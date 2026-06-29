import sys
from pathlib import Path

# Make project root importable from Vercel's /var/task/api/ context
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app  # noqa: F401 — Vercel picks up the ASGI `app`
