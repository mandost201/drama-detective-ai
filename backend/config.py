"""Central configuration for SceneFinder."""
import os
from pathlib import Path


def _sanitize_proxy_env():
    """httpx (used for model downloads) chokes on bracketed IPv6 literals
    like [::1] in no_proxy/NO_PROXY ('Invalid port' error). Strip those
    entries; everything else is left untouched."""
    for key in ("no_proxy", "NO_PROXY"):
        val = os.environ.get(key)
        if not val:
            continue
        kept = [p.strip() for p in val.split(",")
                if p.strip() and "[" not in p and "]" not in p]
        if len(kept) != len(val.split(",")):
            os.environ[key] = ",".join(kept)
            print(f"[config] sanitized {key} for httpx compatibility")


_sanitize_proxy_env()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
# Database file for Drama Detective AI (kept separate from SceneFinder).
DB_PATH = DATA_DIR / "drama_detective.db"
UPLOAD_DIR = DATA_DIR / "uploads"
THUMB_DIR = DATA_DIR / "thumbs"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

for d in (DATA_DIR, UPLOAD_DIR, THUMB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Keep AI model caches inside the project so they survive VM restarts.
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(MODELS_DIR / "fastembed"))
# hf_transfer (if installed) downloads models far more reliably through
# proxies than the default httpx-based downloader.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "").strip()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# Ingestion politeness / limits
TRANSCRIPT_WORKERS = int(os.environ.get("TRANSCRIPT_WORKERS", "4"))
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", "0.4"))  # seconds between YT requests
MAX_VIDEOS_PER_PLAYLIST = int(os.environ.get("MAX_VIDEOS_PER_PLAYLIST", "500"))
