"""Drama Detective AI - REST API over the shared backend package.

The Streamlit UI (streamlit/app.py) is the primary frontend and uses the
same backend modules in-process; this FastAPI app exposes the same
capabilities over HTTP for integrations and testing.
"""
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import db
from . import ingest
from . import search as search_mod
from . import analyze as analyze_mod
from .config import FRONTEND_DIST, UPLOAD_DIR
from .vtt import format_hms
from .youtube import extract_playlist_id, watch_url_at

db.init_db()

app = FastAPI(title="Drama Detective AI", version="1.0.0")


# ------------------------------------------------------------------ models
class PlaylistIn(BaseModel):
    url: str


class SearchIn(BaseModel):
    playlist_id: str
    query: str
    mode: str = "auto"
    top_k: int = 8


# ---------------------------------------------------------------- playlists
@app.post("/api/playlists")
def create_playlist(body: PlaylistIn):
    url = body.url.strip()
    if not extract_playlist_id(url):
        raise HTTPException(400, "Please paste a YouTube playlist URL "
                                 "(it should contain 'list=').")
    pid = db.create_playlist(url)
    jid = ingest.start_playlist_job(pid, url)
    return {"playlist_id": pid, "job_id": jid, "status": "queued"}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    j = db.get_job(jid)
    if not j:
        raise HTTPException(404, "Job not found")
    return j


@app.get("/api/playlists")
def playlists():
    out = []
    for p in db.list_playlists():
        p = dict(p)
        p["analyzed_at_h"] = (time.strftime("%Y-%m-%d %H:%M",
                                            time.localtime(p["analyzed_at"]))
                              if p.get("analyzed_at") else None)
        out.append(p)
    return {"playlists": out}


@app.get("/api/playlists/{pid}")
def playlist_detail(pid: str, q: str = ""):
    p = db.get_playlist(pid)
    if not p:
        raise HTTPException(404, "Playlist not found")
    videos = db.get_videos(pid)
    if q:
        ql = q.lower()
        videos = [v for v in videos if ql in (v["title"] or "").lower()]
    return {"playlist": p, "videos": videos,
            "segment_count": db.segment_count(pid)}


@app.delete("/api/playlists/{pid}")
def delete_playlist(pid: str):
    db.delete_playlist(pid)
    search_mod.invalidate_cache(pid)
    return {"ok": True}


@app.get("/api/videos/{vid}/transcript")
def video_transcript(vid: int):
    v = db.get_video(vid)
    if not v:
        raise HTTPException(404, "Video not found")
    segs = db.get_segments(vid)
    return {"video": v, "segments": [
        {"start": s["start"], "hms": format_hms(s["start"]), "text": s["text"]}
        for s in segs]}


# ------------------------------------------------------------------ search
@app.post("/api/search")
def search(body: SearchIn):
    p = db.get_playlist(body.playlist_id)
    if not p:
        raise HTTPException(404, "Playlist not found")
    if p.get("status") != "ready":
        raise HTTPException(409, "Playlist is still being analyzed - "
                                 "please wait for it to finish.")
    return search_mod.search(body.playlist_id, body.query,
                             mode=body.mode, top_k=body.top_k)


# ------------------------------------------------------------- Option A/B
@app.post("/api/analyze/clip")
async def analyze_clip(playlist_id: str = Form(...),
                       file: UploadFile = File(...)):
    p = db.get_playlist(playlist_id)
    if not p:
        raise HTTPException(404, "Playlist not found")
    suffix = Path(file.filename or "clip").suffix.lower() or ".mp4"
    if suffix not in (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".mp3", ".wav", ".ogg"):
        raise HTTPException(400, "Unsupported file type - please upload a "
                                 "video or audio clip.")
    tmp = UPLOAD_DIR / f"clip_{int(time.time())}_{abs(hash(file.filename)) % 10**6}{suffix}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return analyze_mod.analyze_clip(playlist_id, tmp)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


@app.post("/api/analyze/screenshots")
async def analyze_screenshots(playlist_id: str = Form(...),
                              files: list[UploadFile] = File(...)):
    p = db.get_playlist(playlist_id)
    if not p:
        raise HTTPException(404, "Playlist not found")
    if not files or len(files) > 6:
        raise HTTPException(400, "Upload 1-6 screenshots.")
    paths = []
    for i, f in enumerate(files):
        suffix = Path(f.filename or "shot").suffix.lower() or ".png"
        if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
            raise HTTPException(400, f"Unsupported image type: {f.filename}")
        tmp = UPLOAD_DIR / f"shot_{int(time.time())}_{i}{suffix}"
        with open(tmp, "wb") as fobj:
            shutil.copyfileobj(f.file, fobj)
        paths.append(tmp)
    try:
        return analyze_mod.analyze_screenshots(playlist_id, paths)
    finally:
        for t in paths:
            try:
                t.unlink()
            except OSError:
                pass


@app.get("/api/health")
def health():
    return {"ok": True, "semantic": search_mod.embeddings.available()}


# ------------------------------------------------------- frontend hosting
# Serve the built React SPA with fallback so client-side routes (/p/:id)
# work on refresh/direct navigation. API routes are registered first and
# take precedence; unknown /api/* paths still 404 as JSON.
if FRONTEND_DIST.exists():
    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "Not found")
        target = (FRONTEND_DIST / full_path) if full_path else None
        if target and target.is_file():
            return FileResponse(target)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    def no_ui():
        return {"message": "SceneFinder API is running. Build the frontend "
                           "to get the full UI."}
