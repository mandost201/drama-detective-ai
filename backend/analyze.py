"""Option A: analyze an uploaded clip or screenshots against a playlist.

Clip pipeline (honest, real):
  upload -> ffmpeg audio extract -> faster-whisper transcription (real AI)
  -> align clip words against each video's transcript -> episode + timestamp.

Screenshot pipeline (limited, labelled as such):
  OCR any visible text (tesseract, if installed) -> keyword search;
  perceptual-hash visual similarity vs video thumbnails (weak signal).
  Visual scene search across full videos would require downloading every
  video - this is stated plainly in the UI instead of being faked.
"""
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path

from . import db
from . import search as search_mod
from .align import align_against_playlist
from .config import THUMB_DIR, WHISPER_MODEL
from .vtt import format_hms
from .youtube import watch_url_at

_whisper_lock = threading.Lock()
_whisper_model = None


def _whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu",
                                          compute_type="int8")
            print(f"[analyze] whisper model '{WHISPER_MODEL}' loaded")
        return _whisper_model


def _wav_to_numpy(wav_path: Path):
    """Read a 16-bit PCM WAV into a float32 numpy array (no PyAV needed)."""
    import wave
    import numpy as np
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, \
            "expected 16-bit mono WAV"
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_clip(path: Path):
    """Extract audio and transcribe. Returns (full_text, segments)."""
    wav = path.with_suffix(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path),
         "-ac", "1", "-ar", "16000", "-vn", "-c:a", "pcm_s16le", str(wav)],
        check=True, timeout=300)
    try:
        audio = _wav_to_numpy(wav)
    finally:
        try:
            wav.unlink()
        except OSError:
            pass
    model = _whisper()
    # pass raw audio array: avoids faster-whisper's PyAV decode path
    # (incompatible with av>=15 which dropped the metadata_errors kwarg)
    segments, _ = model.transcribe(audio, beam_size=5)
    out = [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
    return " ".join(t for _, _, t in out), out


def transcribe_audio_file(path: Path):
    """Transcribe an audio/video file directly. Returns list of
    (start, end, text). Works around the faster-whisper/PyAV
    metadata_errors incompatibility on old installs."""
    model = _whisper()
    try:
        segments, _ = model.transcribe(str(path), beam_size=5)
    except TypeError:
        import av
        _orig = av.open

        def _patched(*a, **k):
            k.pop("metadata_errors", None)
            return _orig(*a, **k)
        av.open = _patched
        try:
            segments, _ = model.transcribe(str(path), beam_size=5)
        finally:
            av.open = _orig
    return [(round(s.start, 1), round(s.end, 1), s.text.strip())
            for s in segments if s.text.strip()]


def analyze_clip(pid, upload_path: Path):
    try:
        clip_text, clip_segments = transcribe_clip(upload_path)
    except Exception as e:
        return {"ok": False, "error": f"Could not transcribe the clip: {e}"}
    if not clip_text.strip():
        return {"ok": False,
                "error": "No speech detected in the clip. Scene Finder matches "
                         "clips by their spoken dialogue - a silent clip can't "
                         "be matched. Try describing the scene instead (Option B)."}

    videos = db.get_videos(pid)
    pairs = []
    for v in videos:
        segs = db.get_segments(v["id"])
        if segs:
            pairs.append((v, segs))
    ranked = align_against_playlist(clip_text, pairs)
    if not ranked:
        return {"ok": True, "clip_transcript": clip_text,
                "results": [], "alternatives": [],
                "note": "The clip's dialogue doesn't match any video in this "
                        "playlist. It may be from a different series, or the "
                        "videos may lack captions."}

    def to_result(score, video, a):
        lines = search_mod._nearby_lines(video["id"], a["offset"])
        conf = search_mod._confidence(score)
        reasons = [
            f"Clip dialogue matched {a['matched_words']} words "
            f"({a['coverage'] * 100:.0f}% of the clip's speech).",
            f"Matched line: \"{a['matched_text'][:140]}\"",
        ]
        return {
            "video_id": video["id"], "youtube_id": video["youtube_id"],
            "title": video["title"], "episode": video["position"],
            "timestamp": round(a["offset"], 1),
            "timestamp_hms": format_hms(a["offset"]),
            "url": video["url"],
            "url_at": watch_url_at(video["youtube_id"], a["offset"]),
            "thumbnail": video.get("thumbnail"),
            "confidence": conf, "score": round(score, 3),
            "method": "clip audio transcription + dialogue alignment",
            "reasons": reasons, "matched_lines": lines,
        }

    results = [to_result(s, v, a) for s, v, a in ranked[:1]]
    alternatives = [to_result(s, v, a) for s, v, a in ranked[1:4]]
    return {"ok": True, "clip_transcript": clip_text,
            "results": results, "alternatives": alternatives,
            "method": "clip audio transcription + dialogue alignment",
            "note": None}


# ---------------------------------------------------------- screenshots
def _dhash(img, size=9):
    from PIL import Image
    img = img.convert("L").resize((size, size - 1), Image.LANCZOS)
    px = list(img.getdata())
    bits = 0
    for r in range(size - 1):
        for c in range(size - 1):
            bits = (bits << 1) | (1 if px[r * size + c] > px[r * size + c + 1] else 0)
    return bits


def _hamming(a, b):
    return bin(a ^ b).count("1")


def _ocr_text(path: Path):
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        # upscale small images for better OCR
        if max(img.size) < 1200:
            img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        text = pytesseract.image_to_string(img).strip()
        return text
    except Exception:
        return ""


def _thumb_path(youtube_id: str):
    p = THUMB_DIR / f"{youtube_id}.jpg"
    return p if p.exists() else None


def ensure_thumbnails(videos):
    """Download (and cache) one thumbnail per video. Best effort."""
    for v in videos:
        p = THUMB_DIR / f"{v['youtube_id']}.jpg"
        if p.exists() or not v.get("thumbnail"):
            continue
        try:
            req = urllib.request.Request(v["thumbnail"],
                                         headers={"User-Agent": "SceneFinder/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r, open(p, "wb") as f:
                f.write(r.read())
        except Exception:
            continue


def analyze_screenshots(pid, image_paths):
    videos = db.get_videos(pid)
    ensure_thumbnails(videos)

    ocr_texts = []
    for p in image_paths:
        t = _ocr_text(p)
        if t:
            ocr_texts.append(t)
    ocr_combined = " ".join(ocr_texts)

    # visual similarity vs thumbnails (weak signal - thumbnails only)
    shot_hashes = []
    for p in image_paths:
        try:
            from PIL import Image
            shot_hashes.append(_dhash(Image.open(p)))
        except Exception:
            continue
    visual = []
    for v in videos:
        tp = _thumb_path(v["youtube_id"])
        if not tp or not shot_hashes:
            continue
        try:
            from PIL import Image
            th = _dhash(Image.open(tp))
        except Exception:
            continue
        dists = [_hamming(sh, th) for sh in shot_hashes]
        best = min(dists)
        # dhash is 64-bit; <10 is similar, <16 vaguely similar
        sim = max(0.0, 1 - best / 24)
        if sim > 0.25:
            visual.append((sim, v, best))

    visual.sort(key=lambda t: t[0], reverse=True)

    results, alternatives = [], []
    notes = []
    if ocr_combined.strip():
        kw = search_mod.search(pid, ocr_combined, top_k=5)
        for r in kw["results"] + kw["alternatives"]:
            r = dict(r)
            r["method"] = "on-screen text (OCR) match"
            r["reasons"] = [f"Text visible in your screenshot matched dialogue/captions: "
                            f"\"{ocr_combined[:120]}\""]
            (results if len(results) < 2 else alternatives).append(r)
    else:
        notes.append("No readable text found in the screenshot(s).")

    for sim, v, dist in visual[:3]:
        entry = {
            "video_id": v["id"], "youtube_id": v["youtube_id"],
            "title": v["title"], "episode": v["position"],
            "timestamp": 0.0, "timestamp_hms": "00:00:00",
            "url": v["url"], "url_at": v["url"],
            "thumbnail": v.get("thumbnail"),
            "confidence": "low", "score": round(sim * 0.5, 3),
            "method": "thumbnail visual similarity (weak)",
            "reasons": [f"Your screenshot vaguely resembles this video's thumbnail "
                        f"(visual distance {dist}/64). Thumbnails rarely match a "
                        f"specific scene - treat as a hint only."],
            "matched_lines": [],
        }
        (results if len(results) < 2 else alternatives).append(entry)

    notes.append(
        "Visual scene search is limited: matching an arbitrary frame against "
        "every frame of every video would require downloading the full "
        "playlist. For reliable results, upload a clip with dialogue (Option A) "
        "or describe the scene (Option B).")
    return {"ok": True, "ocr_text": ocr_combined or None,
            "results": results[:3], "alternatives": alternatives[:3],
            "method": "screenshot OCR + thumbnail similarity",
            "note": " ".join(notes)}
