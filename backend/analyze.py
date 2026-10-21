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


# ------------------------------------------------- universal clip finder
def _whisper_transcribe(path, task="transcribe"):
    """(segments, language, language_probability) for a media file.
    Language is auto-detected by Whisper; the original-language text
    is preserved (task='transcribe'). task='translate' renders the
    same speech in English."""
    model = _whisper()
    try:
        segments, info = model.transcribe(str(path), beam_size=5, task=task)
    except TypeError:
        import av
        _orig = av.open

        def _patched(*a, **k):
            k.pop("metadata_errors", None)
            return _orig(*a, **k)
        av.open = _patched
        try:
            segments, info = model.transcribe(str(path), beam_size=5,
                                              task=task)
        finally:
            av.open = _orig
    segs = [(round(s.start, 1), round(s.end, 1), s.text.strip())
            for s in segments if s.text.strip()]
    return segs, getattr(info, "language", None), \
        getattr(info, "language_probability", None)


def _clip_frames(path, count=3):
    """Extract up to `count` frames from a clip with PyAV (works
    where ffmpeg is unavailable, e.g. Streamlit Cloud). Returns a
    list of PIL Images; empty on any failure."""
    try:
        import av
        from PIL import Image  # noqa: F401  (ensures PIL present)
        frames = []
        with av.open(str(path)) as container:
            if not container.streams.video:
                return []
            stream = container.streams.video[0]
            total = (container.duration or 0) / 1e6
            fracs = [0.15, 0.5, 0.85][:count]
            want = [total * f for f in fracs] if total else []
            for frame in container.decode(stream):
                t = frame.time or 0.0
                if want and len(frames) < len(want) and t >= want[len(frames)]:
                    frames.append(frame.to_image())
                if len(frames) >= count:
                    break
            if not frames:  # duration unknown: take the first frame
                for frame in container.decode(stream):
                    frames.append(frame.to_image())
                    break
        return frames
    except Exception as e:
        print(f"[analyze] clip frame extraction failed: {e}")
        return []


def _visual_compare(clip_hashes, frames_result):
    """Compare clip frame hashes against a visual.scene_frames /
    episode_frames result. Coarse supporting signal ONLY: storyboard
    previews are sparse, low-res seek images, so 'similar' supports
    a dialogue match but never proves one by itself."""
    if not clip_hashes:
        return {"status": "unavailable", "best_distance": None,
                "detail": "Clip se frames nahi nikal sake."}
    if not frames_result or not frames_result.get("available"):
        reason = (frames_result or {}).get(
            "reason", "Storyboard frames available nahi.")
        return {"status": "unavailable", "best_distance": None,
                "detail": f"Source frames available nahi: {reason}"}
    from PIL import Image
    cand = []
    for f in frames_result["frames"]:
        try:
            cand.append(_dhash(Image.open(f["path"])))
        except Exception:
            continue
    if not cand:
        return {"status": "unavailable", "best_distance": None,
                "detail": "Source frames parhi nahi ja sakeen."}
    best = min(_hamming(ch, vh) for ch in clip_hashes for vh in cand)
    # dHash is 64-bit: <=10 near-identical, <=18 similar-looking
    status = "similar" if best <= 18 else "different"
    return {"status": status, "best_distance": best,
            "detail": f"Qareeb tareen frame distance {best}/64 "
                      "(storyboard previews sparse hote hain — ye "
                      "sirf supporting signal hai, saboot nahi)."}


def _visual_verify(upload_path, youtube_id, start, end):
    """Frame-check one dialogue candidate against its source video."""
    try:
        from . import visual as visual_mod
        hashes = [_dhash(im) for im in _clip_frames(upload_path)]
        frames = visual_mod.scene_frames(youtube_id, start, end)
        return _visual_compare(hashes, frames)
    except Exception as e:
        return {"status": "unavailable", "best_distance": None,
                "detail": f"Visual check fail: {type(e).__name__}: {e}"}


def _alignment_verdict(score, matched_words):
    """Evidence tier for a dialogue alignment. A similarity score
    alone is never verification - the tier also demands enough
    actually-matched words:

      verified : score >= 0.55 and >= 10 matched words. Clean,
                 extensive dialogue evidence for this specific clip
                 and source segment.
      probable : score >= 0.28 and >= 8 matched words. Real but
                 imperfect evidence (e.g. a loose speech-to-text
                 translation) - the right episode/scene is likely,
                 but a human must confirm with the matched line.
                 A very short clip can also reach probable - never
                 verified - when almost all of its few words match
                 (score >= 0.75 and >= 5 matched words): the phrase
                 is distinctive, but there is too little material
                 to verify an exact segment.
      None     : below the probable floor. Merely similar - must
                 NOT be presented as a match at all."""
    if score >= 0.55 and matched_words >= 10:
        return "verified"
    if score >= 0.28 and matched_words >= 8:
        return "probable"
    if score >= 0.75 and matched_words >= 5:
        return "probable"
    return None


def _clip_result(cand, clip_dur):
    v, a, score = cand["video"], cand["a"], cand["score"]
    start = float(a["offset"])
    end = start + max(2.0, clip_dur)
    lines = search_mod._nearby_lines(v["id"], start)
    reasons = [
        f"Clip dialogue matched {a['matched_words']} words "
        f"({a['coverage'] * 100:.0f}% of the clip's speech) — "
        f"{cand['method']}.",
        f"Matched line: \"{a['matched_text'][:140]}\"",
    ]
    if a["coverage"] < 0.6:
        reasons.append(
            "Clip ka sirf aik hissa match hua — clip edited/montage "
            "ho sakta hai; diya gaya time usi matched hisse ka hai.")
    return {
        "video_id": v["id"], "youtube_id": v["youtube_id"],
        "title": v["title"], "episode": v["position"],
        "timestamp": round(start, 1), "timestamp_hms": format_hms(start),
        "scene_start": round(start, 1), "scene_end": round(end, 1),
        "scene_start_hms": format_hms(start),
        "scene_end_hms": format_hms(end),
        "url": v["url"], "url_at": watch_url_at(v["youtube_id"], start),
        "thumbnail": v.get("thumbnail"),
        "confidence": search_mod._confidence(score),
        "verdict": _alignment_verdict(score, a["matched_words"]),
        "score": round(score, 3), "method": cand["method"],
        "reasons": reasons, "matched_lines": lines,
        "visual": {"status": "not_checked", "best_distance": None,
                   "detail": "Sirf top candidate ki frames check "
                             "hoti hain."},
    }


def _visual_only_scan(pid, upload_path, lang, prob, coverage):
    """A clip with no detectable speech: rank episodes by frame
    similarity against their storyboard previews. These are HINTS
    at low confidence - never verified matches."""
    hashes = [_dhash(im) for im in _clip_frames(upload_path)]
    if not hashes:
        return {"ok": False,
                "error": "Clip mein na dialogue mila na frames nikal "
                         "sake — is clip ko match nahi kiya ja sakta. "
                         "Scene ko alfaz mein describe karke try "
                         "karein (Step 2)."}
    from . import visual as visual_mod
    scored = []
    for v in db.get_videos(pid)[:40]:  # bounded: storyboard downloads
        ef = visual_mod.episode_frames(v["youtube_id"], count=6)
        if not ef.get("available"):
            continue
        cmp = _visual_compare(hashes, ef)
        if cmp.get("best_distance") is not None:
            scored.append((cmp["best_distance"], v, cmp))
    scored.sort(key=lambda t: t[0])
    results = []
    for dist, v, cmp in scored[:3]:
        if dist > 22:  # too different to even hint at
            continue
        results.append({
            "video_id": v["id"], "youtube_id": v["youtube_id"],
            "title": v["title"], "episode": v["position"],
            "timestamp": 0.0, "timestamp_hms": "00:00:00",
            "scene_start": 0.0, "scene_end": 0.0,
            "scene_start_hms": "00:00:00", "scene_end_hms": "00:00:00",
            "url": v["url"], "url_at": v["url"],
            "thumbnail": v.get("thumbnail"),
            "confidence": "low",
            "score": round(max(0.0, 1 - dist / 24) * 0.5, 3),
            "method": "visual frame similarity (storyboard previews)",
            "reasons": [
                f"Clip ke frames is episode ke storyboard frames se "
                f"kuch miltay jultay hain (distance {dist}/64).",
                "Ye sirf aik HINT hai — dialogue saboot ke baghair "
                "ye verified match NAHI hai."],
            "matched_lines": [], "visual": cmp,
        })
    return {"ok": True, "language": lang, "language_probability": prob,
            "clip_transcript": "", "clip_transcript_en": None,
            "results": results, "coverage": coverage,
            "verdict": "not_verified",
            "method": "visual-only scan (no speech in clip)",
            "note": ("Clip mein koi dialogue nahi mila — neeche sirf "
                     "visual hints hain, verified match nahi."
                     if results else
                     "No verified match in this playlist.")}


def analyze_clip_full(pid, upload_path, transcribe_fn=None,
                      visual_check=True):
    """Universal clip pipeline (any drama, any language):

    1. Transcribe the clip in its ORIGINAL language (auto-detected)
       and, when it isn't English, also translate it to English.
    2. Align BOTH texts against every indexed episode of THIS
       playlist only (Unicode-aware word alignment), then fall back
       to semantic search with the clip text.
    3. Visually verify the top dialogue candidate against real
       storyboard frames of its source video (separate status).
    4. A speech-less clip goes to a bounded visual-only scan whose
       results are labelled hints, never verified matches.
    Unindexed episodes are never pretended to be searched - the
    returned coverage says exactly how many were."""
    tf = transcribe_fn or _whisper_transcribe
    try:
        segs_o, lang, prob = tf(upload_path, task="transcribe")
    except Exception as e:
        return {"ok": False,
                "error": f"Could not transcribe the clip: {e}"}
    text_o = " ".join(t for _, _, t in segs_o)
    text_en, segs_en = "", []
    if lang and lang != "en" and text_o.strip():
        try:
            segs_en, _l, _p = tf(upload_path, task="translate")
            text_en = " ".join(t for _, _, t in segs_en)
        except Exception:
            segs_en, text_en = [], ""
    coverage = search_mod.playlist_coverage(pid)
    if not text_o.strip() and not text_en.strip():
        return _visual_only_scan(pid, upload_path, lang, prob, coverage)

    pairs = []
    for v in db.get_videos(pid):
        segs = db.get_segments(v["id"])
        if segs:
            pairs.append((v, segs))
    cand = {}

    def _add(score, video, a, method):
        cur = cand.get(video["id"])
        if cur is None or score > cur["score"]:
            cand[video["id"]] = {"score": score, "video": video,
                                 "a": a, "method": method}
    if text_o.strip():
        for s, v, a in align_against_playlist(text_o, pairs):
            _add(s, v, a, "dialogue alignment (original language)")
    if text_en.strip():
        for s, v, a in align_against_playlist(text_en, pairs):
            _add(s, v, a, "dialogue alignment (translated to English)")
    ranked = sorted(cand.values(), key=lambda c: c["score"],
                    reverse=True)
    clip_dur = 0.0
    for segs in (segs_o, segs_en):
        if segs:
            clip_dur = max(clip_dur, segs[-1][1])
    # Evidence gating: an alignment below the probable floor is
    # merely similar, not a match - it is suppressed entirely and
    # the verdict becomes "not verified" instead of dressing the
    # scene up as a weak candidate.
    kept = [c for c in ranked
            if _alignment_verdict(c["score"], c["a"]["matched_words"])]
    verdict = None
    if kept:
        verdict = _alignment_verdict(kept[0]["score"],
                                     kept[0]["a"]["matched_words"])
    results = [_clip_result(c, clip_dur) for c in kept[:3]]

    if not results:  # semantic fallback with the clip's own words
        for q in (text_en, text_o):
            if len(q.split()) < 12:
                # Too little speech to be evidence: a handful of
                # generic words can score deceptively high in
                # keyword search (0.87 for "please come here right
                # now okay") without identifying any scene.
                continue
            sr = search_mod.search(pid, q, top_k=3)
            for r in sr.get("results", []):
                # A clip transcript is a long, specific query: a weak
                # keyword/semantic hit on it is noise, not a
                # candidate. Only medium-or-better evidence is
                # promoted; anything less stays "no verified match".
                if (r.get("score") or 0) < 0.5:
                    continue
                r = dict(r)
                r["method"] = "semantic search (clip transcript)"
                r.setdefault("visual", {
                    "status": "not_checked", "best_distance": None,
                    "detail": "Semantic candidate — frames check "
                              "nahi hua."})
                results.append(r)
            if results:
                # Semantic similarity alone can never verify an
                # exact clip - at most it is a probable candidate.
                verdict = "probable"
                break

    if results and kept and visual_check:
        top = results[0]
        top["visual"] = _visual_verify(upload_path,
                                       top["youtube_id"],
                                       top["scene_start"],
                                       top["scene_end"])
    return {"ok": True, "language": lang, "language_probability": prob,
            "clip_transcript": text_o,
            "clip_transcript_en": text_en or None,
            "results": results, "coverage": coverage,
            "verdict": verdict or "not_verified",
            "method": "clip transcription (original + translate) + "
                      "dialogue alignment",
            "note": None if results else
                    "No verified match in this playlist."}


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
