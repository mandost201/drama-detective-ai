"""YouTube access: playlist metadata + captions.

Two metadata backends:
  1. YouTube Data API v3 (if YOUTUBE_API_KEY is set) - official, quota based.
  2. yt-dlp flat playlist extraction (no key needed) - fallback default.

Captions: yt-dlp subtitle download first (manual then auto), falling back to
youtube-transcript-api. Everything is read-only and respects access
restrictions: private / deleted / region-blocked videos are skipped, never
pretended.
"""
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .config import REQUEST_DELAY, YOUTUBE_API_KEY
from .vtt import parse_vtt

# Always run yt-dlp from this same Python environment (it may not be on PATH).
YTDLP = [sys.executable, "-m", "yt_dlp"]

PLAYLIST_RE = re.compile(r"[?&]list=([A-Za-z0-9_-]+)")
VIDEO_RES = [
    re.compile(r"[?&]v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"/(?:shorts|live|embed)/([A-Za-z0-9_-]{11})"),
]


def extract_playlist_id(url: str):
    m = PLAYLIST_RE.search(url or "")
    return m.group(1) if m else None


def extract_video_id(url: str):
    """Video ID from a watch / youtu.be / shorts / live URL, else None."""
    for rx in VIDEO_RES:
        m = rx.search(url or "")
        if m:
            return m.group(1)
    return None


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def watch_url_at(video_id: str, seconds: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"


# ---------------------------------------------------------------- metadata
def fetch_playlist(url: str):
    """Return (title, [video dicts]). Raises RuntimeError on failure.

    Accepts a playlist URL, or a single video URL (watch / youtu.be /
    shorts), which is treated as a one-episode collection."""
    pid = extract_playlist_id(url)
    if not pid:
        if extract_video_id(url):
            _title, videos = _fetch_via_ytdlp(url)
            if videos:
                return videos[0]["title"], videos[:1]
        raise RuntimeError("That doesn't look like a YouTube playlist or "
                           "video URL (expected a 'list=' or 'v=' parameter).")
    if YOUTUBE_API_KEY:
        try:
            return _fetch_via_api(pid, url)
        except Exception as e:  # fall back to yt-dlp
            print(f"[youtube] Data API failed ({e}), falling back to yt-dlp")
    return _fetch_via_ytdlp(url)


def _api_get(path, params):
    params = dict(params, key=YOUTUBE_API_KEY)
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"https://www.googleapis.com/youtube/v3/{path}?{qs}",
                                 headers={"User-Agent": "SceneFinder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _fetch_via_api(pid, url):
    pl = _api_get("playlists", {"part": "snippet", "id": pid})
    items = pl.get("items", [])
    if not items:
        raise RuntimeError("Playlist not found or not accessible via the API.")
    title = items[0]["snippet"]["title"]
    videos, page, pos = [], None, 0
    while True:
        res = _api_get("playlistItems", {"part": "snippet,contentDetails",
                                         "playlistId": pid, "maxResults": 50,
                                         "pageToken": page or ""})
        for it in res.get("items", []):
            vid = it["contentDetails"]["videoId"]
            pos += 1
            videos.append({
                "youtube_id": vid,
                "title": it["snippet"]["title"],
                "position": pos,
                "duration": None,
                "url": watch_url(vid),
                "thumbnail": (it["snippet"]["thumbnails"].get("medium") or
                              it["snippet"]["thumbnails"].get("default") or {}).get("url"),
            })
        page = res.get("nextPageToken")
        if not page:
            break
        time.sleep(REQUEST_DELAY)
    return title, videos


def _fetch_via_ytdlp(url):
    cmd = YTDLP + ["--flat-playlist", "--dump-json", "--no-warnings",
           "--socket-timeout", "20", url]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    videos, title, pos = [], None, 0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("_type") == "playlist":
            title = title or e.get("title")
            continue
        vid = e.get("id")
        if not vid or e.get("_type") not in (None, "video", "url"):
            continue
        if not title:
            title = e.get("playlist_title") or e.get("playlist") or title
        pos += 1
        thumbs = e.get("thumbnails") or []
        thumb = thumbs[-1].get("url") if thumbs else None
        videos.append({
            "youtube_id": vid,
            "title": e.get("title") or f"Video {vid}",
            "position": pos,
            "duration": e.get("duration"),
            "url": watch_url(vid),
            "thumbnail": thumb,
        })
    if not videos:
        err = (proc.stderr or "").strip().splitlines()
        hint = err[-1] if err else "no videos found"
        raise RuntimeError(f"Could not read that playlist: {hint}. "
                           "It may be private, deleted, or region-restricted.")
    return title or "YouTube playlist", videos


# ---------------------------------------------------------------- captions
# Language preference for caption tracks. The app originally fetched
# ENGLISH ONLY ("--sub-langs en.*"), so any episode whose YouTube
# captions were Urdu/Hindi/other was falsely reported as having "no
# captions" and excluded from search. Now the languages the app's
# users actually search in are tried first (English, then Urdu, then
# Hindi), and after that ANY language YouTube offers - a transcript
# in any language is searchable and beats no transcript at all.
_LANG_PREF = ("en", "ur", "hi")


def _track_lang(filename, video_id):
    """'YDu9akegEHU.en-US.vtt' -> 'en-US' (the part between the
    video id and the extension)."""
    name = filename[:-4] if filename.endswith(".vtt") else filename
    prefix = (video_id or "") + "."
    if prefix != "." and name.startswith(prefix):
        return name[len(prefix):]
    return name


def _lang_rank(lang):
    base = (lang or "").split("-")[0].lower()
    for i, p in enumerate(_LANG_PREF):
        if base == p:
            return i
    return len(_LANG_PREF)


def _pick_track(files, video_id):
    """Best downloaded .vtt by language preference (en > ur > hi >
    anything else), deterministic by filename after that."""
    if not files:
        return None
    return sorted(files,
                  key=lambda f: (_lang_rank(_track_lang(f.name, video_id)),
                                 f.name))[0]


def fetch_transcript(video_id: str):
    """Return list of (start,end,text) or None if unavailable/restricted."""
    segs, _lang = fetch_transcript_meta(video_id)
    return segs


def fetch_transcript_meta(video_id: str):
    """(segments, caption_language) or (None, None). The language
    is the actual track that was used (e.g. 'en', 'ur', 'hi', or a
    probed other language) - provenance the caller records with the
    transcript. The API fallback cannot report a language."""
    segs, lang, _reason = fetch_transcript_meta_diag(video_id)
    return segs, lang


def fetch_transcript_meta_diag(video_id: str):
    """Like fetch_transcript_meta, but on failure also returns WHY:
    (segments, language, None) or (None, None, reason). The reason
    distinguishes "this video has no captions at all" from "captions
    exist but YouTube blocked/throttled the download" - the two
    failures look identical from a bare None, and only the second
    is worth retrying (2026-10-22: live analyses silently lost
    different episodes per run to exactly this). Goes through the
    long-standing _transcript_ytdlp_meta / _transcript_api seams
    (existing tests patch them); the implementations record their
    last error per video for classification."""
    segs, lang = _transcript_ytdlp_meta(video_id)
    if segs:
        return segs, lang, None
    try:
        asegs = _transcript_api(video_id)
    except Exception:
        asegs = None
    if asegs:
        return asegs, None, None
    return None, None, _classify_fetch_failure(
        _LAST_YTDLP_ERR.get(video_id), _LAST_API_ERR.get(video_id),
        video_id)


# Last fetch error per video, recorded by the real implementations
# below - the raw material for failure classification.
_LAST_YTDLP_ERR = {}
_LAST_API_ERR = {}


def last_failure_reason(video_id: str):
    """The classified reason the most recent caption fetch for
    this video failed, or None when no fetch error was recorded
    (e.g. the fetch path was substituted in tests)."""
    yerr = _LAST_YTDLP_ERR.get(video_id)
    aerr = _LAST_API_ERR.get(video_id)
    if yerr is None and aerr is None:
        return None
    return _classify_fetch_failure(yerr, aerr, video_id)


def _classify_fetch_failure(ytdlp_err, api_err, video_id):
    blob = f"{ytdlp_err or ''} {api_err or ''}"
    if "Sign in to confirm" in blob or "not a bot" in blob:
        return ("YouTube bot-check: captions mojood hain lekin "
                "server IP se download roka gaya.")
    if "429" in blob or "Too Many Requests" in blob:
        return ("YouTube rate limit (HTTP 429): bohat zyada "
                "requests par aarzi rok — dobara koshish se aam "
                "tor par theek ho jata hai.")
    if "IpBlocked" in blob or "RequestBlocked" in blob:
        return ("YouTube ne is server ki IP se transcript ki "
                "request block ki.")
    if not _has_any_caption_tracks(video_id):
        return ("Is video par YouTube captions mojood hi nahi — "
                "Whisper transcription hi wahid rasta hai.")
    tail = (ytdlp_err or api_err or "").strip()[-140:]
    if tail:
        return f"Captions download fail hua ({tail})"
    return "Captions download fail hua (wajah ghair wazeh)."


def _has_any_caption_tracks(video_id):
    """True when --list-subs shows at least one caption track (any
    language, manual or automatic) for the video."""
    try:
        cmd = YTDLP + ["--skip-download", "--list-subs",
                       "--no-warnings", "--socket-timeout", "20",
                       "--extractor-args",
                       "youtube:player_client=android",
                       watch_url(video_id)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=90)
        for line in (proc.stdout or "").splitlines():
            m = _OTHER_LANG_ROW.match(line.strip())
            if m and m.group(1).lower() != "language":
                return True
    except Exception:
        pass
    return False


def _download_subs(video_id, sub_langs, tmp, timeout=180):
    """Download subtitle tracks into tmp. The android player client
    goes first: YouTube bot-walls the web player path from server
    IPs ("Sign in to confirm you're not a bot") while the android
    client - the same route download_audio uses - keeps working.
    Falls back to the default client when android yields nothing.
    Returns (files, last_error_tail): the error tail of the final
    attempt is kept so a failure can be CLASSIFIED (bot-check vs
    rate limit vs no tracks) instead of silently dropped."""
    base = YTDLP + ["--skip-download", "--no-warnings",
                    "--write-subs", "--write-auto-subs",
                    "--sub-langs", sub_langs, "--sub-format", "vtt",
                    "--socket-timeout", "20",
                    "-o", str(tmp / "%(id)s.%(ext)s")]
    last_err = ""
    for extra in (["--extractor-args", "youtube:player_client=android"],
                  []):
        try:
            proc = subprocess.run(base + extra + [watch_url(video_id)],
                                  capture_output=True, text=True,
                                  timeout=timeout)
            last_err = (proc.stderr or proc.stdout or "")[-400:]
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        files = sorted(tmp.glob("*.vtt"))
        if files:
            return files, last_err
    return [], last_err


def _transcript_ytdlp(video_id: str):
    segs, _lang = _transcript_ytdlp_meta(video_id)
    return segs


def _transcript_ytdlp_meta(video_id: str):
    segs, lang, _err = _transcript_ytdlp_diag(video_id)
    return segs, lang


def _transcript_ytdlp_diag(video_id: str):
    """yt-dlp subtitle download with one retry: YouTube answers
    bursts of subtitle requests with HTTP 429 (transient), and a
    single retry after a pause rescues otherwise-good episodes.
    Returns (segments, track_language, error_tail) - the error
    tail is None on success."""
    err = None
    for attempt in range(2):
        segs, lang, err = _transcript_ytdlp_once(video_id)
        if segs:
            _LAST_YTDLP_ERR[video_id] = None
            return segs, lang, None
        if attempt == 0:
            time.sleep(20)
    _LAST_YTDLP_ERR[video_id] = err
    return None, None, err


_OTHER_LANG_ROW = re.compile(r"^([a-zA-Z][a-zA-Z-]{1,14})\s{2,}\S")


def _probe_other_language(video_id):
    """One cheap --list-subs request: the best caption language
    OUTSIDE en/ur/hi that YouTube offers for this video, or None.
    (Downloading 'all' tracks instead would fire dozens of requests
    per video and trip YouTube's rate limit.)"""
    try:
        cmd = YTDLP + ["--skip-download", "--list-subs", "--no-warnings",
                       "--socket-timeout", "20", watch_url(video_id)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=90)
        langs = []
        for line in (proc.stdout or "").splitlines():
            m = _OTHER_LANG_ROW.match(line.strip())
            if m and m.group(1).lower() not in ("language",):
                langs.append(m.group(1))
        rest = [l for l in dict.fromkeys(langs)
                if l.split("-")[0].lower() not in _LANG_PREF]
        return sorted(rest)[0] if rest else None
    except Exception:
        return None


def _transcript_ytdlp_once(video_id: str):
    tmp = Path(tempfile.mkdtemp(prefix="sf_sub_"))
    try:
        # Pass 1: English / Urdu / Hindi tracks (manual or auto),
        # best language wins. This is the pass that rescues episodes
        # whose only captions are not in English.
        files, err = _download_subs(video_id, "en.*,ur.*,hi.*", tmp)
        best = _pick_track(files, video_id)
        if best is not None:
            try:
                segs = parse_vtt(best.read_text(encoding="utf-8",
                                               errors="ignore"))
            except Exception:
                segs = None
            if segs:
                return segs, _track_lang(best.name, video_id), None
        # Pass 2: ONE other language, chosen from the track list -
        # any transcript beats no transcript, without a request storm.
        lang = _probe_other_language(video_id)
        if lang:
            files, err2 = _download_subs(video_id, lang, tmp)
            if err2:
                err = err2
            best = _pick_track(files, video_id)
            if best is not None:
                try:
                    segs = parse_vtt(best.read_text(encoding="utf-8",
                                                   errors="ignore"))
                except Exception:
                    segs = None
                if segs:
                    return segs, _track_lang(best.name, video_id), None
        return None, None, err
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    finally:
        for f in tmp.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass


def _transcript_api(video_id: str):
    segs, _err = _transcript_api_diag(video_id)
    return segs


def _transcript_api_diag(video_id: str):
    """(segments, error) via youtube-transcript-api; the error
    names the exception (e.g. IpBlocked) so the failure can be
    classified instead of swallowed."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception as e:
        _LAST_API_ERR[video_id] = f"{type(e).__name__}: {e}"
        return None, _LAST_API_ERR[video_id]
    api = YouTubeTranscriptApi()
    last_err = None
    # try manual english first, then any english auto-generated
    tr = None
    for langs in (["en"], ["en-US", "en-GB"]):
        try:
            tr = api.fetch(video_id, languages=langs)
            break
        except Exception as e:
            last_err = e
    if tr is None:
        # last resort: whatever transcript exists
        try:
            lst = api.list(video_id)
            t = next(iter(lst), None)
            tr = t.fetch() if t else None
        except Exception as e:
            last_err = e
    if not tr:
        if last_err is not None:
            _LAST_API_ERR[video_id] = \
                f"{type(last_err).__name__}: {last_err}"
            return None, _LAST_API_ERR[video_id]
        _LAST_API_ERR[video_id] = "no transcript via API"
        return None, _LAST_API_ERR[video_id]
    segs = [(s.start, s.start + s.duration, s.text.replace("\n", " ").strip())
            for s in tr if s.text.strip()]
    _LAST_API_ERR[video_id] = None
    return (segs or None), None


def download_audio(video_id: str, out_path):
    """Download best audio track as mp3. Returns the mp3 Path.
    Raises RuntimeError on failure."""
    from pathlib import Path
    out_path = Path(out_path)
    stem = out_path.with_suffix("")  # yt-dlp adds .%(ext)s itself
    final = stem.with_suffix(".mp3")
    # Player-client fallbacks: a 403 on one route (the live Ep16
    # backfill died on a bare "HTTP Error 403" from the android
    # route) is often route-specific - the media URL each client
    # receives differs, so another client may succeed.
    last_err = ""
    for extra in (["--extractor-args", "youtube:player_client=android"],
                  ["--extractor-args", "youtube:player_client=ios"],
                  []):
        cmd = YTDLP + ["-f", "ba/b", "-x", "--audio-format", "mp3",
                       "--no-playlist", "--no-warnings",
                       "-o", str(stem) + ".%(ext)s",
                       f"https://www.youtube.com/watch?v={video_id}"] + extra
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=900)
        # clean up any non-mp3 leftovers from -x
        for p in stem.parent.glob(stem.name + ".*"):
            if p != final:
                try:
                    p.unlink()
                except OSError:
                    pass
        if final.exists() and final.stat().st_size >= 50000:
            return final
        last_err = (proc.stderr or proc.stdout or "")[-300:]
    raise RuntimeError(
        f"audio download failed for {video_id}: {last_err}")


def download_audio_raw(video_id: str, out_stem):
    """Download the best audio track WITHOUT conversion (no ffmpeg
    on Streamlit Cloud): the file lands as .m4a/.webm/.opus next to
    `out_stem` and is decoded later with PyAV. Same player-client
    fallbacks as download_audio. Returns the Path; raises
    RuntimeError on failure."""
    from pathlib import Path
    out_stem = Path(out_stem)
    last_err = ""
    for extra in (["--extractor-args", "youtube:player_client=android"],
                  ["--extractor-args", "youtube:player_client=ios"],
                  []):
        cmd = YTDLP + ["-f", "ba/b[height<=144]/b",
                       "--no-playlist", "--no-warnings",
                       "-o", str(out_stem) + ".%(ext)s",
                       f"https://www.youtube.com/watch?v={video_id}"] \
            + extra
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=600)
        cands = sorted(out_stem.parent.glob(out_stem.name + ".*"),
                       key=lambda p: p.stat().st_size, reverse=True)
        for p in cands:
            if p.stat().st_size >= 50000:
                return p
        last_err = (proc.stderr or proc.stdout or "")[-300:]
    raise RuntimeError(
        f"raw audio download failed for {video_id}: {last_err}")
