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
    segs = _transcript_ytdlp(video_id)
    if segs:
        return segs
    try:
        return _transcript_api(video_id)
    except Exception:
        return None


def _download_subs(video_id, sub_langs, tmp, timeout=180):
    """Download subtitle tracks into tmp. The android player client
    goes first: YouTube bot-walls the web player path from server
    IPs ("Sign in to confirm you're not a bot") while the android
    client - the same route download_audio uses - keeps working.
    Falls back to the default client when android yields nothing."""
    base = YTDLP + ["--skip-download", "--no-warnings",
                    "--write-subs", "--write-auto-subs",
                    "--sub-langs", sub_langs, "--sub-format", "vtt",
                    "--socket-timeout", "20",
                    "-o", str(tmp / "%(id)s.%(ext)s")]
    for extra in (["--extractor-args", "youtube:player_client=android"],
                  []):
        try:
            subprocess.run(base + extra + [watch_url(video_id)],
                           capture_output=True, text=True, timeout=timeout)
        except Exception:
            pass
        files = sorted(tmp.glob("*.vtt"))
        if files:
            return files
    return []


def _transcript_ytdlp(video_id: str):
    """yt-dlp subtitle download with one retry: YouTube answers
    bursts of subtitle requests with HTTP 429 (transient), and a
    single retry after a pause rescues otherwise-good episodes."""
    for attempt in range(2):
        segs = _transcript_ytdlp_once(video_id)
        if segs:
            return segs
        if attempt == 0:
            time.sleep(20)
    return None


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
        files = _download_subs(video_id, "en.*,ur.*,hi.*", tmp)
        best = _pick_track(files, video_id)
        if best is not None:
            try:
                segs = parse_vtt(best.read_text(encoding="utf-8",
                                               errors="ignore"))
            except Exception:
                segs = None
            if segs:
                return segs
        # Pass 2: ONE other language, chosen from the track list -
        # any transcript beats no transcript, without a request storm.
        lang = _probe_other_language(video_id)
        if lang:
            files = _download_subs(video_id, lang, tmp)
            best = _pick_track(files, video_id)
            if best is not None:
                try:
                    segs = parse_vtt(best.read_text(encoding="utf-8",
                                                   errors="ignore"))
                except Exception:
                    segs = None
                if segs:
                    return segs
        return None
    except Exception:
        return None
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
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    # try manual english first, then any english auto-generated
    for langs in (["en"], ["en-US", "en-GB"]):
        try:
            tr = api.fetch(video_id, languages=langs)
            break
        except Exception:
            tr = None
    else:
        tr = None
    if tr is None:
        # last resort: whatever transcript exists
        try:
            lst = api.list(video_id)
            t = next(iter(lst), None)
            tr = t.fetch() if t else None
        except Exception:
            tr = None
    if not tr:
        return None
    segs = [(s.start, s.start + s.duration, s.text.replace("\n", " ").strip())
            for s in tr if s.text.strip()]
    return segs or None


def download_audio(video_id: str, out_path):
    """Download best audio track as mp3. Returns the mp3 Path.
    Raises RuntimeError on failure."""
    from pathlib import Path
    out_path = Path(out_path)
    stem = out_path.with_suffix("")  # yt-dlp adds .%(ext)s itself
    cmd = YTDLP + ["-f", "ba/b", "-x", "--audio-format", "mp3",
                   "--no-playlist", "--no-warnings",
                   "--extractor-args", "youtube:player_client=android",
                   "-o", str(stem) + ".%(ext)s",
                   f"https://www.youtube.com/watch?v={video_id}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    final = stem.with_suffix(".mp3")
    # clean up any non-mp3 leftovers from -x
    for p in stem.parent.glob(stem.name + ".*"):
        if p != final:
            try:
                p.unlink()
            except OSError:
                pass
    if not final.exists() or final.stat().st_size < 50000:
        raise RuntimeError(
            f"audio download failed for {video_id}: "
            f"{(proc.stderr or proc.stdout)[-300:]}")
    return final
