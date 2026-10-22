"""Visual frame previews for the Exact Clip Finder.

Frames come from YouTube's own *storyboard* preview images - the same
public seek-preview thumbnails YouTube serves to every viewer. No video
is downloaded and no access control is bypassed: yt-dlp fetches the
storyboard sheets (sb0), and individual frames are cropped out of the
sheets locally with Pillow.

What this is and is not:
  * It returns REAL frames at real timestamps so the *user* can
    visually confirm the scene.
  * The app does NOT run a vision model and never claims to have
    "recognized" characters or actions in a frame. The frames are
    evidence for human eyes, labelled as such in the UI.
  * If storyboards are unavailable for a video (or the download or
    parse fails), it reports that honestly with a reason.

Downloads are cached per video under DATA_DIR/storyboards so repeated
checks are instant.
"""
import email
import io
import json
import re
import subprocess
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path

from .config import DATA_DIR
from .vtt import format_hms

_SB_DIR = DATA_DIR / "storyboards"
_SLIDE_RE = re.compile(
    r"Slide #\d+:\s*(\d+):(\d+):(\d+)[.,](\d+)\s*[–-]\s*"
    r"(\d+):(\d+):(\d+)[.,](\d+)")


def _hms_to_s(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _run(cmd, timeout=240):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout)


def _metadata_for(video_id, timeout, extra):
    """(duration, frame_w, frame_h) from ONE player client's
    yt-dlp -j storyboard info, or None."""
    proc = _run([sys.executable, "-m", "yt_dlp", "-j",
                 "--no-warnings",
                 f"https://www.youtube.com/watch?v={video_id}"]
                + list(extra), timeout=timeout)
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        info = json.loads(proc.stdout)
    except Exception:
        return None
    duration = info.get("duration")
    for f in info.get("formats", []):
        if f.get("format_id") == "sb0":
            return duration, f.get("width"), f.get("height")
    return None


# Player clients tried for storyboards, in order: (extractor-args,
# cache tag). Web first (its spec is the canonical one), android
# second (it keeps working from server IPs the web path is
# bot-walled on). Different clients return DIFFERENT storyboard
# specs (frame size, sheet layout, density), so a video's metadata
# and its sheets must come from the SAME client - mixing a web
# spec with android sheets made frames unlocatable (measured
# 2026-10-22: most episodes failed one way or another until the
# pairing was enforced; afterwards 28/36 fetched cleanly).
_CLIENTS = [((), ""),
            (("--extractor-args", "youtube:player_client=android"),
             "_and")]

# video_id -> the (extra, tag) whose metadata last succeeded, so
# _download_storyboard fetches the same client's sheets.
_WINNER = {}


def _metadata(video_id, timeout=120):
    """Storyboard metadata from the first player client that
    answers; the winner is remembered for _download_storyboard."""
    for extra, tag in _CLIENTS:
        meta = _metadata_for(video_id, timeout, extra)
        if meta:
            _WINNER[video_id] = (extra, tag)
            return meta
    return None


def _download_for(video_id, timeout, extra, tag):
    """Path of the cached .mhtml storyboard for ONE player
    client, downloading if needed."""
    _SB_DIR.mkdir(parents=True, exist_ok=True)
    path = _SB_DIR / f"{video_id}{tag}.mhtml"
    if path.exists() and path.stat().st_size > 10000:
        return path
    tmp = str(path) + ".part"
    proc = _run([sys.executable, "-m", "yt_dlp", "-f", "sb0",
                 "--no-warnings", "-o", tmp,
                 f"https://www.youtube.com/watch?v={video_id}"]
                + list(extra), timeout=timeout)
    p = Path(tmp)
    if proc.returncode == 0 and p.exists() and p.stat().st_size > 10000:
        p.replace(path)
        return path
    try:
        p.unlink()
    except OSError:
        pass
    return None


def _download_storyboard(video_id, timeout=240):
    """Path of the cached .mhtml storyboard, preferring the
    client whose metadata won for this video (same-client
    consistency), then any client that yields a file."""
    order = []
    if video_id in _WINNER:
        order.append(_WINNER[video_id])
    order += [c for c in _CLIENTS if c not in order]
    for extra, tag in order:
        path = _download_for(video_id, timeout, extra, tag)
        if path:
            return path
    return None


def _sheets_for(video_id):
    """(meta, sheets, ranges) with metadata and sheets fetched
    consistently from ONE player client, or None when neither
    client yields a usable, layout-consistent storyboard."""
    for extra, tag in _CLIENTS:
        meta = _metadata_for(video_id, 120, extra)
        if not meta:
            continue
        path = _download_for(video_id, 240, extra, tag)
        if not path:
            continue
        sheets, ranges = _parse_sheets(path)
        if not sheets:
            continue
        sheet_w, sheet_h = sheets[0].size
        if not meta[1] or not meta[2] or sheet_w % meta[1] \
                or sheet_h % meta[2]:
            continue
        return meta, sheets, ranges
    return None


def _parse_sheets(mhtml_path):
    """Return (sheets, slide_ranges): sheets = list of PIL Images,
    slide_ranges = list of (start_s, end_s) per sheet or None."""
    raw = Path(mhtml_path).read_bytes()
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    images, html_text = [], ""
    for part in msg.walk():
        ct = part.get_content_type()
        if ct.startswith("image/"):
            data = part.get_payload(decode=True)
            if data:
                from PIL import Image
                images.append(Image.open(io.BytesIO(data)).copy())
        elif ct == "text/html":
            try:
                html_text = part.get_content()
            except Exception:
                payload = part.get_payload(decode=True)
                html_text = payload.decode("utf-8", "replace") if payload else ""
    ranges = []
    for m in _SLIDE_RE.finditer(html_text or ""):
        g = m.groups()
        ranges.append((_hms_to_s(*g[:4]), _hms_to_s(*g[4:])))
    if len(ranges) != len(images):
        ranges = None
    return images, ranges


def _frames_at(video_id, targets, meta=None):
    """Crop real storyboard frames at the given timestamps.
    Shared by scene_frames / episode_frames. Never raises."""
    try:
        meta = meta or _metadata(video_id)
        if not meta:
            return {"available": False, "frames": [],
                    "reason": "YouTube did not offer storyboard "
                              "previews for this video."}
        duration, fw, fh = meta
        mhtml = _download_storyboard(video_id)
        if not mhtml:
            return {"available": False, "frames": [],
                    "reason": "Storyboard preview download failed "
                              "(network or YouTube restriction)."}
        sheets, ranges = _parse_sheets(mhtml)
        if not sheets:
            return {"available": False, "frames": [],
                    "reason": "Storyboard file could not be parsed."}
        sheet_w, sheet_h = sheets[0].size
        if not fw or not fh or sheet_w % fw or sheet_h % fh:
            # Metadata and sheets came from different player
            # clients (their storyboard specs differ) - retry
            # once with strictly paired fetches before giving up.
            got = _sheets_for(video_id)
            if not got:
                return {"available": False, "frames": [],
                        "reason": "Unexpected storyboard layout; "
                                  "frames cannot be located "
                                  "reliably."}
            (duration, fw, fh), sheets, ranges = got
            sheet_w, sheet_h = sheets[0].size
        cols, rows = sheet_w // fw, sheet_h // fh
        per_sheet = cols * rows

        def frame_at(t):
            # locate sheet + cell for timestamp t
            if ranges:
                idx = None
                for i, (s0, s1) in enumerate(ranges):
                    if s0 <= t <= s1:
                        idx = i
                        break
                if idx is None:
                    idx = len(ranges) - 1 if t > ranges[-1][1] else 0
                s0, s1 = ranges[idx]
            else:
                total = per_sheet * len(sheets)
                step = (duration or 0) / max(1, total)
                idx = min(len(sheets) - 1, int(t / max(1e-9, step * per_sheet)))
                s0, s1 = idx * step * per_sheet, (idx + 1) * step * per_sheet
            cell = int((t - s0) / max(1e-9, (s1 - s0)) * per_sheet)
            cell = max(0, min(per_sheet - 1, cell))
            cx, cy = (cell % cols) * fw, (cell // cols) * fh
            return sheets[idx].crop((cx, cy, cx + fw, cy + fh))

        out_dir = _SB_DIR / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        last = None
        for t in targets:
            if last is not None and abs(t - last) < 5:
                continue
            last = t
            img = frame_at(t)
            p = out_dir / f"{video_id}_{int(t)}.png"
            img.save(str(p))
            frames.append({"timestamp": round(float(t), 1),
                           "hms": format_hms(t), "path": str(p)})
        return {"available": True, "frames": frames,
                "reason": "Frames are YouTube storyboard previews "
                          "(public seek-preview images)."}
    except subprocess.TimeoutExpired:
        return {"available": False, "frames": [],
                "reason": "Storyboard download timed out."}
    except Exception as e:
        return {"available": False, "frames": [],
                "reason": f"Visual check failed: {type(e).__name__}: {e}"}


def boundary_frames(video_id, start, end):
    """Real storyboard frames straddling a scene range's edges:
    just before the start, just after it, just before the end, and
    just after it. Comparing across each edge shows whether a
    visual cut actually happens near the estimated boundary - the
    only frame evidence a storyboard can honestly offer (its
    granularity is seconds, not frames). Never raises."""
    targets = [max(0.0, start - 8.0), start + 2.0,
               max(start, end - 2.0), end + 8.0]
    return _frames_at(video_id, targets)


def scene_frames(video_id, start, end, max_frames=3):
    """Real frames at the start / middle / end of [start, end].

    Returns {"available": bool, "reason": str,
             "frames": [{"timestamp", "hms", "path"}]}."""
    targets = [start, (start + end) / 2.0, max(start, end - 1.0)]
    out = _frames_at(video_id, targets)
    out["frames"] = out["frames"][:max_frames]
    return out


def episode_frames(video_id, count=8):
    """Evenly spaced real frames across the WHOLE episode.

    The visual fallback for episodes with no transcript at all: the
    user can spot a scene with their own eyes and note its
    approximate time, even when no dialogue is indexed. Same honesty
    rules as scene_frames - real frames only, no recognition claims."""
    got = _metadata(video_id)
    if not got or not got[0]:
        return {"available": False, "frames": [],
                "reason": "YouTube did not offer storyboard previews "
                          "for this video."}
    duration = got[0]
    targets = [duration * (i + 0.5) / count for i in range(count)]
    return _frames_at(video_id, targets, got)
