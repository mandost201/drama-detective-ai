"""Parse WebVTT / SRT subtitle text into (start, end, text) segments."""
import re

TS = re.compile(r"(\d+):(\d+):(\d+)[.,](\d+)")
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def ts_to_seconds(ts: str) -> float:
    m = TS.search(ts)
    if not m:
        return 0.0
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000.0


def clean_text(t: str) -> str:
    t = TAG.sub(" ", t)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    t = WS.sub(" ", t).strip()
    return t


def parse_vtt(raw: str):
    """Return list of (start, end, text). Merges duplicate auto-caption lines."""
    cues = []
    lines = raw.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            parts = line.split("-->")
            start = ts_to_seconds(parts[0])
            end = ts_to_seconds(parts[1]) if len(parts) > 1 else start + 2
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                # skip cue identifiers that look like timestamps alone
                text_lines.append(lines[i])
                i += 1
            text = clean_text(" ".join(text_lines))
            if text:
                cues.append((start, end, text))
        else:
            i += 1
    return _merge_repeats(cues)


def parse_srv3_json(data: str):
    """Fallback parser for YouTube srv3 JSON captions (not currently used)."""
    return []


def _merge_repeats(cues):
    """Auto captions repeat lines across cues; keep new words only."""
    merged = []
    prev_words = []
    for start, end, text in cues:
        words = text.split()
        # drop leading words identical to previous cue tail
        k = 0
        maxk = min(len(words), len(prev_words))
        while k < maxk and words[k] == prev_words[k]:
            k += 1
        new_words = words[k:]
        if new_words:
            if merged and start - merged[-1][1] < 1.5 and k >= 2:
                # extend previous cue with genuinely new words
                ps, pe, pt = merged[-1]
                merged[-1] = (ps, end, (pt + " " + " ".join(new_words)).strip())
            else:
                merged.append((start, end, " ".join(words)))
        prev_words = words
    # drop empty / tiny junk
    return [(s, e, t) for s, e, t in merged if len(t) >= 2]


def format_hms(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
