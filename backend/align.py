"""Align a short clip transcript against a video's full transcript.

Finds where the clip's spoken words occur in the video, returning the video
timestamp and a score. Uses fuzzy window matching (difflib matching blocks),
so imperfect speech-to-text still matches - exact contiguous runs score
highest, scattered word overlap scores lower. Pure stdlib, fast.

Word tokenization is Unicode-aware: dialogue in Urdu, Arabic, Hindi,
Turkish, Korean, Chinese etc. scripts tokenizes into words, not into
nothing (the original [a-z0-9] pattern silently erased every
non-Latin script, which made original-language matching impossible
for most of the world's dramas).
"""
import re
from difflib import SequenceMatcher

WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)

STOPWORDS = frozenset("""
a an the and or but if then so of at by for with to in on as it its it's is are
was were be been being have has had do does did will would can could should
shall may might must this that these those i you he she we they them him her
my your his our their me us oh uh um ah hey hello hi no yes not very just
like come comes came get got going go went well now here there what when
where who why how which
""".split())


def normalize(text: str):
    return WORD.findall(text.lower())


def content_words(words):
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def build_word_index(segments):
    """Flatten segments into words with interpolated timestamps."""
    words, times = [], []
    for s in segments:
        ws = normalize(s["text"])
        if not ws:
            continue
        dur = max(0.1, s["end"] - s["start"])
        for j, w in enumerate(ws):
            words.append(w)
            times.append(s["start"] + dur * j / len(ws))
    return words, times


def _window_score(clip_words, window):
    sm = SequenceMatcher(None, clip_words, window, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / max(1, len(clip_words))


def fuzzy_align(clip_words, video_words, video_times):
    """Return (score, offset, matched_text) or None."""
    W = len(clip_words)
    if W < 4 or len(video_words) < 4:
        return None
    # cheap prefilter: need at least 2 shared content words
    clip_content = set(content_words(clip_words))
    if not clip_content:
        clip_content = set(clip_words)
    video_set = set(video_words)
    if len(clip_content & video_set) < 2:
        return None

    step = max(1, W // 4)
    limit = len(video_words) - min(W, len(video_words)) + 1
    best_score, best_i = 0.0, 0
    for i in range(0, max(1, limit), step):
        s = _window_score(clip_words, video_words[i:i + W])
        if s > best_score:
            best_score, best_i = s, i
    # refine around the best window at stride 1
    lo, hi = max(0, best_i - step), min(limit, best_i + step + 1)
    for i in range(lo, hi):
        s = _window_score(clip_words, video_words[i:i + W])
        if s > best_score:
            best_score, best_i = s, i

    if best_score < 0.22:
        return None
    # point the timestamp at the actual matched words (largest matching
    # block), not the start of the fuzzy window
    best_window = video_words[best_i:best_i + W]
    sm = SequenceMatcher(None, clip_words, best_window, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    matched_words = [clip_words[b.a + k] for b in blocks for k in range(b.size)]
    if len(matched_words) < 3 or len(content_words(matched_words)) < 2:
        return None
    biggest = max(blocks, key=lambda b: b.size)
    # Anchor at the earliest SUBSTANTIAL block, not the single
    # longest run: a clip's start maps to where its first matched
    # words occur (the longest run can sit mid-scene and pull the
    # timestamp late, e.g. +12s on a real Urdu clip).
    threshold = max(3, biggest.size // 2)
    _substantial = [b for b in blocks if b.size >= threshold]
    # When every matching block is short (fragmented matches across
    # two noisy transcripts), no block reaches the threshold and
    # min() on the empty set crashed the whole clip search
    # (found 2026-10-21 against a real clip). Fall back to the
    # biggest block itself - it is the strongest anchor available.
    first_big = min(_substantial, key=lambda b: b.b) \
        if _substantial else biggest
    offset = video_times[best_i + first_big.b]
    span = min(W, 30)
    return {
        "score": best_score,
        "offset": offset,
        "clip_anchor": first_big.a,
        "matched_words": len(matched_words),
        "clip_words": W,
        "coverage": best_score,
        "matched_text": " ".join(video_words[best_i:best_i + span]),
    }


def align_against_playlist(clip_text, videos_segments):
    """videos_segments: list of (video, segments). Returns ranked
    [(score, video, alignment), ...]."""
    clip_words = normalize(clip_text)
    out = []
    for video, segments in videos_segments:
        words, times = build_word_index(segments)
        a = fuzzy_align(clip_words, words, times)
        if a:
            out.append((a["score"], video, a))
    out.sort(key=lambda t: t[0], reverse=True)
    return out
