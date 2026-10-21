"""Scene search over an analyzed playlist.

Two honest signals, blended:
  * keyword  - SQLite FTS5 BM25 over transcript segments (exact dialogue)
  * semantic - cosine similarity of query vs segment embeddings (meaning)

Queries may be English, Roman Urdu, Urdu or Hindi. Roman Urdu / Urdu /
Hindi queries are expanded with the English words official subtitles
typically use (see backend.normalize), so "vakil sahab dulha dulhan aa
gaye" can match the subtitle "Lawyer, the groom and bride are here".

A third guard, distinctive-term coverage, keeps generic words ("dies",
"death", "scene") from producing high-confidence false positives: a result
only keeps a high score when the query's distinctive terms (names, places,
specific actions) actually appear in the matched scene's text - directly,
via a known spelling variant, or via a concept expansion.
Results map to episode (playlist position), timestamp, and a deep link.
Confidence is derived from the actual scores and labelled plainly.
"""
import re

import numpy as np

from . import db
from . import embeddings
from . import normalize
from .vtt import format_hms
from .youtube import watch_url_at

# Words too generic to identify a scene on their own. Everything else in the
# query is "distinctive" and must show up in the matched scene for a high
# confidence score. Covers English + Roman Urdu + Urdu script + Devanagari.
_GENERIC_TERMS = frozenset("""
dies die death dead killing kill killed kills murdered murder suicide
scene scenes seen episode episodes moment moments part parts clip video
where when what which who whom whose how this that these those there here
no will would can could shall should may might must for your yours you we
it its his her their them he she him do does did
the a an in on of to and or is are was were be been has have had
ki ka ke ko se ne mein me main hai hain tha thi they thay
wala wale wali yeh ye woh wo ek aik aur lekin par per bhi
sahab sahib janab aap tum mera meri mere uska uski iska iski apna apni
kuch koi sab log bahut bohat abhi kabhi phir agar magar kyun kyu kaise
kahan idhar udhar ho aa gaye gaya gayi sakti sakta sakte hota hoti raha
rahi karna karne liya liye diye hua hui hue chuka chuki nahi aayega
aayegi jayega jayegi jate jata jati hoga hogi honge karega karegi milega
milegi aaya aayi
ہے ہیں کا کی کے کو میں نے سے اور یہ وہ بھی پر تھا تھی ایک اس ان جب تو ہی نہیں کیا ہو گئی گیا گئے
है हैं का की के को में ने से और यह वह भी पर था थी एक इस जब तो ही नहीं क्या हो गई गया गये
""".split())


def _skeleton(s):
    """Consonant skeleton: catches transliteration variants
    (suleman ~ suleiman ~ suleyman -> slmn)."""
    return "".join(c for c in s.lower() if c.isalpha() and c not in "aeiou")


def distinctive_terms(query):
    """The query's identifying terms: names, places, specific things.
    Tokenization is script-aware (Latin / Urdu / Devanagari); Roman Urdu
    terms are canonicalized so spelling variants count once."""
    terms = []
    for t in normalize.tokens(query):
        if t in _GENERIC_TERMS or len(t) <= 1:
            continue
        terms.append(normalize.canonical(t) if t.isascii() else t)
    return list(dict.fromkeys(terms))


def _term_in_text(term, text):
    # Multilingual match: the term itself, a known spelling variant, an
    # English concept expansion ("shaadi" -> "marriage"), or a strict
    # fuzzy hit for long names. See backend.normalize.term_matches.
    return normalize.term_matches(term, text or "")


def coverage_factor(distinctive, text):
    """Return (factor, hits). factor in [0.35, 1.0]: 1.0 when every
    distinctive term appears in the text, down to 0.35 when none do."""
    if not distinctive:
        return 1.0, 0
    hits = sum(1 for t in distinctive if _term_in_text(t, text or ""))
    cov = hits / len(distinctive)
    return 0.35 + 0.65 * cov, hits

# in-memory cache of playlist embeddings: pid -> dict
_emb_cache = {}


def _playlist_embeddings(pid):
    if pid in _emb_cache:
        return _emb_cache[pid]
    rows = db.load_embeddings(pid)
    cache = {"seg_ids": [], "video_ids": [], "starts": [], "matrix": None}
    if rows:
        mat = np.stack([np.frombuffer(blob, dtype=np.float32)
                        for _, _, _, _, blob in rows])
        cache = {"seg_ids": [r[0] for r in rows],
                 "video_ids": [r[1] for r in rows],
                 "starts": [r[2] for r in rows],
                 "matrix": mat}
    _emb_cache[pid] = cache
    return cache


def invalidate_cache(pid):
    _emb_cache.pop(pid, None)


def _videos_by_id(pid):
    return {v["id"]: v for v in db.get_videos(pid)}


def _confidence(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _nearby_lines(video_id, center_start, n=2):
    segs = db.get_segments(video_id)
    if not segs:
        return []
    idx = min(range(len(segs)), key=lambda i: abs(segs[i]["start"] - center_start))
    lo, hi = max(0, idx - n), min(len(segs), idx + n + 1)
    return [{"start": s["start"], "hms": format_hms(s["start"]), "text": s["text"]}
            for s in segs[lo:hi]]


def _scene_text(video_id, center_start, window=60.0):
    """All transcript text within +/-window seconds of a match.

    A described scene spans a minute or two of dialogue, while a single
    subtitle segment is one line. Distinctive-term coverage is therefore
    measured against the scene neighborhood, not one line - otherwise a
    faithful multi-line description could never reach full coverage."""
    segs = db.get_segments(video_id)
    return " ".join(s["text"] for s in segs
                    if abs(s["start"] - center_start) <= window)


def playlist_coverage(pid):
    """How much of the playlist can actually be searched:
    {"total", "searchable", "missing", "pct"}. A video counts as
    searchable only when it has an indexed transcript. Episodes
    without captions are reported - never silently skipped."""
    vids = db.get_videos(pid)
    total = len(vids)
    searchable = sum(1 for v in vids if v["has_transcript"])
    return {"total": total, "searchable": searchable,
            "missing": total - searchable,
            "pct": round(100.0 * searchable / total, 1) if total else 0.0}


def scene_block(video_id, anchor_start, gap_threshold=20.0):
    """The FULL contiguous dialogue block containing the anchor
    segment: expand while the silence between consecutive caption
    segments stays <= gap_threshold. No focus trim, no span cap -
    this is the transcript's own upper bound for "the complete
    scene". When an episode's dialogue is dense throughout, the
    block can span most of the episode; that is reported as-is,
    because it is exactly what the transcript can and cannot
    separate. Returns (start, end) in seconds."""
    segs = db.get_segments(video_id)
    if not segs:
        return anchor_start, anchor_start
    idx = min(range(len(segs)),
              key=lambda i: abs(segs[i]["start"] - anchor_start))
    lo = idx
    while lo > 0 and segs[lo]["start"] - segs[lo - 1]["end"] <= gap_threshold:
        lo -= 1
    hi = idx
    while (hi < len(segs) - 1
           and segs[hi + 1]["start"] - segs[hi]["end"] <= gap_threshold):
        hi += 1
    return round(float(segs[lo]["start"]), 1), round(float(segs[hi]["end"]), 1)


def scene_range(video_id, anchor_start, terms=None, gap_threshold=20.0,
                focus_window=90.0, lead=10.0, max_span=300.0):
    """Estimate the scene's (start, end) around the anchor segment.

    This is an ESTIMATE, labelled as such in the UI - never an exact
    cut. Real subtitle tracks have pauses inside a scene (entrances,
    reaction shots), so the method works on the anchor's dialogue
    block rather than single lines:
      1. Dialogue block: expand from the anchor while the silence
         between consecutive caption segments stays <= gap_threshold
         (20s - smaller gaps turned out to split single scenes).
      2. Focus trim: inside that block and within +/-focus_window of
         the anchor, take the outermost segments matching the user's
         distinctive terms, plus a short lead-in/out, so unrelated
         dialogue sharing the block does not inflate the range.
      3. Cap the span at max_span around the anchor.
    """
    segs = db.get_segments(video_id)
    if not segs:
        return anchor_start, anchor_start
    idx = min(range(len(segs)),
              key=lambda i: abs(segs[i]["start"] - anchor_start))
    lo = idx
    while lo > 0 and segs[lo]["start"] - segs[lo - 1]["end"] <= gap_threshold:
        lo -= 1
    hi = idx
    while (hi < len(segs) - 1
           and segs[hi + 1]["start"] - segs[hi]["end"] <= gap_threshold):
        hi += 1
    start, end = segs[lo]["start"], segs[hi]["end"]
    if terms:
        core = [i for i in range(lo, hi + 1)
                if abs(segs[i]["start"] - anchor_start) <= focus_window
                and any(_term_in_text(t, segs[i]["text"]) for t in terms)]
        if core:
            c_lo, c_hi = core[0], core[-1]
            j = c_lo
            while (j - 1 >= lo
                   and segs[j]["start"] - segs[j - 1]["end"] <= gap_threshold
                   and segs[c_lo]["start"] - segs[j - 1]["start"] <= lead):
                j -= 1
            c_lo = j
            j = c_hi
            while (j + 1 <= hi
                   and segs[j + 1]["start"] - segs[j]["end"] <= gap_threshold
                   and segs[j + 1]["end"] - segs[c_hi]["end"] <= lead):
                j += 1
            c_hi = j
            start, end = segs[c_lo]["start"], segs[c_hi]["end"]
    if end - start > max_span:
        start = max(start, anchor_start - max_span / 2.0)
        end = min(end, anchor_start + max_span / 2.0)
    return round(float(start), 1), round(float(end), 1)


def _build_result(video, start, score, method, matched_lines, reasons):
    return {
        "video_id": video["id"],
        "youtube_id": video["youtube_id"],
        "title": video["title"],
        "episode": video["position"],
        "timestamp": round(start, 1),
        "timestamp_hms": format_hms(start),
        "url": video["url"],
        "url_at": watch_url_at(video["youtube_id"], start),
        "thumbnail": video.get("thumbnail"),
        "confidence": _confidence(score),
        "score": round(float(score), 3),
        "method": method,
        "reasons": reasons,
        "matched_lines": matched_lines,
    }


def keyword_search(pid, query, top_k=8):
    # FTS runs on the expanded query: the user's own words first, then the
    # English subtitle words their Roman Urdu / Urdu / Hindi terms map to.
    hits = db.fts_search(pid, normalize.expand_query(query), limit=300)
    if not hits:
        return [], "keyword"
    videos = _videos_by_id(pid)
    dist = distinctive_terms(query)
    # best segment per video (fts rank is negative BM25: more negative = better)
    best = {}
    for h in hits:
        vid = h["video_id"]
        key = -h["rank"]
        if vid not in best or key > best[vid][0]:
            best[vid] = (key, h)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    top_score = ranked[0][1][0] if ranked else 1.0
    results = []
    for vid, (key, h) in ranked[:top_k]:
        video = videos.get(vid)
        if not video:
            continue
        score = min(1.0, key / max(top_score, 1e-9))
        # boost exact-phrase hits
        phrase = query.strip().lower()
        if phrase and phrase in h["text"].lower():
            score = min(1.0, score + 0.15)
        lines = _nearby_lines(vid, h["start"])
        cov, cov_hits = coverage_factor(dist, _scene_text(vid, h["start"]))
        reasons = [f"Dialogue match: \"{h['text'][:120]}\""]
        if dist and cov < 1.0:
            reasons.append(
                f"Only {cov_hits}/{len(dist)} distinctive "
                f"term(s) ({', '.join(dist)}) appear in this scene's dialogue")
        r = _build_result(video, h["start"], score, "keyword", lines, reasons)
        r["cov_factor"] = round(cov, 3)
        results.append(r)
    return results, "keyword"


def semantic_search(pid, query, top_k=8):
    if not embeddings.available():
        return [], "semantic-unavailable"
    cache = _playlist_embeddings(pid)
    if cache["matrix"] is None:
        return [], "semantic-empty"
    dist = distinctive_terms(query)
    q = embeddings.embed([query])[0]
    sims = cache["matrix"] @ q  # cosine (normalized)
    order = np.argsort(-sims)
    videos = _videos_by_id(pid)
    seen, results = set(), []
    for idx in order:
        if sims[idx] < 0.32:
            break
        vid = int(cache["video_ids"][idx])
        if vid in seen:
            continue
        seen.add(vid)
        video = videos.get(vid)
        if not video:
            continue
        start = float(cache["starts"][idx])
        lines = _nearby_lines(vid, start)
        context = _scene_text(vid, start) or " ".join(l["text"] for l in lines)
        cov, cov_hits = coverage_factor(dist, context)
        reasons = ["Meaning/semantic match to your description"]
        if dist and cov < 1.0:
            reasons.append(
                f"Only {cov_hits}/{len(dist)} distinctive "
                f"term(s) ({', '.join(dist)}) appear near this scene")
        r = _build_result(video, start, float(sims[idx]), "semantic",
                          lines, reasons)
        r["cov_factor"] = round(cov, 3)
        results.append(r)
        if len(results) >= top_k:
            break
    return results, "semantic"


def search(pid, query, mode="auto", top_k=8):
    """Combined search. Returns dict with results, alternatives,
    method info.

    A query written in Urdu or Devanagari script whose own tokens
    match nothing (names have no concept-map entries) is retried
    with its Latin transliterations - the first candidate that
    produces results wins, and the result records which query
    actually ran ("effective_query") so nothing is hidden."""
    res = _search_impl(pid, query, mode=mode, top_k=top_k)
    if res["results"]:
        return res
    for cand in normalize.latin_candidates(query):
        alt = _search_impl(pid, cand, mode=mode, top_k=top_k)
        if alt["results"]:
            alt["effective_query"] = cand
            return alt
    return res


def _search_impl(pid, query, mode="auto", top_k=8):
    query = (query or "").strip()
    if not query:
        return {"results": [], "alternatives": [], "method": "none",
                "note": "Empty query."}
    kw, kw_method = keyword_search(pid, query, top_k=top_k)
    sem, sem_method = semantic_search(pid, query, top_k=top_k)

    merged = {}
    # Blend weights express relative trust when BOTH signals exist. When
    # only keyword matching produced results (e.g. the semantic model is
    # unavailable), the keyword signal defines the scale on its own -
    # otherwise every keyword-only result is capped near "low" no matter
    # how exact the dialogue match is. The distinctive-term coverage
    # guard below still decides what may be called high confidence.
    kw_weight = 0.55 if sem else 1.0
    for r in kw:
        merged[r["video_id"]] = dict(r, score=r["score"] * kw_weight,
                                     method="keyword")
    for r in sem:
        if r["video_id"] in merged:
            m = merged[r["video_id"]]
            m["score"] = min(1.0, m["score"] + r["score"] * 0.45)
            m["method"] = "keyword+semantic"
            # keep the stronger distinctive-term coverage of the two signals
            m["cov_factor"] = max(m.get("cov_factor", 1.0),
                                  r.get("cov_factor", 1.0))
            if r["score"] > 0.5:
                m["reasons"].append("Also a strong semantic match to the description")
        else:
            merged[r["video_id"]] = dict(r, score=r["score"] * 0.9)
    # Distinctive-term coverage guard: generic words ("dies", "death", "scene")
    # must not produce high-confidence false positives on their own.
    for m in merged.values():
        cov = m.get("cov_factor", 1.0)
        if cov < 1.0:
            m["score"] = m["score"] * cov
    ranked = sorted(merged.values(), key=lambda r: r["score"], reverse=True)
    # Recompute distinctive-term coverage directly on each result's scene
    # (the transcript around the matched timestamp). This is bulletproof:
    # it uses the actual dialogue of the scene shown to the user, not a
    # merged factor from different match signals.
    _dist = distinctive_terms(query)
    for r in ranked:
        conf = _confidence(r["score"])
        # Hard cap: if the user named distinctive terms (names etc.) and
        # none of them appear in the scene, it can never be high confidence
        # no matter how strong the generic-word match is. Partial coverage
        # caps at medium.
        _rtext = _scene_text(r["video_id"], r["timestamp"]) or " ".join(
            (l.get("text", "") if isinstance(l, dict) else str(l))
            for l in r.get("matched_lines", []))
        _cov, _ = coverage_factor(_dist, _rtext)
        if _dist and _cov <= 0.36:  # floor value = zero terms matched
            conf = "low"
        elif _dist and _cov < 1.0 and conf == "high":
            conf = "medium"
        r["confidence"] = conf
        r["cov_factor"] = round(_cov, 3)
        r["score"] = round(r["score"], 3)

    # ---- Exact Clip Finder: scene ranges + playlist coverage ----------
    # Every result gets an estimated scene start/end (dialogue-block
    # expansion + focus trim, see scene_range), and the main watch link
    # opens at the scene start. Confidence is capped when playlist
    # coverage is poor: with most episodes unsearchable, the true
    # scene may sit in an episode that could not be checked at all.
    cov = playlist_coverage(pid)
    for r in ranked:
        # Primary scene range = the COMPLETE dialogue block around
        # the match (span-capped) - the scene estimate. The
        # focus-trimmed window around the distinctive terms is kept
        # separately as the relevant core; it used to BE the scene
        # range, which understated real scenes (an 11-second window
        # for a 90-second scene on the ground-truth fixture).
        rs, re_ = scene_range(r["video_id"], r["timestamp"], None)
        r["scene_start"], r["scene_end"] = rs, re_
        r["scene_start_hms"] = format_hms(rs)
        r["scene_end_hms"] = format_hms(re_)
        fs, fe = scene_range(r["video_id"], r["timestamp"], _dist)
        r["focus_start"], r["focus_end"] = fs, fe
        r["focus_start_hms"] = format_hms(fs)
        r["focus_end_hms"] = format_hms(fe)
        # The complete dialogue block around the match - the
        # transcript's upper bound for the scene, reported alongside
        # the focus-trimmed estimate so neither number pretends to
        # be an exact cut.
        bs, be = scene_block(r["video_id"], r["timestamp"])
        r["block_start"], r["block_end"] = bs, be
        r["block_start_hms"] = format_hms(bs)
        r["block_end_hms"] = format_hms(be)
        r["matched_at"] = r["timestamp"]
        r["matched_at_hms"] = r["timestamp_hms"]
        # Boundary provenance: a search result's range comes from
        # the transcript alone unless/until the frame check runs
        # (the clip finder runs it for its top result; the UI also
        # offers it on demand). Never label it frame-verified here.
        r["boundary"] = {
            "kind": "transcript_estimate",
            "detail": "Range transcript ke dialogue block se "
                      "andarza hai — frames se verify nahi hua."}
        r["url_at"] = watch_url_at(r["youtube_id"], rs)
        r["match_type"] = "transcript"
        if cov["pct"] < 50.0 and r["confidence"] == "high":
            r["confidence"] = "medium"
            r["reasons"].append(
                f"Confidence capped: only {cov['searchable']} of "
                f"{cov['total']} episodes in this playlist are "
                "searchable (the rest have no usable captions), so the "
                "scene could be in an episode that was not checked.")

    results = ranked[:3]
    alternatives = ranked[3:3 + max(0, top_k - 3)]

    methods = []
    if kw:
        methods.append("dialogue keyword match")
    if sem and sem_method == "semantic":
        methods.append("semantic meaning match")
    if not methods:
        if sem_method == "semantic-unavailable":
            methods.append("keyword only (semantic model unavailable)")
        elif sem_method == "semantic-empty":
            methods.append("keyword only (no embeddings indexed)")
    method_note = " + ".join(methods) if methods else "no matches"

    note = None
    if not ranked:
        note = ("No matching scenes found in this playlist's transcripts. "
                "Try fewer/different words, or check that the videos have "
                "captions (videos without captions can't be searched).")
    elif ranked[0].get("cov_factor", 1.0) <= 0.36:
        dist = distinctive_terms(query)
        note = ("No scene mentions the distinctive terms in your description"
                + (f" ({', '.join(dist)})" if dist else "")
                + ". The matches below are weak — try different words or "
                  "check the spelling of names.")
    return {"results": results, "alternatives": alternatives,
            "method": method_note, "note": note,
            "coverage": cov,
            "semantic_available": embeddings.available()}
