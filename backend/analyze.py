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
    segments, _ = model.transcribe(audio, beam_size=5, **_DECODE_KW)
    out = [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
    return " ".join(t for _, _, t in out), out


# Deterministic decoding for EVERY transcription in this app.
# Measured 2026-10-21: faster-whisper's DEFAULT decoding uses
# temperature-fallback SAMPLING - when a segment trips the
# logprob/compression thresholds (common on noisy fan-edit clips
# with background music), it re-decodes at rising temperatures and
# the SAME clip produced wildly different transcripts run to run
# (clean English one run, repetition-loop gibberish the next), so a
# clip from the selected drama intermittently returned "No verified
# match". temperature=0.0 disables fallback sampling and
# condition_on_previous_text=False stops repetition loops from
# snowballing: with both pinned, the ground-truth clip produced the
# IDENTICAL transcript and verdict (probable, score 0.313) on 3/3
# runs, and a full episode indexed by the tiny model matched at
# 0.557. This changes no evidence threshold - it stops the input
# evidence itself from randomly degrading.
_DECODE_KW = dict(temperature=0.0, condition_on_previous_text=False)


def transcribe_audio_file(path: Path):
    """Transcribe an audio/video file directly. Returns list of
    (start, end, text). Works around the faster-whisper/PyAV
    metadata_errors incompatibility on old installs."""
    segs, _lang, _prob = transcribe_audio_file_meta(path)
    return segs


def transcribe_audio_file_meta(path: Path):
    """(segments, detected_language, language_probability) for an
    audio/video file. faster-whisper's detection info is exactly
    the provenance a transcript cache must keep: which language the
    speech was heard as, and how confident the detector was. A low
    probability is data, not an error - callers record it and warn,
    they do not silently trust or silently discard the transcript."""
    model = _whisper()
    try:
        segments, info = model.transcribe(str(path), beam_size=5, **_DECODE_KW)
    except TypeError:
        import av
        _orig = av.open

        def _patched(*a, **k):
            k.pop("metadata_errors", None)
            return _orig(*a, **k)
        av.open = _patched
        try:
            segments, info = model.transcribe(str(path), beam_size=5, **_DECODE_KW)
        finally:
            av.open = _orig
    rows = [(round(s.start, 1), round(s.end, 1), s.text.strip(),
             getattr(s, "no_speech_prob", 0.0))
            for s in segments if s.text.strip()]
    segs = [(a, b, t) for a, b, t, _p in rows]
    if rows and all(p > 0.6 for _a, _b, _t, p in rows):
        # faster-whisper hallucinates on silence/near-silence (a
        # lone "You" at no_speech_prob 0.68 on a pure-silence file,
        # measured 2026-10-21). When EVERY segment is more likely
        # no-speech than speech - 0.6 is Whisper's own
        # no_speech_threshold - the honest result is "no speech",
        # and the caller's empty-transcript path reports exactly
        # that instead of indexing an invented line as a transcript.
        return [], getattr(info, "language", None), \
            getattr(info, "language_probability", None)
    return segs, getattr(info, "language", None), \
        getattr(info, "language_probability", None)


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
_NAMED_MODELS = {}


def _whisper_named(name):
    """A Whisper model loaded by name (separate from the configured
    default). Used by the clip rescue path, which re-transcribes a
    failed clip with the stronger 'small' model."""
    if name not in _NAMED_MODELS:
        from faster_whisper import WhisperModel
        _NAMED_MODELS[name] = WhisperModel(name, device="cpu",
                                           compute_type="int8")
        print(f"[analyze] whisper model '{name}' loaded (named)")
    return _NAMED_MODELS[name]


def _whisper_transcribe(path, task="transcribe", model_name=None):
    """(segments, language, language_probability) for a media file.
    Language is auto-detected by Whisper; the original-language text
    is preserved (task='transcribe'). task='translate' renders the
    same speech in English."""
    model = _whisper_named(model_name) if model_name else _whisper()
    try:
        segments, info = model.transcribe(str(path), beam_size=5, task=task, **_DECODE_KW)
    except TypeError:
        import av
        _orig = av.open

        def _patched(*a, **k):
            k.pop("metadata_errors", None)
            return _orig(*a, **k)
        av.open = _patched
        try:
            segments, info = model.transcribe(str(path), beam_size=5,
                                              task=task, **_DECODE_KW)
        finally:
            av.open = _orig
    segs = [(round(s.start, 1), round(s.end, 1), s.text.strip())
            for s in segments if s.text.strip()]
    return segs, getattr(info, "language", None), \
        getattr(info, "language_probability", None)


def _chunked_transcribe(path, chunk_s=12.0, model_name="small"):
    """Transcribe a clip in SHORT chunks; (segments, languages).

    A whole-file decode of a long music-under-dialogue clip can
    fall into fluent confabulation (measured on the live app,
    2026-10-22: an Urdu clip "heard" as invented English). Short
    chunks reset the decoder context every chunk and re-run
    language detection per chunk, so real dialogue - including
    chunks in the playlist's own script - can survive where the
    whole-file decode invented text. Segment times are absolute.
    Chunks that hear nothing contribute nothing; nothing is
    invented here either. Never raises."""
    import tempfile
    import wave
    sr = 16000
    try:
        import av
        from av.audio.resampler import AudioResampler
        resampler = AudioResampler(format="s16", layout="mono",
                                   rate=sr)
        samples = bytearray()
        with av.open(str(path)) as container:
            if not container.streams.audio:
                return [], []
            for frame in container.decode(container.streams.audio[0]):
                for pkt in resampler.resample(frame):
                    samples.extend(pkt.to_ndarray().tobytes())
    except Exception:
        return [], []
    total_s = len(samples) / 2.0 / sr
    if total_s < 1.0:
        return [], []
    segs_all, langs = [], []
    n_chunks = int(total_s // chunk_s) + (1 if total_s % chunk_s
                                          else 0)
    with tempfile.TemporaryDirectory() as td:
        for i in range(n_chunks):
            start_s = i * chunk_s
            a = int(start_s * sr) * 2
            b = int(min(total_s, start_s + chunk_s) * sr) * 2
            if b - a < sr:      # < 0.5s tail is not evidence
                continue
            wp = Path(td) / f"chunk{i}.wav"
            try:
                with wave.open(str(wp), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sr)
                    w.writeframes(bytes(samples[a:b]))
                segs, lang, _p = _whisper_transcribe(
                    wp, model_name=model_name)
            except Exception:
                continue
            if lang and lang not in langs:
                langs.append(lang)
            for s0, s1, text in segs:
                segs_all.append((round(start_s + s0, 1),
                                 round(start_s + s1, 1), text))
    segs_all.sort(key=lambda s: s[0])
    return segs_all, langs


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


def _diversity(text):
    """Unique-token ratio of a transcript. Real dialogue - even a
    short scene - is lexically diverse (>= ~0.3). A Whisper
    repetition-loop hallucination ("you are the one who is the one
    who is...") collapses to a handful of unique tokens no matter
    how long it runs. Returns None for texts too short to judge."""
    toks = text.split()
    if len(toks) < 20:
        return None
    return round(len(set(toks)) / len(toks), 3)


def _repeated_ngram_coverage(text, n=4):
    """Share of a text's n-gram positions whose n-gram occurs at
    least twice - i.e. how much of the text is repeated material.
    Natural dialogue repeats almost no 4-word sequence; a
    formulaic transcript (a religious formula or song line
    recited over and over) is mostly repetition."""
    toks = text.split()
    if len(toks) < n + 4:
        return 0.0
    from collections import Counter
    grams = Counter(tuple(toks[i:i + n])
                    for i in range(len(toks) - n + 1))
    total = sum(grams.values())
    repeated = sum(c for c in grams.values() if c >= 2)
    return repeated / max(1, total)


def _pass_is_weak(text):
    """A transcription pass that cannot carry scene evidence:
    either a long repetition loop (diversity collapse) or a
    formulaic text dominated by one repeated phrase. Measured
    (2026-10-21): a wailing/death clip's translate pass was 29
    words - "I swear to God, there is no god but God" twice plus
    "I love you, my son" - diverse by ratio (0.52) yet pure
    formula; it aligned 'probable' into its own episode and
    semantically into an UNRELATED drama, because such phrases
    occur in countless scenes. Weak passes are excluded from
    matching; the rescue path (chunked, per-segment) is where
    noisy-but-real speech gets its chance instead."""
    if len(text.split()) < 15:
        return False
    div = _diversity(text)
    if div is not None and div < 0.15:
        return True
    return _repeated_ngram_coverage(text) >= 0.5


def _collapse_loops(text, run=6, keep=2):
    """Collapse long same-token runs (Whisper repetition artifacts)
    in an alignment QUERY: a run of >= `run` identical tokens keeps
    `keep` copies. The real words around the artifact survive; the
    poison that dilutes sequence alignment is removed. (Measured:
    one real clip's transcript went from 126 tokens - over half a
    single repeated token - to 55 real words.) Display transcripts
    are never altered; this only cleans matching queries."""
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        j = i
        while j < len(toks) and toks[j] == toks[i]:
            j += 1
        out.extend([toks[i]] * (keep if j - i >= run else j - i))
        i = j
    return " ".join(out)


def _rescue_alignment(esc_segs, pairs, clip_dur):
    """Chunked, script-bridged alignment - the rescue path for a
    clip whose primary transcription produced no match.

    Why it exists (measured on a real user's clip, 2026-10-21): the
    episode captions were Hindi auto-captions in Devanagari script;
    the clip model heard the Urdu dialogue in Urdu script at 57%
    confidence, full of errors and a repetition chunk - whole-text
    token alignment found literally nothing in ANY script space.
    A stronger model heard the same dialogue largely in Devanagari,
    but code-switched (half English), so whole-text alignment still
    diluted below the floor. What worked: aligning the clip in
    PIECES (per segment + the non-Latin chunk), with both sides
    transliterated into a common Latin space, and letting two
    INDEPENDENT pieces of the clip corroborate one scene.

    Verdict discipline is unchanged: a single chunk must meet the
    normal _alignment_verdict floors (plus a mass rule against
    saturated short-chunk scores); otherwise at least two
    distinct clip segments must independently match the same
    episode within 120s, with >= 10 matched words in total, the
    strongest chunk >= 0.35, every aggregated chunk >= 0.30, and
    >= 2 DISTINCTIVE matched words (words rare across the
    playlist - generic dramatic vocabulary repeated by several
    chunks is not corroboration; measured false aggregation
    2026-10-22) - and that aggregation is capped at probable.
    Returns (result, verdict, info) or None."""
    from . import normalize as norm_mod
    lat_pairs = [(v, [{"start": s["start"], "end": s["end"],
                       "text": norm_mod.transliterate(s["text"])}
                      for s in segs]) for v, segs in pairs]
    queries, seen = [], set()

    def _add_q(text, space, label, src=None):
        t = (text or "").strip()
        if len(t.split()) < 4 or t in seen:
            return
        seen.add(t)
        queries.append((t, space, label, src))

    full_raw = " ".join(t for _, _, t in esc_segs)
    _add_q(_collapse_loops(norm_mod.transliterate(full_raw)),
           "lat", "full (script-bridged)")
    _add_q(_collapse_loops(full_raw), "raw", "full")
    nonlatin = []
    for idx, (_a, _b, t) in enumerate(esc_segs):
        if sum(1 for ch in t if ord(ch) > 0x2500) >= 4:
            nonlatin.append(t)
        _add_q(norm_mod.transliterate(t), "lat",
               "segment (script-bridged)", src=idx)
        _add_q(t, "raw", "segment", src=idx)
    if nonlatin:
        _add_q(_collapse_loops(norm_mod.transliterate(
            " ".join(nonlatin))), "lat", "non-Latin chunk "
            "(script-bridged)")
    groups = {}
    for text, space, label, src in queries:
        space_pairs = lat_pairs if space == "lat" else pairs
        for s, v, a in align_against_playlist(text, space_pairs):
            groups.setdefault(v["id"], []).append(
                {"score": s, "video": v, "a": a, "method": label,
                 "src": src, "qtext": text, "space": space})
    # Corpus frequency per space, for the aggregation gate below:
    # a matched word that occurs in most episodes' dialogue is
    # generic dramatic vocabulary (aur / kya / kaha / nahi ...),
    # not identifying evidence - however many chunks repeat it.
    # The bar is >= 2 playlist-rare matched words (not more):
    # within ONE drama the genuine dialogue shares much of its
    # vocabulary with sibling episodes, and the measured true
    # Wehshi aggregation carried exactly 2 rare words against the
    # false Raja aggregation's 0 (2026-10-22).
    from .align import normalize as _tok
    _df, _n_eps = {}, max(1, len(pairs))
    for _space, _space_pairs in (("lat", lat_pairs), ("raw", pairs)):
        _counts = {}
        for _v, _segs in _space_pairs:
            _ep_words = set()
            for _s in _segs:
                _ep_words.update(_tok(_s["text"]))
            for _w in _ep_words:
                _counts[_w] = _counts.get(_w, 0) + 1
        _df[_space] = _counts

    def _distinctive_matched(distinct):
        found = set()
        cap = max(2, int(0.25 * _n_eps))
        for c in distinct:
            space_df = _df.get(c.get("space") or "raw", {})
            shared = set(_tok(c["qtext"])) & \
                set(_tok(c["a"].get("matched_text") or ""))
            for w in shared:
                if space_df.get(w, 0) <= cap:
                    found.add(w)
        return len(found)
    best = None
    for _vid, cands in groups.items():
        cands.sort(key=lambda c: c["a"]["offset"])
        for i in range(len(cands)):
            cluster = [c for c in cands
                       if 0 <= c["a"]["offset"] - cands[i]["a"]["offset"]
                       <= 120]
            by_src = {}
            for c in cluster:
                if c["src"] is not None and (
                        c["src"] not in by_src or
                        c["score"] > by_src[c["src"]]["score"]):
                    by_src[c["src"]] = c
            # Aggregation-eligible chunks only: a chunk scoring
            # below 0.30 is noise, not corroboration - it must not
            # lend its matched words to the total (measured
            # 2026-10-22: a 0.23 chunk's 6 generic words pushed a
            # wrong-episode aggregation over the line).
            distinct = [c for c in by_src.values()
                        if c["score"] >= 0.30]
            total_m = sum(c["a"]["matched_words"] for c in distinct)
            # EVERY chunk gets its own verdict check - the
            # highest-scoring chunk is not always the one that
            # meets a floor (a wide chunk can qualify on matched
            # words at a lower score).
            v_single, single = None, None
            for c in cluster:
                # A lone SHORT chunk must not pass on a saturated
                # score: the normal verdict floors assume a long
                # query, but a 5-word generic phrase ("mad what
                # are you talking about") matching 5 of its own 5
                # words scores 0.83 while identifying nothing -
                # measured live 2026-10-22, when exactly such a
                # chunk anchored the Wehshi clip ~107s early on a
                # look-alike hospital scene in the right episode.
                # A single chunk needs real identifying mass
                # (>= 8 matched words from a >= 10-word query);
                # anything less can only be promoted by the
                # >= 2-distinct-segment aggregation below.
                if c["a"]["matched_words"] < 8 or \
                        len(c["qtext"].split()) < 10:
                    continue
                v = _alignment_verdict(c["score"],
                                        c["a"]["matched_words"])
                if v == "verified" or (v and not v_single):
                    v_single, single = v, c
                if v == "verified":
                    break
            if single is None:
                single = max(cluster, key=lambda c: c["score"])
            agg = (len(distinct) >= 2 and total_m >= 10
                   and single["score"] >= 0.35
                   and _distinctive_matched(distinct) >= 2)
            if not v_single and not agg:
                continue
            verdict = v_single or "probable"
            rank = (1 if v_single else 0, total_m, single["score"])
            if best is None or rank > best[0]:
                top = max(cluster, key=lambda c: (
                    c["a"]["matched_words"], c["score"]))
                best = (rank, top, verdict, cluster, distinct)
    if best is None:
        return None
    _rank, top, verdict, cluster, distinct = best
    cand = {"score": top["score"], "video": top["video"],
            "a": top["a"],
            "method": "dialogue alignment (rescue: chunked + "
                      "script bridge)"}
    r = _clip_result(cand, clip_dur)
    r["reasons"].append(
        "Ye match rescue tajziye se mila: clip ke dialogue ko "
        "hisson mein, aur episode captions ko clip ki script se "
        "aam Latin script mein badal kar (script bridge) milaya "
        f"gaya — {len(distinct)} aazad hisse aik hi scene ki taraf "
        "ishara karte hain.")
    for c in sorted(cluster, key=lambda c: -c["score"])[:3]:
        r["reasons"].append(
            f"Chunk saboot: score {c['score']:.2f}, "
            f"{c['a']['matched_words']} alfaaz @ "
            f"{int(c['a']['offset'])}s ({c['method']}).")
    # Timestamp fine-tune (episode choice UNCHANGED): each matched
    # clip segment is an anchor - we know where that segment sits
    # inside the clip AND where its matched words sit in the
    # episode, so each anchor implies where the clip's 0s lands in
    # the episode. The median implied start replaces the raw
    # anchor-line offset, and the transcript scene range is
    # clamped to a small margin around the clip's own window -
    # the caption-block estimate alone overshot the true scene
    # start by ~74s on the measured Wehshi case (block widening
    # runs into the previous conversation), while the anchors are
    # the clip's own evidence.
    implied = []
    for c in distinct:
        cs, ce, _t = esc_segs[c["src"]]
        n_words = max(1, len(c["qtext"].split()))
        anchor_clip_t = cs + (c["a"].get("clip_anchor", 0)
                              / n_words) * max(0.1, ce - cs)
        implied.append(c["a"]["offset"] - anchor_clip_t)
    info_extra = {}
    if implied:
        implied.sort()
        t0 = implied[len(implied) // 2]
        if 0 <= t0 and abs(t0 - top["a"]["offset"]) <= clip_dur + 120:
            r["match_start"] = round(t0, 1)
            r["match_end"] = round(t0 + clip_dur, 1)
            r["match_start_hms"] = format_hms(t0)
            r["match_end_hms"] = format_hms(t0 + clip_dur)
            r["timestamp"] = round(t0, 1)
            r["timestamp_hms"] = format_hms(t0)
            r["url_at"] = watch_url_at(r["youtube_id"], t0)
            lo, hi = t0 - 20, t0 + clip_dur + 20
            if r["scene_start"] < lo:
                r["scene_start"] = round(lo, 1)
                r["scene_start_hms"] = format_hms(lo)
            if r["scene_end"] > hi:
                r["scene_end"] = round(hi, 1)
                r["scene_end_hms"] = format_hms(hi)
            r["reasons"].append(
                "Timestamp anchor hisaab se theek kiya gaya: clip "
                f"ke {len(implied)} hisse episode ki timing se "
                f"mile — clip episode ke {format_hms(t0)} se "
                "shuru hota hai.")
            info_extra = {"timestamp_anchors": len(implied),
                          "clip_start_in_episode": round(t0, 1)}
    info = {"queries": len(queries),
            "cluster_chunks": len(cluster),
            "distinct_segments": len(distinct),
            "distinctive_matched": _distinctive_matched(distinct),
            "cluster_matched_words": sum(
                c["a"]["matched_words"] for c in distinct),
            **info_extra}
    return r, verdict, info


def _probe_duration(path):
    """Container duration in seconds via PyAV, or None. Used only
    for diagnostics - a transcript-derived duration (last segment
    end) is the fallback when the probe fails."""
    try:
        import av
        with av.open(str(path)) as c:
            if c.duration:
                return round(c.duration / 1_000_000, 1)
    except Exception:
        pass
    return None


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


def verify_boundaries(youtube_id, start, end, frames=None):
    """Frame-check a scene range's two boundaries against the
    episode's own storyboard frames.

    For each edge, the frame just OUTSIDE the range is compared
    with the frame just INSIDE it (dHash distance; >=16 reads as a
    visual change, <10 as the same shot continuing):

      start_cut / end_cut = True  - a visual cut sits near the
                                    estimated boundary: the range
                                    is consistent with the frames.
      = False                     - the same shot flows across the
                                    boundary: the transcript range
                                    is probably cutting a longer
                                    scene; the kind STAYS
                                    transcript_estimate and the
                                    detail says so.
      = None                      - no usable frame pair.

    kind is "frame_checked" only when BOTH edges show a cut.
    Storyboard granularity is seconds - this verifies the boundary
    region, never a frame-exact cut. Returns a dict with kind,
    status, start_cut, end_cut, detail."""
    if frames is None:
        from . import visual as visual_mod
        got = visual_mod.boundary_frames(youtube_id, start, end)
        if not got.get("available") or not got.get("frames"):
            return {"kind": "transcript_estimate",
                    "status": "unavailable",
                    "start_cut": None, "end_cut": None,
                    "detail": "Storyboard frames available nahi — "
                              "range sirf transcript par mabni "
                              "andaza hai."}
        frames = got["frames"]

    def pick(target, before):
        cands = [f for f in frames
                 if (f["timestamp"] < target if before
                     else f["timestamp"] >= target)]
        if not cands:
            return None
        return min(cands, key=lambda f: abs(f["timestamp"] - target))

    def cut(before_f, inside_f):
        if not before_f or not inside_f:
            return None
        try:
            from PIL import Image
            d = _hamming(_dhash(Image.open(before_f["path"])),
                         _dhash(Image.open(inside_f["path"])))
        except Exception:
            return None
        if d >= 16:
            return True
        if d < 10:
            return False
        return None  # ambiguous middle: no claim either way

    start_cut = cut(pick(start, True), pick(start, False))
    end_cut = cut(pick(end, False), pick(end, True))
    if start_cut and end_cut:
        return {"kind": "frame_checked", "status": "checked",
                "start_cut": True, "end_cut": True,
                "detail": "Frames mein scene ki dono sarhadon ke "
                          "qareeb visual cut nazar aaya — range "
                          "frames se consistent hai (storyboard "
                          "ki granularity tak)."}
    notes = []
    if start_cut is False:
        notes.append("shuruaat par frames aik hi shot ko jari "
                     "dikhatay hain — scene shayad is se pehle "
                     "shuru hoti hai")
    if end_cut is False:
        notes.append("aakhir par frames aik hi shot ko jari "
                     "dikhatay hain — scene shayad is ke baad bhi "
                     "jari hai")
    if start_cut and not notes:
        notes.append("shuruaat frames se consistent hai")
    if end_cut and not notes:
        notes.append("aakhir frames se consistent hai")
    return {"kind": "transcript_estimate", "status": "checked",
            "start_cut": start_cut, "end_cut": end_cut,
            "detail": "Frames check hua lekin sarhad(en) confirm "
                      "nahi hueen: " + "; ".join(notes) + "."}


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
    # The COMPLETE scene around the match, estimated from the
    # episode transcript itself - reported separately from the
    # clip's matched window, which stays the precise claim. The
    # watch link still opens at the matched moment; the scene
    # range is the surrounding context, labelled an estimate.
    s_start, s_end = search_mod.scene_range(v["id"], start)
    b_start, b_end = search_mod.scene_block(v["id"], start)
    if s_start < start - 3:
        reasons.append(
            f"Clip scene ke darmiyan se shuru hota hai — poora "
            f"scene taqreeban {format_hms(s_start)} se shuru hota "
            f"hai (transcript estimate); clip ka matched hissa "
            f"{format_hms(start)} – {format_hms(end)} hai.")
    if s_end > end + 3:
        reasons.append(
            f"Scene clip ke khatam honay ke baad bhi taqreeban "
            f"{format_hms(s_end)} tak jari rehta hai "
            f"(transcript estimate).")
    return {
        "video_id": v["id"], "youtube_id": v["youtube_id"],
        "title": v["title"], "episode": v["position"],
        "timestamp": round(start, 1), "timestamp_hms": format_hms(start),
        "match_start": round(start, 1), "match_end": round(end, 1),
        "match_start_hms": format_hms(start),
        "match_end_hms": format_hms(end),
        "scene_start": round(s_start, 1), "scene_end": round(s_end, 1),
        "scene_start_hms": format_hms(s_start),
        "scene_end_hms": format_hms(s_end),
        "block_start": round(b_start, 1), "block_end": round(b_end, 1),
        "block_start_hms": format_hms(b_start),
        "block_end_hms": format_hms(b_end),
        "boundary": {"kind": "transcript_estimate",
                     "status": "not_checked",
                     "start_cut": None, "end_cut": None,
                     "detail": "Scene range transcript par mabni "
                               "andaza hai."},
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
                      visual_check=True, escalate_fn=None,
                      chunked_fn=None):
    """Universal clip pipeline (any drama, any language):

    1. Transcribe the clip in its ORIGINAL language (auto-detected)
       and, when it isn't English, also translate it to English.
    2. Align BOTH texts against every indexed episode of THIS
       playlist only (Unicode-aware word alignment), then fall back
       to semantic search with the clip text.
    3. Visually verify the top dialogue candidate against real
       storyboard frames of its source video (separate status).
    4. When whole-clip decodes fail, a chunked transcription
       rescue (short chunks, per-chunk language detection) gets
       one attempt through the same alignment gates. Storyboard
       frame matching alone never names an episode: measured on
       real clips (2026-10-22) it confidently picked wrong
       episodes/scenes, so frames verify - they do not guess.
    Unindexed episodes are never pretended to be searched - the
    returned coverage says exactly how many were."""
    tf = transcribe_fn or _whisper_transcribe
    try:
        segs_o, lang, prob = tf(upload_path, task="transcribe")
    except Exception as e:
        return {"ok": False,
                "error": f"Could not transcribe the clip: {e}",
                "failure_kind": "clip_unreadable",
                "diagnostics": {
                    "clip_read": False,
                    "audio_extraction": f"failed: {type(e).__name__}"}}
    text_o = " ".join(t for _, _, t in segs_o)
    text_en, segs_en = "", []
    if lang and lang != "en" and text_o.strip():
        try:
            segs_en, _l, _p = tf(upload_path, task="translate")
            text_en = " ".join(t for _, _, t in segs_en)
        except Exception:
            segs_en, text_en = [], ""
    coverage = search_mod.playlist_coverage(pid)
    pairs = []
    for v in db.get_videos(pid):
        segs = db.get_segments(v["id"])
        if segs:
            pairs.append((v, segs))
    diag = {
        "clip_read": True,
        "clip_duration_s": None,  # filled in below
        "audio_extraction": "ok" if (segs_o or segs_en) else
                            "no_speech_detected",
        "language": lang, "language_probability": prob,
        "transcript_words_original": len(text_o.split()),
        "transcript_words_english": len(text_en.split()),
        "playlist_videos_total": coverage.get("total"),
        "episodes_with_transcripts": coverage.get("searchable"),
        "episodes_searched": len(pairs),
        "segments_searched": sum(len(s) for _v, s in pairs),
        "best_alignment": None,
    }
    if not text_o.strip() and not text_en.strip():
        diag["clip_duration_s"] = _probe_duration(upload_path)
        out = _visual_only_scan(pid, upload_path, lang, prob, coverage)
        out["failure_kind"] = "no_speech"
        out["diagnostics"] = diag
        return out
    # Degenerate-transcript gate: a repetition-loop hallucination
    # is not evidence. Measured 2026-10-21: an unrelated drama's
    # music/wailing clip transcribed to hundreds of looped words
    # ("you are the one who is the one who..."), alignment found
    # nothing, but the semantic fallback promoted the generic loop
    # text to a PROBABLE match in the wrong drama (score 0.74).
    # When BOTH passes are loops, the clip simply was not heard -
    # report exactly that instead of matching the noise.
    div_o, div_en = _diversity(text_o), _diversity(text_en)
    diag["diversity_original"] = div_o
    diag["diversity_english"] = div_en
    if (div_o is not None and div_o < 0.15
            and (div_en is None or div_en < 0.15)):
        diag["clip_duration_s"] = _probe_duration(upload_path)
        diag["verdict"] = "not_verified"
        diag["failure_kind"] = "transcript_degenerate"
        return {"ok": True, "language": lang,
                "language_probability": prob,
                "clip_transcript": text_o,
                "clip_transcript_en": text_en or None,
                "results": [], "coverage": coverage,
                "verdict": "not_verified",
                "failure_kind": "transcript_degenerate",
                "diagnostics": diag,
                "method": "clip transcription (degenerate - "
                          "repetition loop)",
                "note": "Clip ki transcription kharab ho gayi "
                        "(Whisper ne aik hi jumla loop mein suna) — "
                        "is text se koi episode verify nahi ho "
                        "sakta."}
    cand = {}

    def _add(score, video, a, method):
        cur = cand.get(video["id"])
        if cur is None or score > cur["score"]:
            cand[video["id"]] = {"score": score, "video": video,
                                 "a": a, "method": method}
    weak_o, weak_en = _pass_is_weak(text_o), _pass_is_weak(text_en)
    if weak_o or weak_en:
        diag["weak_passes"] = (["original"] if weak_o else []) + \
                              (["english"] if weak_en else [])
    if text_o.strip() and not weak_o:
        for s, v, a in align_against_playlist(text_o, pairs):
            _add(s, v, a, "dialogue alignment (original language)")
    if text_en.strip() and not weak_en:
        for s, v, a in align_against_playlist(text_en, pairs):
            _add(s, v, a, "dialogue alignment (translated to English)")
    ranked = sorted(cand.values(), key=lambda c: c["score"],
                    reverse=True)
    clip_dur = 0.0
    for segs in (segs_o, segs_en):
        if segs:
            clip_dur = max(clip_dur, segs[-1][1])
    diag["clip_duration_s"] = _probe_duration(upload_path) or \
        round(clip_dur, 1)
    if ranked:
        _b = ranked[0]
        diag["best_alignment"] = {
            "score": round(_b["score"], 3),
            "matched_words": _b["a"]["matched_words"],
            "episode_title": _b["video"]["title"]}
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

    # Combined evidence (dialogue + visual TOGETHER): a dialogue
    # alignment that is real but BELOW the probable floor can still
    # be a genuine match when the clip's own frames independently
    # match that scene's frames. Neither signal alone is promoted -
    # dialogue must be weak-positive (>= 0.15 with >= 5 matched
    # words) AND the frame comparison must come back "similar".
    # The result is capped at probable and says exactly why.
    if not results and ranked and visual_check:
        c0 = ranked[0]
        if c0["score"] >= 0.15 and c0["a"]["matched_words"] >= 5:
            s0 = float(c0["a"]["offset"])
            vis = _visual_verify(upload_path,
                                 c0["video"]["youtube_id"],
                                 s0, s0 + max(2.0, clip_dur))
            if vis.get("status") == "similar":
                r = _clip_result(c0, clip_dur)
                r["visual"] = vis
                r["method"] = (c0["method"] +
                               " + visual frame match "
                               "(combined evidence)")
                r["reasons"].append(
                    "Dialogue saboot akela probable ki hadd se "
                    "neeche tha, lekin clip ke frames is scene ke "
                    "frames se mazbooti se milte hain (visual "
                    f"distance {vis.get('best_distance')}/64) — "
                    "dono saboot mila kar ye probable hai, "
                    "verified NAHI.")
                results = [r]
                verdict = "probable"

    # Rescue path: when the primary transcription produced NO
    # result at all, re-transcribe once with a stronger model and
    # retry with chunked, script-bridged alignment (see
    # _rescue_alignment for the measured failure this answers). The
    # escalated transcript must pass the same diversity gate - a
    # loop is not evidence at any model size.
    rescue_info = None
    if not results:
        # Cheapest first: chunked, script-bridged alignment on the
        # PRIMARY transcript's own segments (no new transcription).
        # When the primary model already heard the clip well, this
        # alone can rescue it; when it heard garbage, it simply
        # finds nothing and the escalation below still runs.
        if segs_o and not _pass_is_weak(text_o):
            got = _rescue_alignment(segs_o, pairs, clip_dur)
            if got:
                _r, verdict, rescue_info = got
                results = [_r]
                diag["rescue"] = rescue_info
                diag["rescue_stage"] = "primary_segments"
    if not results:
        efn = escalate_fn
        if efn is None and transcribe_fn is None:
            def efn(p, task="transcribe"):
                return _whisper_transcribe(p, task=task,
                                           model_name="small")
        if efn is not None:
            diag["rescue_attempted"] = True
            try:
                esc_segs, _el, _ep = efn(upload_path,
                                         task="transcribe")
            except Exception as _e:
                # NEVER swallow this silently again: on the live
                # app the escalation failed invisibly and the user
                # saw a bare "no match" with no trace of the rescue
                # (2026-10-21). The diagnostics must say why.
                esc_segs = []
                diag["rescue_error"] = \
                    f"{type(_e).__name__}: {_e}"[:300]
            if not esc_segs and "rescue_error" not in diag:
                # The escalation returned normally but produced
                # no segments at all - say so explicitly; a bare
                # "attempted" line hid exactly this on live.
                diag["rescue_error"] = \
                    "escalated transcription returned no segments"
            if esc_segs:
                esc_text = " ".join(t for _, _, t in esc_segs)
                diag["rescue_escalated_words"] = len(esc_text.split())
                diag["rescue_escalated_lang"] = _el
                diag["rescue_escalated_sample"] = esc_text[:160]
                _div = _diversity(esc_text)
                if (_div is None or _div >= 0.15) and \
                        not _pass_is_weak(esc_text):
                    got = _rescue_alignment(esc_segs, pairs, clip_dur)
                    if got:
                        _r, verdict, rescue_info = got
                        results = [_r]
                        diag["rescue"] = rescue_info
                        diag["rescue_stage"] = "escalated"

    if not results:
        # CHUNKED TRANSCRIPTION RESCUE - one more transcription
        # approach before giving up on speech: short chunks with
        # per-chunk language detection (see _chunked_transcribe
        # for the measured confabulation this answers). The
        # combined transcript must pass the SAME gates as every
        # other rescue (diversity, weak-pass, alignment floors) -
        # a different decode, never a lower bar.
        diag["chunked_attempted"] = True
        try:
            cfn = chunked_fn or _chunked_transcribe
            ch_segs, ch_langs = cfn(upload_path)
        except Exception as _ce:
            ch_segs, ch_langs = [], []
            diag["chunked_error"] = \
                f"{type(_ce).__name__}: {_ce}"[:300]
        if ch_segs:
            ch_text = " ".join(t for _, _, t in ch_segs)
            diag["chunked_words"] = len(ch_text.split())
            diag["chunked_langs"] = ch_langs
            diag["chunked_sample"] = ch_text[:160]
            _div = _diversity(ch_text)
            if (_div is None or _div >= 0.15) and \
                    not _pass_is_weak(ch_text):
                got = _rescue_alignment(ch_segs, pairs, clip_dur)
                if got:
                    _r, verdict, rescue_info = got
                    results = [_r]
                    diag["rescue"] = rescue_info
                    diag["rescue_stage"] = "chunked"
        elif "chunked_error" not in diag:
            diag["chunked_error"] = \
                "chunked transcription returned no segments"

    if not results:  # semantic fallback with the clip's own words
        for q in (text_en, text_o):
            if len(q.split()) < 12:
                # Too little speech to be evidence: a handful of
                # generic words can score deceptively high in
                # keyword search (0.87 for "please come here right
                # now okay") without identifying any scene.
                continue
            if _pass_is_weak(q):
                # A formulaic / loop query ("I swear to God..."
                # repeated) embeds close to countless scenes -
                # it is not scene evidence.
                continue
            sr = search_mod.search(pid, q, top_k=3)
            for r in sr.get("results", []):
                # A clip transcript is a long, specific query: a weak
                # keyword/semantic hit on it is noise, not a
                # candidate. Only medium-or-better evidence is
                # promoted; anything less stays "no verified match".
                if (r.get("score") or 0) < 0.5:
                    continue
                # Distinctive-evidence requirement (measured false
                # positives, 2026-10-21): a high score alone is NOT
                # enough for a clip. Two unrelated-drama clips were
                # promoted to probable purely on generic dramatic
                # vocabulary - the search layer's own report showed
                # only 5/19 and 7/16 of the clips' distinctive
                # terms present in the "matched" scenes. A real
                # match must contain a substantial share of the
                # clip's OWN identifying terms in its dialogue.
                _dist = search_mod.distinctive_terms(q)
                if not _dist:
                    diag.setdefault("semantic_rejected", []).append(
                        {"title": r.get("title"),
                         "reason": "clip has no distinctive terms"})
                    continue
                if len(_dist) < 5:
                    # Too little identifying material of its own:
                    # with fewer than 5 distinctive terms, generic
                    # vocabulary alone decides the "coverage" -
                    # measured FP: a formulaic clip carried only 4
                    # real terms (swear/god/love/son) and passed at
                    # 5/7 before hygiene shrank the list.
                    diag.setdefault("semantic_rejected", []).append(
                        {"title": r.get("title"),
                         "reason": "too few distinctive terms",
                         "distinctive_total": len(_dist)})
                    continue
                _covf, _hits = search_mod.coverage_factor(
                    _dist, search_mod._scene_text(r["video_id"],
                                                  r["timestamp"]))
                if _hits < 4 or _hits / len(_dist) < 0.5:
                    diag.setdefault("semantic_rejected", []).append(
                        {"title": r.get("title"),
                         "distinctive_hits": _hits,
                         "distinctive_total": len(_dist)})
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

    if not results:
        # VISUAL MATCHING - measured and deliberately NOT a verdict
        # channel. Standalone storyboard matching was evaluated on
        # the real Wehshi clip + playlist (2026-10-22) with two
        # independent methods - colour/hash descriptors and CLIP
        # image embeddings. Both produced confident WRONG answers:
        # a wrong episode passed the anchor bar (hash), and on the
        # true episode a look-alike hospital scene outscored the
        # real scene while shuffled-order and foreign-video
        # controls scored as high as the real clip (CLIP).
        # Storyboards carry ~1 frame per 10s and drama scenes
        # repeat the same rooms/actors, so frames cannot honestly
        # NAME an episode. They still verify dialogue results
        # (combined-evidence check and boundary frames above, and
        # the app's per-episode frame browser) - verification,
        # never guessing.
        diag["visual_note"] = (
            "Storyboard (tasveer) matching akelay episode nahi "
            "chunta: asal clips par is ne ghalat episode/scene "
            "diye hain (2026-10-22 ki jaanch) - frames sirf "
            "dialogue result ki tasdeeq ke liye hain.")

    _top_real_visual = bool(results) and \
        results[0].get("visual", {}).get("status") in ("similar",
                                                       "different")
    if results and visual_check and (kept or rescue_info or
                                     _top_real_visual):
        top = results[0]
        if top.get("visual", {}).get("status") in (None, "not_checked"):
            top["visual"] = _visual_verify(upload_path,
                                           top["youtube_id"],
                                           top["scene_start"],
                                           top["scene_end"])
        # Frame-check the SCENE range's boundaries themselves
        # (episode-internal cuts - a different question from the
        # clip-vs-episode frame similarity above).
        if "boundary" not in top:
            top["boundary"] = verify_boundaries(top["youtube_id"],
                                                top["scene_start"],
                                                top["scene_end"])
    # WHY nothing matched matters as much as THAT nothing matched:
    # "no data existed to search" and "searched, evidence did not
    # match" are different failures with different remedies, and the
    # UI reports them differently. best_alignment in diagnostics
    # shows the strongest raw evidence even when it was suppressed.
    if results:
        failure_kind = "ok"
    elif not pairs:
        failure_kind = "no_searchable_episodes"
    elif max(diag["transcript_words_original"],
             diag["transcript_words_english"]) < 5:
        failure_kind = "transcript_too_short"
    else:
        failure_kind = "searched_no_match"
    diag["verdict"] = verdict or "not_verified"
    diag["failure_kind"] = failure_kind
    return {"ok": True, "language": lang, "language_probability": prob,
            "clip_transcript": text_o,
            "clip_transcript_en": text_en or None,
            "results": results, "coverage": coverage,
            "verdict": verdict or "not_verified",
            "failure_kind": failure_kind,
            "diagnostics": diag,
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
