"""Drama Detective AI — find the exact episode & timestamp of any scene.

Workflow:
  1. Enter the drama name and paste a public YouTube playlist URL
     (or a single video URL).
  2. The app indexes episode titles and every available caption /
     transcript, in the background, with progress.
  3. Describe a scene in English, Roman Urdu, Urdu or Hindi - dialogue,
     characters, clothing, location, events.
  4. Get the best-matching episode, the exact timestamp, a clickable
     link that opens the video at that second, the matching dialogue
     as evidence, a plain confidence level, and alternative candidates.

Evidence rules (never faked): matches are computed only from real,
accessible caption/transcript text. Episodes without captions can be
transcribed with Whisper from the UI. Confidence comes from explicit
matching signals - it is never displayed as 100%.

Runs free on Streamlit Community Cloud (no credit card needed).
"""
import sys
import time
from pathlib import Path

# repo root (parent of streamlit/) holds the reusable backend package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
# tiny Whisper fits the free 1GB container for bulk playlist backfill;
# uploads/clips use "base" for better quality (see _whisper_model below).
os.environ.setdefault("WHISPER_MODEL", "tiny")
os.environ.setdefault("SEEDS_PATH",
                      str(Path(__file__).resolve().parent / "seeds.json"))

import streamlit as st
import streamlit.components.v1 as components

from backend import db, ingest
from backend import analyze as analyze_mod
from backend import search as search_mod
from backend import selection as sel_mod
from backend import visual as visual_mod
from backend import embeddings as emb_mod
from backend.youtube import extract_playlist_id, extract_video_id

# ---------------------------------------------------------------- seed transcripts
# Episodes whose YouTube captions are missing/broken get a human-verified
# transcript bundled with the app, so they stay searchable forever.
SEEDS_PATH = Path(__file__).resolve().parent / "seeds.json"

def _load_seeds():
    try:
        import json
        if SEEDS_PATH.exists():
            return json.loads(SEEDS_PATH.read_text())
    except Exception as e:
        print(f"[seeds] load failed: {e}")
    return {}

def _seeds_hash():
    import hashlib
    try:
        return hashlib.md5(SEEDS_PATH.read_bytes()).hexdigest()[:12]
    except Exception:
        return "none"

def _marker_path(pid):
    from backend.config import DATA_DIR
    return DATA_DIR / f".seeds_applied_{pid}.json"

def _read_marker(pid):
    import json
    try:
        p = _marker_path(pid)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {}

def _write_marker(pid, h, seeded_yids):
    import json
    try:
        _marker_path(pid).write_text(json.dumps({"hash": h, "videos": sorted(seeded_yids)}))
    except Exception as e:
        print(f"[seeds] marker write failed: {e}")

def _embed_segments_for_videos(pid, video_ids):
    """Embed only the segments of the given videos (used after seeding)."""
    if not emb_mod.available():
        return
    batch, seg_ids = [], []
    for v in db.get_videos(pid):
        if v["id"] not in video_ids:
            continue
        for s in db.get_segments(v["id"]):
            if len(s["text"]) < 3:
                continue
            seg_ids.append(s["id"])
            batch.append(s["text"][:500])
    import numpy as np
    B = 64
    for i in range(0, len(batch), B):
        try:
            vecs = emb_mod.embed(batch[i:i + B])
        except Exception as e:
            print(f"[seeds] embedding batch failed: {e}")
            break
        pairs = [(seg_ids[i + j], int(vecs.shape[1]),
                  vecs[j].astype(np.float32).tobytes())
                 for j in range(vecs.shape[0])]
        db.save_embeddings(pairs)

def apply_seed_transcripts(pid):
    """Fill in transcripts for videos that have none, from bundled seeds.
    Version-aware: when seeds.json changes, previously-seeded videos get
    re-seeded (never clobbers real YouTube captions). Returns count applied."""
    seeds = _load_seeds()
    if not seeds:
        return 0
    h = _seeds_hash()
    marker = _read_marker(pid)
    if marker.get("hash") == h:
        return 0  # already up to date
    prev_seeded = set(marker.get("videos") or [])
    # Migration: the first seed rollout wrote no marker. Detect videos that
    # were seeded by it (their segments contain seed text) so updates apply.
    seed_texts = {str(s[2]) for segs in seeds.values() for s in segs}
    if not marker:
        for v in db.get_videos(pid):
            yid = v.get("youtube_id")
            if yid in seeds and v.get("has_transcript"):
                try:
                    cur = {s["text"] for s in db.get_segments(v["id"])}
                except Exception:
                    cur = set()
                if cur & seed_texts:
                    prev_seeded.add(yid)
    seeded, seeded_yids = [], set()
    for v in db.get_videos(pid):
        yid = v.get("youtube_id")
        if yid not in seeds:
            continue
        # seed only videos without real captions, or ones we seeded before
        if v.get("has_transcript") and yid not in prev_seeded:
            continue
        segs = [(float(s[0]), float(s[1]), str(s[2])) for s in seeds[yid]]
        db.add_segments(v["id"], segs)  # replaces existing segments
        db.mark_transcript(v["id"], True, source="seed")
        seeded.append(v["id"])
        seeded_yids.add(yid)
        print(f"[seeds] seeded transcript for {yid} ({len(segs)} segments)")
    if seeded:
        _embed_segments_for_videos(pid, set(seeded))
        search_mod.invalidate_cache(pid)
    _write_marker(pid, h, seeded_yids | (prev_seeded & set(seeds)))
    return len(seeded)

# ---------------------------------------------------------------- video uploads
# Users can upload any video file -> transcribed on-device -> searchable,
# jumping to the exact second with the built-in player.
def _uploads_dir():
    from backend.config import DATA_DIR
    d = DATA_DIR / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _upload_index_path():
    return _uploads_dir() / "index.json"

def _upload_video_path(youtube_id):
    """Return the local file path for an uploaded video, or None."""
    import json
    try:
        idx = json.loads(_upload_index_path().read_text())
        p = Path(idx.get(youtube_id, ""))
        if p.exists():
            return p
    except Exception:
        pass
    return None

def _register_upload(youtube_id, path):
    import json
    ip = _upload_index_path()
    try:
        idx = json.loads(ip.read_text()) if ip.exists() else {}
    except Exception:
        idx = {}
    idx[youtube_id] = str(path)
    ip.write_text(json.dumps(idx))

@st.cache_resource
def _whisper_model():
    from faster_whisper import WhisperModel
    # "base" fits the free 1GB container; "small" is better if RAM allows.
    return WhisperModel("base", device="cpu", compute_type="int8")


def _transcribe_upload_info(vpath, task="transcribe"):
    """(segments, language, language_probability) for an upload.
    Language is auto-detected by Whisper; task='translate' renders
    the same speech in English (language then reports the SOURCE
    language Whisper detected)."""
    model = _whisper_model()
    try:
        segments, info = model.transcribe(str(vpath), beam_size=5, task=task,
                                          temperature=0.0,
                                          condition_on_previous_text=False)
    except TypeError:
        import av
        _orig = av.open

        def _patched(*a, **k):
            k.pop("metadata_errors", None)
            return _orig(*a, **k)
        av.open = _patched
        try:
            segments, info = model.transcribe(str(vpath), beam_size=5,
                                              task=task,
                                              temperature=0.0,
                                              condition_on_previous_text=False)
        finally:
            av.open = _orig
    segs = [(round(s.start, 1), round(s.end, 1), s.text.strip())
            for s in segments if s.text.strip()]
    return segs, getattr(info, "language", None), \
        getattr(info, "language_probability", None)


def _transcribe_upload(vpath, task="transcribe"):
    """Transcribe an uploaded file, working around the faster-whisper/PyAV
    metadata_errors incompatibility (old faster-whisper + new PyAV).
    task='transcribe' (default) or 'translate' (translate to English)."""
    segs, _lang, _prob = _transcribe_upload_info(vpath, task=task)
    return segs


def _transcribe_upload_info_small(vpath, task="transcribe"):
    """Escalation transcription with the SMALL model - loaded only
    for this call and freed afterwards (free-tier RAM cannot hold
    two models). Used solely when the base-model clip pipeline
    found no match: a stronger hearing plus the backend's chunked,
    script-bridged rescue alignment gets one more, honest chance."""
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    try:
        kw = dict(beam_size=5, task=task, temperature=0.0,
                  condition_on_previous_text=False)
        try:
            segments, info = model.transcribe(str(vpath), **kw)
        except TypeError:
            import av
            _orig = av.open

            def _patched(*a, **k):
                k.pop("metadata_errors", None)
                return _orig(*a, **k)
            av.open = _patched
            try:
                segments, info = model.transcribe(str(vpath), **kw)
            finally:
                av.open = _orig
        segs = [(round(s.start, 1), round(s.end, 1), s.text.strip())
                for s in segments if s.text.strip()]
        return segs, getattr(info, "language", None), \
            getattr(info, "language_probability", None)
    finally:
        del model
        import gc
        gc.collect()

def _is_upload_video(r):
    return str(r.get("youtube_id") or "").startswith("upload-")

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Drama Detective AI — Episode & Scene Finder",
    page_icon="🔎",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; max-width: 860px; }
      .stButton > button { min-height: 3rem; font-size: 1.05rem; }
      .stTextInput input, .stTextArea textarea { font-size: 1.05rem; }
      h1 { font-size: 1.9rem !important; }
      h2 { font-size: 1.35rem !important; margin-top: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

db.init_db()

st.title("🔎 Drama Detective AI")
st.caption("Find the exact episode & timestamp of any drama scene — "
           "describe it in English, Roman Urdu, Urdu or Hindi.")

# ---------------------------------------------------------------- session state
for _k, _v in [("active_pid", None), ("ingest_jid", None),
               ("last_query", ""), ("last_results", None),
               ("fb_mode", None), ("run_search", False),
               ("submitted", None), ("results_pid", None),
               ("analyze_error", None), ("visual_res", None),
               ("tx_jid", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------- step 1: import
st.header("Step 1 — Add a drama playlist")
with st.expander("➕ Analyze a new playlist", expanded=st.session_state.active_pid is None):
    drama_name = st.text_input(
        "Drama name",
        placeholder="e.g. Raja London Ka",
        key="drama_name",
    )
    url = st.text_input(
        "YouTube playlist URL (or a single video URL)",
        placeholder="https://www.youtube.com/playlist?list=...",
        key="playlist_url",
    )
    if st.button("🔍 Analyze playlist", type="primary", disabled=not (url or "").strip()):
        u = url.strip()
        if not extract_playlist_id(u) and not extract_video_id(u):
            st.error("Ye link samajh nahi aaya. Aik public YouTube playlist "
                     "link paste karein (list= wala) ya kisi ek video ka link. "
                     "Private playlists nahi khulti.")
        else:
            with st.spinner("Starting analysis…"):
                pid = db.create_playlist(u)
                if (drama_name or "").strip():
                    db.update_playlist(pid, drama_name=drama_name.strip())
                jid = ingest.start_playlist_job(pid, u)
                st.session_state.active_pid = pid
                st.session_state.ingest_jid = jid
                st.session_state.last_results = None
                st.session_state.results_pid = None
                st.session_state.visual_res = None
                st.session_state.analyze_error = None
                # The new playlist becomes the selection NOW: setting
                # the picker's widget value before it is instantiated
                # below. (Leaving this to the selectbox `index` was
                # the wrong-playlist bug: Streamlit ignores `index`
                # once a picker value exists, so searches kept
                # running against the previously selected drama.)
                st.session_state["playlist_picker"] = pid
                st.session_state.submitted = {
                    "pid": pid, "name": (drama_name or "").strip()}
                # Identity transparency: if this exact source was
                # submitted before, say so - the new analysis is a
                # separate entry with its own data, never a silent
                # merge with (or reuse of) the earlier one.
                _sid = db.source_id_for(u)
                _others = [p for p in db.find_by_source(_sid)
                           if p["id"] != pid]
                if _others:
                    _o = _others[0]
                    st.session_state.source_notice = (
                        f"🔁 Isi source ({_sid}) ki aik entry pehle "
                        f"se hai: '{_o.get('drama_name') or _o.get('title')}' "
                        f"(status: {_o.get('status')}). Ye nayi "
                        "analysis us se ALAG entry hai — purani entry "
                        "ka data is mein istemal nahi hoga.")
            st.rerun()

# ---------------------------------------------------------------- ingest progress
def poll_job(jid):
    """Render live progress for a running ingest job. Returns the job dict."""
    bar = st.progress(0.0, text="Starting…")
    status_box = st.empty()
    while True:
        job = db.get_job(jid)
        if not job:
            break
        stage, cur, tot = job.get("stage") or "", job.get("current") or 0, job.get("total") or 1
        frac = min(1.0, max(0.0, cur / max(1, tot)))
        bar.progress(frac, text=f"{stage} — {cur}/{tot}")
        status = job.get("status")
        if status in ("done", "error"):
            bar.empty()
            status_box.empty()
            return job
        if status == "error":
            status_box.error(job.get("error") or "Analysis failed.")
        time.sleep(2)
    return job

pid = st.session_state.active_pid
jid = st.session_state.ingest_jid
if jid:
    job = db.get_job(jid)
    if job and job.get("status") not in ("done", "error"):
        st.info("Analyzing playlist — captions download ho rahi hain. Long "
                "playlists ko kuch minutes lag sakte hain; page khula rakhein "
                "ya wapas aa jayein, kaam background mein chalta rahega.")
        job = poll_job(jid)
        if job and job.get("status") == "done":
            with st.spinner("Filling in verified transcripts for episodes missing captions…"):
                n = apply_seed_transcripts(pid)
                if n:
                    st.success(f"Added verified transcripts for {n} episode(s) missing captions.")
            # Validate what the analysis ACTUALLY produced and say so:
            # how many videos were found vs how many got indexed
            # captions. Never claim a playlist is ready when it is not.
            _pl = db.get_playlist(pid) or {}
            _vs = db.get_videos(pid)
            _cap = sum(1 for v in _vs if v.get("has_transcript"))
            _sub = st.session_state.get("submitted") or {}
            _nm = _sub.get("name") or _pl.get("drama_name") or "Playlist"
            if _pl.get("status") == "ready" and _vs:
                _cov = search_mod.playlist_coverage(pid)
                st.success(
                    f"✅ '{_nm}' ready hai: {len(_vs)} videos mile, "
                    f"{_cap} ke captions/transcripts indexed "
                    f"({_cov['pct']}% coverage). Ab search aur clip "
                    "finder isi playlist par chalenge.")
                if sel_mod.confirmed_identity(_pl):
                    st.caption(sel_mod.confirmed_identity(_pl))
                st.session_state["playlist_picker"] = pid
            else:
                st.session_state.analyze_error = (
                    f"'{_nm}' ki playlist ready nahi hui: status="
                    f"{_pl.get('status')}, videos mile={len(_vs)}, "
                    f"captions indexed={_cap}. Ho sakta hai link "
                    "private ho, ghalat ho, ya YouTube ne videos na di "
                    "hon. Clip/scene search is playlist par nahi "
                    "chalega jab tak ye ready na ho.")
        st.rerun()
    elif job and job.get("status") == "error":
        st.error(f"Analysis failed: {job.get('error')}")
        st.session_state.analyze_error = (
            f"Analysis failed: {job.get('error')}")
        st.session_state.ingest_jid = None

if st.session_state.get("analyze_error"):
    st.error(st.session_state.analyze_error)

if st.session_state.get("source_notice"):
    st.info(st.session_state.pop("source_notice"))

# ---------------------------------------------------------------- playlist picker
# Selection is by stable playlist ID (widget key "playlist_picker"),
# never by display label: labels can collide, and a label-keyed dict
# silently dropped playlists. The analyze handler sets the picker's
# session value to the new playlist, so after an analysis the picker,
# the captions index, the search and the clip finder ALL point at the
# playlist the user just submitted.
playlists = db.list_playlists()
ready = [p for p in playlists if p.get("status") == "ready"]
if not ready:
    st.info("👆 Upar drama ka naam aur playlist link dein — phir har episode "
            "searchable ho jayegi.")
    st.stop()

_labels_by_id = sel_mod.unique_labels(ready)
_ready_ids = [p["id"] for p in ready]
if st.session_state.get("playlist_picker") not in _ready_ids:
    st.session_state["playlist_picker"] = sel_mod.default_selection(
        ready, st.session_state.get("active_pid"))
pid = st.selectbox("📚 Playlist", _ready_ids,
                   format_func=lambda i: _labels_by_id.get(i, str(i)),
                   key="playlist_picker")
st.session_state.active_pid = pid
choice = _labels_by_id.get(pid, str(pid))

# Results belong to the playlist that produced them. If the selection
# changed since the last search, drop the old results instead of
# showing one drama's matches under another drama's name.
if sel_mod.stale_results(st.session_state.get("results_pid"), pid):
    st.session_state.last_results = None
    st.session_state.results_pid = None
    st.session_state.visual_res = None
    st.session_state.fb_mode = None

pl = db.get_playlist(pid)
DRAMA = (pl.get("drama_name") or pl.get("title") or "This drama").strip()
# Backfill verified transcripts for episodes missing captions (cheap no-op
# when everything already has transcripts).
_n_seeded = apply_seed_transcripts(pid)
if _n_seeded:
    st.success(f"✅ Added verified transcripts for {_n_seeded} episode(s) missing captions.")
vids = db.get_videos(pid)
with_cap = sum(1 for v in vids if v.get("has_transcript"))
_pl_sel = db.get_playlist(pid) or {}
if sel_mod.confirmed_identity(_pl_sel):
    st.caption(sel_mod.confirmed_identity(_pl_sel))
_stats = db.playlist_stats(pid)
_tx_note = (f" • {_stats['transcribed']} transcribed (Whisper/subtitle/"
            "upload)" if _stats["transcribed"] else "")
_skip_note = (f" • {_stats['skipped']} skipped (private/deleted)"
              if _stats["skipped"] else "")
st.caption(f"🎬 {DRAMA} • {len(vids)} episodes • {with_cap} with searchable "
           f"captions/transcripts • {db.segment_count(pid):,} segments indexed"
           + _tx_note + _skip_note)
_cov = search_mod.playlist_coverage(pid)
if _cov["total"]:
    if _cov["missing"]:
        st.caption(f"📊 Playlist coverage: {_cov['searchable']}/{_cov['total']} "
                   f"episodes searchable ({_cov['pct']}%) — {_cov['missing']} "
                   "episode(s) ke captions nahi, wo search nahi ho sakte "
                   "(neeche transcribe button hai).")
    else:
        st.caption(f"📊 Playlist coverage: tamam {_cov['total']} episodes "
                   "searchable (100%).")

# Playlist inventory: the four numbers the clip finder depends on,
# reported separately - kitni videos mili, kitni accessible hain,
# kitni ke transcripts index hain, aur kitni fail hui. A clip can
# only match an episode from the "indexed" group.
with st.expander("📋 Playlist inventory — discovered / accessible / "
                 "indexed / failed"):
    _inv_indexed = [v for v in vids if v.get("has_transcript")]
    _inv_failed = [v for v in vids if not v.get("has_transcript")]
    st.write(f"• Discovered videos: **{len(vids)}**")
    st.write(f"• Accessible (private/deleted nahi): "
             f"**{len(vids) - _stats['skipped']}**")
    st.write(f"• Indexed transcripts (searchable): "
             f"**{len(_inv_indexed)}** — "
             f"{db.segment_count(pid):,} segments")
    st.write(f"• Failed / transcript nahi: **{len(_inv_failed)}**")
    for _fv in _inv_failed[:10]:
        st.caption(f"— {_fv.get('title', '')} (transcript nahi — "
                   "clip is episode mein ho to match nahi ho sakta "
                   "jab tak transcribe na ho)")
    if len(_inv_failed) > 10:
        st.caption(f"…aur {len(_inv_failed) - 10} aur episodes")

# ---- episodes without captions: real fallbacks, honestly reported --
# An episode counts as searchable ONLY when transcript segments are
# actually indexed for it. For the rest, this panel offers three real
# routes: (1) Whisper speech-to-text as a tracked background job with
# progress and per-episode errors, (2) the user's own subtitle file
# (.srt/.vtt) indexed for that episode, (3) a visual fallback -
# real frames from across the episode to spot a scene by eye.
_missing = [v for v in vids if not v.get("has_transcript")]
if _missing:
    st.warning(f"⚠️ {len(_missing)} episode(s) ke captions YouTube par "
               "nahi hain — ye abhi search nahi ho sakte. Neeche har "
               "episode ke liye asal tareeqe diye hain (Whisper "
               "transcription, apni subtitle file, ya frames dekh kar "
               "scene pehchanna).")
    _tx_job = db.get_job(st.session_state.tx_jid) \
        if st.session_state.get("tx_jid") else None
    _tx_running = bool(_tx_job and _tx_job.get("status") == "running")
    if _tx_running:
        _cur = _tx_job.get("current") or 0
        _tot = max(1, _tx_job.get("total") or 1)
        st.progress(min(1.0, _cur / _tot),
                    text=f"🎙️ Transcription chal rahi hai: "
                         f"{_tx_job.get('stage') or ''} ({_cur}/{_tot})")
        st.caption("Progress dekhne ke liye page refresh karein — "
                   "kaam background mein chalta rehta hai, aur har "
                   "episode usi waqt searchable banti hai jab us ka "
                   "transcript index ho jata hai.")
    elif _tx_job and _tx_job.get("status") == "done":
        _rep = ingest.last_transcription_report(pid) or {}
        st.success(f"🎙️ Transcription mukammal: "
                   f"{_rep.get('done', 0)} episode(s) ke transcripts "
                   "index ho gaye — wo ab searchable hain.")
        for _lc in _rep.get("low_confidence", []):
            st.warning(f"⚠️ {_lc.get('title')}: transcript index ho "
                       f"gaya lekin Whisper ka zuban par aitmaad kam "
                       f"tha ({_lc.get('probability')}) — dialogue "
                       "ghalat ho sakta hai; zaroorat ho to behtar "
                       "model se dobara transcribe karein.")
        for _f in _rep.get("failed", []):
            st.error(f"❌ {_f.get('title')}: {_f.get('error')}")
    elif _tx_job and _tx_job.get("status") == "error":
        st.error(f"🎙️ Transcription ruk gayi: {_tx_job.get('error')}")
    _cur_model = os.environ.get("WHISPER_MODEL", "tiny")
    _model_sizes = {"tiny": "~75MB", "base": "~140MB",
                    "small": "~460MB", "medium": "~1.5GB"}
    st.caption(
        "⏱️ Resource note: Whisper is app ke free server par chalta "
        "hai — koi API key ya paisa nahi chahiye, lekin slow hai: aik "
        "35–40 minute ki episode ko takreeban utna hi waqt lag sakta "
        "hai, aur app ka awake rehna zaroori hai. Is liye pehle wohi "
        "episodes transcribe karein jin ki sab se zyada zaroorat hai; "
        "baqi baad mein. Subtitle file wala tareeqa sab se tez hai "
        "agar aap ke paas .srt/.vtt maujood ho. "
        f"Maujooda Whisper model: **{_cur_model}** "
        f"(size {_model_sizes.get(_cur_model, '?')}, pehli dafa "
        "download hota hai). Har transcript ke saath us ki zuban, "
        "model aur aitmaad mehfooz hota hai — tafseel neeche "
        "'Transcripts ki tafseel' mein hai.")
    if st.button(f"🎙️ Sab {len(_missing)} missing episodes transcribe "
                 "karein (background job)",
                 key="tx_all_btn", disabled=_tx_running):
        st.session_state.tx_jid = ingest.start_transcription_job(pid)
        st.rerun()
    with st.expander(f"📋 Har missing episode ke options "
                     f"({len(_missing)})"):
        for v in _missing:
            st.markdown(f"**Episode {v['position']} — {v['title']}**")
            if st.button("🎙️ Sirf ye episode transcribe karein",
                         key=f"tx_one_{v['id']}", disabled=_tx_running):
                st.session_state.tx_jid = ingest.start_transcription_job(
                    pid, video_ids={v["id"]})
                st.rerun()
            _sub = st.file_uploader(
                "📄 Apni subtitle file dein (.srt / .vtt) — ye episode "
                "foran searchable ho jayegi",
                type=["srt", "vtt"], key=f"sub_up_{v['id']}")
            if _sub is not None and st.button(
                    "📄 Ye subtitles index karein",
                    key=f"sub_btn_{v['id']}"):
                try:
                    _text = _sub.getvalue().decode("utf-8",
                                                    errors="ignore")
                    _n = ingest.index_subtitle_text(
                        pid, v["id"], _text, _sub.name)
                    st.success(f"✅ {_n} subtitle lines index ho "
                               f"gayin — Episode {v['position']} ab "
                               "searchable hai!")
                except Exception as e:
                    st.error(f"Subtitles index nahi hueen: {e}")
            with st.expander(f"🖼️ Episode {v['position']} ke frames — "
                             "visual fallback"):
                st.caption("Transcript ke baghair bhi, episode ke "
                           "mukhtalif hisson ke asal frames dekh kar "
                           "scene pehchana ja sakta hai aur us ki "
                           "taqreeban timing note ki ja sakti hai.")
                if st.button("Frames load karein",
                             key=f"frames_{v['id']}"):
                    with st.spinner("Storyboard frames la raha hoon…"):
                        st.session_state[f"ep_frames_{v['id']}"] = \
                            visual_mod.episode_frames(v["youtube_id"])
                _ef = st.session_state.get(f"ep_frames_{v['id']}")
                if _ef:
                    if _ef.get("available"):
                        _cols = st.columns(4)
                        for _i, _fr in enumerate(_ef["frames"]):
                            _cols[_i % 4].image(
                                _fr["path"], caption=f"⏱ {_fr['hms']}")
                    else:
                        st.info("Frames available nahi: "
                                + str(_ef.get("reason")))
            st.divider()

# ---- transcript provenance: where every transcript came from ----
# Nothing about a transcript is taken on faith: each searchable
# episode lists its source, its caption/detected language, the
# speech model that made it and (for Whisper) the detector's
# confidence. Transcripts made by a different Whisper model than
# the current one are flagged as stale and can be refreshed - the
# replacement only lands if the new transcription succeeds.
_searchable = [v for v in vids if v.get("has_transcript")]
if _searchable:
    _cur_model_ui = os.environ.get("WHISPER_MODEL", "tiny")
    _src_label = {"captions": "YouTube captions", "whisper": "Whisper",
                  "subtitle": "aap ki di hui subtitle file",
                  "seed": "verified seed transcript", "upload": "upload"}
    _stale_ids = [v["id"] for v in _searchable
                  if v.get("transcript_source") == "whisper"
                  and v.get("transcript_model")
                  and v["transcript_model"] != _cur_model_ui]
    with st.expander(f"📜 Transcripts ki tafseel — har episode ka "
                     f"transcript kahan se aaya ({len(_searchable)})"):
        for v in _searchable:
            _src = _src_label.get(v.get("transcript_source") or
                                  "captions",
                                  v.get("transcript_source") or "?")
            _bits = [f"**Episode {v['position']}** — {_src}"]
            if v.get("transcript_model"):
                _bits.append(f"model: {v['transcript_model']}")
            _bits.append("zuban: "
                         + (v.get("transcript_lang") or "maloom nahi"))
            _p = v.get("transcript_prob")
            if _p is not None:
                _bits.append(f"zuban ka aitmaad: {_p:.0%}"
                             + (" ⚠️ kam" if _p < 0.5 else ""))
            st.markdown(" • ".join(_bits))
        if _stale_ids:
            st.warning(f"🎙️ {len(_stale_ids)} episode(s) purane "
                       f"Whisper model se transcribed hain — "
                       f"maujooda model **{_cur_model_ui}** hai. "
                       "Dobara transcription sirf kamyabi par purani "
                       "transcript badalti hai; nakami par purani "
                       "mehfooz rehti hai.")
            _txj2 = db.get_job(st.session_state.tx_jid) \
                if st.session_state.get("tx_jid") else None
            if st.button("🎙️ Purane model walay episodes dobara "
                         "transcribe karein",
                         key="tx_stale_btn",
                         disabled=bool(_txj2 and _txj2.get("status")
                                       == "running")):
                st.session_state.tx_jid = \
                    ingest.start_transcription_job(
                        pid, video_ids=set(_stale_ids))
                st.rerun()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("🕘 Recent searches")
    try:
        _recent = db.recent_searches(pid, limit=8)
    except Exception:
        _recent = []
    if not _recent:
        st.caption("Abhi koi search nahi hui. Searches sirf is app ke apne "
                   "database mein rehti hain — kahin bheji nahi jati.")
    for _s in _recent:
        _label = (_s.get("query") or "")[:60]
        if st.button(f"🔎 {_label}", key=f"recent_{_s['id']}"):
            st.session_state["scene_query"] = _s.get("query") or ""
            st.session_state["run_search"] = True
            st.rerun()
    st.divider()
    st.subheader("ℹ️ How it works")
    st.caption(
        "Episodes ki official captions/transcripts index hoti hain. "
        "Aap ki description (dialogue match + meaning match) unhi asal "
        "lines se compare hoti hai — koi line invent nahi ki jati. "
        "Confidence hamesha asal matching signals se banta hai; "
        "kabhi 100% nahi dikhaya jata.")

# ---------------------------------------------------------------- step 2: scene search
st.header("Step 2 — Describe the scene (Exact Clip Finder)")
_id_notice = sel_mod.identity_notice(
    st.session_state.get("submitted"), pid,
    {p["id"]: p for p in playlists})
if _id_notice:
    st.warning(_id_notice)
st.caption("Har result ke saath scene ka estimated start–end range, playlist "
           "coverage aur matching dialogue saboot ke tor par diya jata hai. "
           "Range dialogue ki continuity se estimate hoti hai — ye exact "
           "cut ka daawa nahi hai.")
query = st.text_area(
    "Scene description (English / Roman Urdu / Urdu / Hindi)",
    placeholder="e.g. vakil sahab dulha dulhan aa gaye hain, vakil kehta hai "
                "Rania Mirza ye shaadi nahi ho sakti, fees aur chai ke paise "
                "diye jate hain…",
    key="scene_query",
    height=90,
)
go = st.button("🔎 Find scene", type="primary")

_do_search = (go and (query or "").strip()) or (
    st.session_state.pop("run_search", False) and (query or "").strip())
if go and not (query or "").strip():
    st.warning("Pehle scene describe karein — dialogue, characters, kapre, "
               "jagah, jo yaad hai likh dein.")

if _do_search:
    q = query.strip()
    with st.spinner("Searching episodes… (pehli search par AI model load "
                    "hota hai, ~1 minute lag sakta hai)"):
        res = search_mod.search(pid, q)
    st.session_state.last_query = q
    st.session_state.last_results = res
    st.session_state.results_pid = pid
    st.session_state.fb_mode = None
    st.session_state.visual_res = None
    try:
        _top = (res.get("results") or [None])[0]
        db.log_search(pid, q,
                      _top["video_id"] if _top else None,
                      _top["timestamp"] if _top else None,
                      _top["confidence"] if _top else None)
    except Exception as e:
        print(f"[app] log_search failed: {e}")

# ---------------------------------------------------------------- results
res = st.session_state.get("last_results")
if res is not None:
    q = st.session_state.get("last_query", "")
    st.caption(f"Search method: {res.get('method')}")
    _rcov = res.get("coverage") or {}
    if _rcov.get("total"):
        if _rcov.get("missing"):
            st.warning(f"📊 Coverage: {_rcov['searchable']}/{_rcov['total']} "
                       f"episodes searched ({_rcov['pct']}%). "
                       f"{_rcov['missing']} episode(s) ke captions maujood "
                       "nahi thay — wo is search mein shamil nahi hue, is "
                       "liye asal scene un mein bhi ho sakta hai.")
        else:
            st.caption(f"📊 Coverage: tamam {_rcov['total']} episodes "
                       "search hue (100%).")
    if res.get("note"):
        st.warning(res["note"])
    results = res.get("results") or []
    if not results:
        st.error("Is description ka koi verified match nahi mila. Alfaz badal "
                 "kar dobara koshish karein — khaas taur par kisi character ka "
                 "naam ya dialogue ka ek tukra sab se zyada madad karta hai. "
                 "Ye bhi check karein ke episode ke captions maujood hain "
                 "(upar wali caption coverage dekhein).")
    _CONF_EMOJI = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}
    for i, r in enumerate(results, 1):
        with st.container(border=True):
            st.subheader(f"#{i} — Episode {r['episode']}: {r['title']}")
            st.caption(f"🎬 Drama: {DRAMA} • Match type: 📜 dialogue "
                       "(transcript) — frames ke liye neeche visual check")
            cols = st.columns([1, 1, 1])
            cols[0].metric("⏱ Scene (est.)",
                           f"{r.get('scene_start_hms', r['timestamp_hms'])} – "
                           f"{r.get('scene_end_hms', r['timestamp_hms'])}")
            cols[1].metric("Confidence", _CONF_EMOJI.get(r["confidence"], r["confidence"]))
            if _is_upload_video(r):
                vpath = _upload_video_path(r["youtube_id"])
                if vpath:
                    cols[2].caption("▶ Neeche player scene start se chalega")
                else:
                    cols[2].caption("Video file server par nahi mili")
            else:
                cols[2].link_button("▶ Watch from scene start", r["url_at"])
            if r.get("matched_at_hms"):
                st.caption(f"🎯 Matched line @ {r['matched_at_hms']} — "
                           "watch link scene ke start se khulta hai, "
                           "ye line usi scene ke andar aati hai.")
            if (r.get("focus_start") is not None
                    and (r["focus_start"] > r.get("scene_start", 0) + 5
                         or r["focus_end"] < r.get("scene_end", 0) - 5)):
                st.caption(f"🎯 Matched hisse ke gird ka takhmina: "
                           f"{r['focus_start_hms']} – "
                           f"{r['focus_end_hms']} — scene ka wo hissa "
                           "jahan aap ke alfaz aate hain.")
            if (r.get("block_start") is not None
                    and (r["block_start"] < r.get("scene_start", 0) - 10
                         or r["block_end"] > r.get("scene_end", 0) + 10)):
                st.caption(f"📚 Poora dialogue block (transcript): "
                           f"{r['block_start_hms']} – "
                           f"{r['block_end_hms']} — transcript is se "
                           "zyada bareek scene ki sarhad nahi de "
                           "sakta; upar wala scene range usi block "
                           "ka muntakhab hissa hai.")
            _rb = r.get("boundary") or {}
            if _rb:
                st.caption("🧭 Sarhadein: "
                           + ("frames se check ho gayin — "
                              if _rb.get("kind") == "frame_checked"
                              else "transcript estimate — ")
                           + _rb.get("detail", ""))
            if r.get("reasons"):
                st.write("**Why this matches:**")
                for reason in r["reasons"]:
                    st.write(f"- {reason}")
            if r.get("matched_lines"):
                with st.expander("📜 Matching dialogue (evidence)"):
                    for line in r["matched_lines"]:
                        if isinstance(line, dict):
                            st.write(f"> `{line.get('hms', '')}` {line.get('text', '')}")
                        else:
                            st.write(f"> {line}")
            if _is_upload_video(r):
                vpath = _upload_video_path(r["youtube_id"])
                if vpath:
                    st.video(str(vpath), start_time=int(r.get("scene_start", r["timestamp"])))
            elif i == 1:
                with st.expander("▶️ Scene preview & 🖼️ visual check"):
                    st.caption("Preview YouTube ke official player ke andar "
                               "scene start se end tak chalega. Ye koi "
                               "downloadable clip NAHI hai — sirf YouTube "
                               "player hai.")
                    components.iframe(
                        f"https://www.youtube.com/embed/{r['youtube_id']}"
                        f"?start={int(r.get('scene_start', r['timestamp']))}"
                        f"&end={int(r.get('scene_end', r['timestamp']))}&rel=0",
                        height=315)
                    if st.button("🖼️ Visual check — scene ke asal frames dikhao",
                                 key="visual_btn"):
                        with st.spinner("Episode ke preview frames la raha "
                                        "hoon… (pehli dafa ~1 minute lag "
                                        "sakta hai)"):
                            st.session_state.visual_res = visual_mod.scene_frames(
                                r["youtube_id"],
                                r.get("scene_start", r["timestamp"]),
                                r.get("scene_end", r["timestamp"]))
                    if st.button("🧭 Sarhadein frames se check karein — "
                                 "scene waqai yahin shuru aur khatam "
                                 "hoti hai?", key="bnd_btn"):
                        with st.spinner("Boundary frames check ho rahe "
                                        "hain…"):
                            st.session_state.bnd_res = \
                                analyze_mod.verify_boundaries(
                                    r["youtube_id"],
                                    r.get("scene_start", r["timestamp"]),
                                    r.get("scene_end", r["timestamp"]))
                    _bres = st.session_state.get("bnd_res")
                    if _bres:
                        st.caption(
                            ("🧭 Boundary check: frames se consistent "
                             "— " if _bres.get("kind") == "frame_checked"
                             else "🧭 Boundary check: ")
                            + _bres.get("detail", ""))
                    _vres = st.session_state.get("visual_res")
                    if _vres:
                        if _vres.get("available"):
                            _frames = _vres["frames"]
                            _fcols = st.columns(len(_frames))
                            for _fc, _fr in zip(_fcols, _frames):
                                _fc.image(_fr["path"],
                                          caption=f"⏱ {_fr['hms']}")
                            st.caption(
                                "Ye frames episode ke asal storyboard "
                                "previews hain (YouTube ki public preview "
                                "images) — inhe apne scene se mila kar "
                                "dekhein. Dialogue evidence ke saath ye "
                                "visual tasdeeq hai; app khud frame ke "
                                "characters ko identify nahi karta.")
                        else:
                            st.info("Visual check is video ke liye "
                                    "available nahi: "
                                    + str(_vres.get("reason")))

    # ------------------------------------------------ feedback / correction
    if results:
        top = results[0]
        st.markdown("**Kya yehi sahi episode hai?**")
        fb_cols = st.columns(2)
        if fb_cols[0].button("✅ Haan, yehi hai", key="fb_ok"):
            try:
                db.log_feedback(pid, q, top["video_id"], "correct")
                st.session_state.fb_mode = "thanks"
            except Exception as e:
                st.error(f"Feedback save nahi hua: {e}")
        if fb_cols[1].button("❌ Nahi, ghalat hai", key="fb_wrong"):
            st.session_state.fb_mode = "correct_me"
        if st.session_state.get("fb_mode") == "thanks":
            st.success("Shukriya! Aap ka feedback save ho gaya. 🙏")
        elif st.session_state.get("fb_mode") == "correct_me":
            ep_options = {f"Episode {v['position']} — {v['title']}": v["id"]
                          for v in vids}
            pick = st.selectbox("Sahi episode konsi hai?", list(ep_options.keys()),
                                key="fb_pick")
            note = st.text_input("Note (optional)", key="fb_note",
                                 placeholder="e.g. scene is episode ke aakhir mein hai")
            if st.button("Save correction", key="fb_save"):
                try:
                    db.log_feedback(pid, q, top["video_id"], "wrong",
                                    correct_video_id=ep_options[pick],
                                    note=note or None)
                    st.session_state.fb_mode = "thanks"
                    st.rerun()
                except Exception as e:
                    st.error(f"Correction save nahi hua: {e}")

    alts = res.get("alternatives") or []
    if alts:
        with st.expander(f"Other possible matches ({len(alts)})"):
            for r in alts:
                if _is_upload_video(r):
                    st.markdown(
                        f"- **{r['title']}** @ `{r.get('scene_start_hms', r['timestamp_hms'])} – "
                        f"{r.get('scene_end_hms', r['timestamp_hms'])}` "
                        f"({_CONF_EMOJI.get(r['confidence'], r['confidence'])})")
                else:
                    st.markdown(
                        f"- **Episode {r['episode']}** @ `{r.get('scene_start_hms', r['timestamp_hms'])} – "
                        f"{r.get('scene_end_hms', r['timestamp_hms'])}` — "
                        f"[{r['title']}]({r['url_at']}) "
                        f"({_CONF_EMOJI.get(r['confidence'], r['confidence'])})")

# ------------------------------------------------- find which episode a clip is from
st.divider()
st.header("🎯 Find which episode a clip is from")
st.caption("Aik short clip upload karein — us ke dialogue transcribe karke "
           "selected playlist se match kiya jata hai, exact episode aur "
           "timestamp ke saath.")
# Show the exact playlist the clip will be matched against, so a wrong
# selection (e.g. an uploaded single-video playlist) is obvious before matching.
st.info(f"🎯 Clip will be matched against: **{choice}**  \n"
        "Wrong playlist? Change it with the 📚 Playlist picker above first.")
up2 = st.file_uploader(
    "Upload a clip",
    type=["mp4", "mkv", "webm", "mov", "mp3", "wav", "m4a"],
    key="clip_up",
)
if st.button("🔍 Find episode for this clip", type="primary", key="find_ep",
             disabled=up2 is None):
    if up2 is None:
        st.warning("Please choose a clip first.")
        st.stop()
    # Context lock: if the playlist the user last submitted never
    # became ready and the current selection is a different one, the
    # clip would be matched against the WRONG drama. Block with a
    # clear error instead of returning a wrong episode.
    _block = sel_mod.clip_block_reason(
        st.session_state.get("submitted"), pid,
        {p["id"] for p in ready})
    if _block:
        st.error(_block)
        st.stop()
    cpath = _uploads_dir() / f"clip_{up2.name}"
    with open(cpath, "wb") as f:
        f.write(up2.getbuffer())
    try:
        with st.spinner("Clip transcribe ho raha hai (asal zuban + "
                        "English) aur playlist se match ho raha hai…"):
            # Universal pipeline (backend/analyze.analyze_clip_full):
            # original-language transcription with auto language
            # detection, an English translation pass when needed,
            # Unicode-aware dialogue alignment against THIS playlist
            # only, semantic fallback, and a separate visual frame
            # verification for the top candidate. The app's own
            # Whisper model is passed in so only one model loads.
            def _tf(path, task="transcribe"):
                return _transcribe_upload_info(path, task=task)
            clip_res = analyze_mod.analyze_clip_full(
                pid, cpath, transcribe_fn=_tf,
                escalate_fn=_transcribe_upload_info_small)
    except Exception as e:
        st.error(f"Could not analyze the clip: {e}")
        st.stop()
    finally:
        try:
            cpath.unlink()
        except OSError:
            pass
    if not clip_res.get("ok"):
        st.error(clip_res.get("error") or "Clip analyze nahi ho saka.")
        st.stop()
    if clip_res.get("language"):
        _lp = clip_res.get("language_probability")
        st.caption(f"🗣️ Clip ki zuban (auto-detected): "
                   f"**{clip_res['language']}**"
                   + (f" — detector confidence {_lp:.0%}"
                      if _lp is not None else "")
                   + ". Asal zuban ka text mehfooz rakha gaya hai; "
                     "zaroorat par English translation se bhi match "
                     "kiya gaya hai.")
    _ccov = clip_res.get("coverage") or {}
    if _ccov.get("missing"):
        st.warning(f"📊 Clip sirf {_ccov['searchable']}/{_ccov['total']} "
                   f"indexed episodes mein dhoonda gaya. "
                   f"{_ccov['missing']} episode(s) index nahi thay — "
                   "unhe search kiya hi nahi gaya, is liye clip un "
                   "mein bhi ho sakta hai.")
    _diag = clip_res.get("diagnostics") or {}
    if _diag:
        with st.expander("🔍 Clip diagnostics — pipeline ki report"):
            st.write(f"• Clip file read: "
                     f"**{'OK' if _diag.get('clip_read') else 'FAIL'}**"
                     f" • Duration: **{_diag.get('clip_duration_s')}s**"
                     f" • Audio extraction: "
                     f"**{_diag.get('audio_extraction')}**")
            st.write(f"• Clip transcript: asal zuban mein "
                     f"**{_diag.get('transcript_words_original')}** "
                     f"alfaaz, English mein "
                     f"**{_diag.get('transcript_words_english')}** "
                     "alfaaz")
            st.write(f"• Search hui: "
                     f"**{_diag.get('episodes_searched')}** episodes "
                     f"(playlist ki {_diag.get('playlist_videos_total')}"
                     f" mein se {_diag.get('episodes_with_transcripts')}"
                     f" indexed thay), "
                     f"{_diag.get('segments_searched')} segments")
            _ba = _diag.get("best_alignment")
            if _ba:
                st.write(f"• Sab se mazboot dialogue saboot: score "
                         f"**{_ba['score']}**, matched alfaaz "
                         f"**{_ba['matched_words']}** "
                         f"({_ba['episode_title']}) — agar ye hadd se "
                         "neeche tha to result roka gaya, chhupaya "
                         "nahi gaya.")
            st.write(f"• Failure kind: "
                     f"**{clip_res.get('failure_kind')}**")
    _results_clip = clip_res.get("results") or []
    if clip_res.get("note") and _results_clip:
        st.warning(clip_res["note"])
    if not _results_clip:
        _fk = clip_res.get("failure_kind")
        if _fk == "no_searchable_episodes":
            st.error("**Match mumkin hi nahi tha — data missing hai.** "
                     "Selected playlist ki kisi bhi episode ka "
                     "transcript index nahi hai, is liye clip kisi "
                     "episode se match ho hi nahi sakta tha. Ye "
                     "'searched, no match' wala case NAHI hai. Step 1 "
                     "mein playlist ka analysis mukammal karein ya "
                     "missing episodes ko transcribe karein, phir "
                     "dobara try karein.")
        elif _fk == "no_speech":
            st.error("**Clip mein koi dialogue nahi mila.** Ya clip "
                     "mein sirf music hai, ya audio sunai nahi de "
                     "raha. Neeche sirf visual hints ho sakte hain — "
                     "verified match nahi.")
        elif _fk == "transcript_too_short":
            st.error("**Clip ki guftagu bohat kam thi** (5 se kam "
                     "alfaaz transcribe hue) — itni kam guftagu se "
                     "episode verify nahi ho sakta. Thora lamba ya "
                     "wazeh dialogue wala clip try karein.")
        elif _fk == "transcript_degenerate":
            st.error("**Clip saaf sunai nahi diya.** Whisper ne clip "
                     "ki awaz ko loop/gibberish ki surat mein suna "
                     "(aksar tez background music ya rone ki awaz "
                     "wale scenes mein hota hai) — is kharab text se "
                     "koi episode match karna galat hota, is liye "
                     "koi result nahi diya gaya. Wazeh dialogue wala "
                     "hissa clip karke dobara try karein.")
        else:
            st.error("**No verified match in this playlist.** "
                     f"{_diag.get('episodes_searched', 0)} indexed "
                     "episodes asal mein search hue, lekin clip ka "
                     "saboot kisi se match nahi hua (sab se mazboot "
                     "saboot diagnostics mein hai). Agar clip kisi aur "
                     "drama ka hai to us drama ki playlist Step 1 mein "
                     "add karke wahan try karein — search hamesha sirf "
                     "selected playlist tak mehdood rehta hai.")
    else:
        # Verdict banners are evidence-tiered (backend verdict):
        # "verified" needs extensive clean dialogue evidence,
        # "probable" is real but imperfect evidence the user must
        # confirm against the matched line, and anything weaker was
        # already suppressed by the pipeline as not a match at all.
        _verdict = clip_res.get("verdict") or "not_verified"
        _how = _results_clip[0].get("method", "")
        if _verdict == "verified":
            st.success(f"🟢 **EXACT MATCH VERIFIED** ({_how}). "
                       "Dialogue saboot mazboot hai — episode, time "
                       "aur matching lines neeche hain.")
        elif _verdict == "probable":
            st.warning(f"🟡 **PROBABLE MATCH** ({_how}). Saboot "
                       "asal hai lekin kaamil nahi (masalan dheeli "
                       "speech-to-text translation) — matching "
                       "dialogue aur watch link se khud confirm kar "
                       "lein. Ye verified exact match ka daawa nahi "
                       "hai.")
        else:
            st.warning("Neeche ke candidates **verified match nahi** "
                       "hain — ye sirf hints hain (masalan bina "
                       "dialogue wale clip ke visual andaze). Inhe "
                       "saboot ke tor par na lein.")
        _VIS_ICON = {"similar": "🖼️ Visual check: frames miltay "
                                "jultay hain (supporting signal)",
                     "different": "🖼️ Visual check: frames match "
                                  "NAHI hue — dialogue match ko "
                                  "bhi shak se dekhein",
                     "unavailable": "🖼️ Visual check: available nahi",
                     "not_checked": "🖼️ Visual check: nahi hua"}
        for r in _results_clip:
            ep = r.get("episode", "?")
            title = r.get("title", "")
            url_at = r.get("url_at", "")
            conf = r.get("confidence", "")
            score = r.get("score", 0)
            s_hms = r.get("scene_start_hms", r.get("timestamp_hms", ""))
            e_hms = r.get("scene_end_hms", r.get("timestamp_hms", ""))
            head = f"**{DRAMA} — Episode {ep}** @ `{s_hms} – {e_hms}` — "
            head += (f"[{title}]({url_at})" if url_at else title)
            head += f" — {conf} (match {score:.0%})"
            st.markdown(head)
            st.caption(f"Method: {r.get('method', '')}")
            for _reason in r.get("reasons", []):
                st.caption("• " + _reason)
            _vis = r.get("visual") or {}
            _vstat = _vis.get("status", "not_checked")
            st.caption(_VIS_ICON.get(_vstat, "")
                       + (f" — {_vis.get('detail', '')}"
                          if _vis.get("detail") else ""))
            if r.get("match_start_hms"):
                st.caption(f"🎯 Clip ka matched hissa: "
                           f"{r['match_start_hms']} – "
                           f"{r['match_end_hms']} • Poora scene "
                           f"(transcript estimate): {s_hms} – {e_hms}")
            _bnd = r.get("boundary") or {}
            if _bnd:
                if _bnd.get("kind") == "frame_checked":
                    st.caption("🧭 Sarhadein: frames se check ho "
                               "gayin — " + _bnd.get("detail", ""))
                else:
                    st.caption("🧭 Sarhadein: transcript estimate — "
                               + _bnd.get("detail", ""))
            if r.get("matched_lines"):
                with st.expander("📜 Matching dialogue (saboot) — "
                                 "episode ki asal lines, timestamps "
                                 "ke saath"):
                    for _ml in r["matched_lines"]:
                        st.markdown(f"- `{_ml}`")
    if clip_res.get("clip_transcript"):
        with st.expander(f"Clip transcript (asal zuban: "
                         f"{clip_res.get('language') or '?'})"):
            st.write(clip_res["clip_transcript"])
    if clip_res.get("clip_transcript_en"):
        with st.expander("Clip transcript (English translation)"):
            st.write(clip_res["clip_transcript_en"])

# ---------------------------------------------------------------- upload & analyze your own video
st.divider()
st.header("🎞 Or analyze your own video")
st.caption("Koi bhi video upload karein — wo fully transcribe ho jayegi, "
           "searchable ban jayegi, aur scene wale second se play hogi.")
up = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mkv", "webm", "mov", "mp3", "wav", "m4a"],
    key="vid_up",
)
if st.button("🔍 Analyze uploaded video", type="primary", key="analyze_up",
             disabled=up is None):
    if up is None:
        st.warning("Please choose a video file first.")
        st.stop()
    udir = _uploads_dir()
    suffix = Path(up.name).suffix or ".mp4"
    # create the playlist first so we can name the file after it
    new_pid = db.create_playlist(f"upload:{up.name}", title=f"📁 {up.name}")
    yid = f"upload-{new_pid}"
    vpath = udir / f"{new_pid}{suffix}"
    with open(vpath, "wb") as f:
        f.write(up.getbuffer())
    _register_upload(yid, vpath)
    try:
        with st.spinner("Transcribing video… long videos can take a while, please keep this page open."):
            segs = _transcribe_upload(vpath)
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        st.stop()
    if not segs:
        st.warning("No speech detected in this video.")
        st.stop()
    db.add_videos(new_pid, [{"youtube_id": yid, "title": up.name, "position": 1,
                             "url": "", "thumbnail": None}])
    v = db.get_videos(new_pid)[0]
    db.add_segments(v["id"], segs)
    db.mark_transcript(v["id"], True, source="upload")
    db.update_playlist(new_pid, status="ready", video_count=1)
    with st.spinner("Indexing for search…"):
        _embed_segments_for_videos(new_pid, {v["id"]})
    search_mod.invalidate_cache(new_pid)
    st.session_state.active_pid = new_pid
    st.success(f"✅ Analyzed {len(segs)} segments from **{up.name}**. Describe a scene above to jump to it!")
    st.rerun()

st.divider()
st.caption("⚖️ Honest limits: matches sirf un asal captions/transcripts se "
           "bante hain jo YouTube par publicly accessible hain. Jis episode "
           "ke captions nahi, wo pehle Whisper se transcribe karna padta hai "
           "(upar button hai). Private ya deleted videos search nahi ho "
           "sakte. Analyzed data is app ke database mein rehta hai.")
