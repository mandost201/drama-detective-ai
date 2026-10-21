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

from backend import db, ingest
from backend import search as search_mod
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
        db.mark_transcript(v["id"], True)
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


def _transcribe_upload(vpath, task="transcribe"):
    """Transcribe an uploaded file, working around the faster-whisper/PyAV
    metadata_errors incompatibility (old faster-whisper + new PyAV).
    task='transcribe' (default) or 'translate' (translate to English)."""
    model = _whisper_model()
    try:
        segments, _info = model.transcribe(str(vpath), beam_size=5, task=task)
    except TypeError:
        import av
        _orig = av.open

        def _patched(*a, **k):
            k.pop("metadata_errors", None)
            return _orig(*a, **k)
        av.open = _patched
        try:
            segments, _info = model.transcribe(str(vpath), beam_size=5, task=task)
        finally:
            av.open = _orig
    return [(round(s.start, 1), round(s.end, 1), s.text.strip())
            for s in segments if s.text.strip()]

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
               ("fb_mode", None), ("run_search", False)]:
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
        st.rerun()
    elif job and job.get("status") == "error":
        st.error(f"Analysis failed: {job.get('error')}")
        st.session_state.ingest_jid = None

# ---------------------------------------------------------------- playlist picker
playlists = db.list_playlists()
ready = [p for p in playlists if p.get("status") == "ready"]
if not ready:
    st.info("👆 Upar drama ka naam aur playlist link dein — phir har episode "
            "searchable ho jayegi.")
    st.stop()

def _pl_label(p):
    name = p.get("drama_name") or p.get("title") or p["url"][:40]
    return f"{name} ({p.get('video_count', 0)} videos)"

options = {_pl_label(p): p["id"] for p in ready}
labels = list(options.keys())
default_idx = 0
if pid in options.values():
    default_idx = labels.index(next(k for k, v in options.items() if v == pid))
else:
    # Fresh session: default to the largest ready playlist (the user's real
    # playlist), not whatever the database returns first (e.g. a single
    # test/upload video) - so clip matching never silently targets the
    # wrong playlist.
    _biggest = max(ready, key=lambda p: p.get("video_count", 0) or 0)
    _biggest_label = next((k for k, v in options.items() if v == _biggest["id"]), None)
    if _biggest_label is not None:
        default_idx = labels.index(_biggest_label)
choice = st.selectbox("📚 Playlist", labels, index=default_idx)
pid = options[choice]
st.session_state.active_pid = pid

pl = db.get_playlist(pid)
DRAMA = (pl.get("drama_name") or pl.get("title") or "This drama").strip()
# Backfill verified transcripts for episodes missing captions (cheap no-op
# when everything already has transcripts).
_n_seeded = apply_seed_transcripts(pid)
if _n_seeded:
    st.success(f"✅ Added verified transcripts for {_n_seeded} episode(s) missing captions.")
vids = db.get_videos(pid)
with_cap = sum(1 for v in vids if v.get("has_transcript"))
st.caption(f"🎬 {DRAMA} • {len(vids)} episodes • {with_cap} with searchable "
           f"captions/transcripts • {db.segment_count(pid):,} segments indexed")

# ---- transcribe episodes that have no captions (runs on the server)
_missing = [v for v in vids if not v.get("has_transcript")]
if _missing:
    _tx = st.session_state.get("tx_thread")
    _tx_running = _tx is not None and _tx.is_alive()
    if _tx_running:
        st.info("🎙️ Episodes without captions are being transcribed in the "
                "background… you can keep searching; newly transcribed "
                "episodes become searchable automatically.")
    else:
        if _tx is not None:
            st.session_state["tx_thread"] = None
        st.warning(f"⚠️ {len(_missing)} episode(s) ke captions YouTube par "
                   "nahi hain — ye abhi search nahi ho sakte.")
        if st.button(f"🎙️ Transcribe {len(_missing)} episode(s) without captions",
                     key="tx_btn",
                     help="Uses AI transcription (Whisper) on the server so every episode becomes searchable. Takes a while; runs in the background."):
            import threading
            from backend import ingest as _ingest

            def _run():
                try:
                    n = _ingest.transcribe_missing_episodes(pid)
                    print(f"[app] background transcription done: {n} episodes")
                except Exception as e:
                    print(f"[app] background transcription failed: {e}")

            th = threading.Thread(target=_run, daemon=True)
            st.session_state["tx_thread"] = th
            th.start()
            st.success("✅ Transcription started in the background! You can keep using the site.")

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
st.header("Step 2 — Describe the scene")
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
    st.session_state.fb_mode = None
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
            st.caption(f"🎬 Drama: {DRAMA}")
            cols = st.columns([1, 1, 1])
            cols[0].metric("⏱ Timestamp", r["timestamp_hms"])
            cols[1].metric("Confidence", _CONF_EMOJI.get(r["confidence"], r["confidence"]))
            if _is_upload_video(r):
                vpath = _upload_video_path(r["youtube_id"])
                if vpath:
                    cols[2].caption("▶ Neeche player isi second se chalega")
                else:
                    cols[2].caption("Video file server par nahi mili")
            else:
                cols[2].link_button("▶ Watch scene", r["url_at"])
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
                    st.video(str(vpath), start_time=int(r["timestamp"]))

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
                        f"- **{r['title']}** @ `{r['timestamp_hms']}` "
                        f"({_CONF_EMOJI.get(r['confidence'], r['confidence'])})")
                else:
                    st.markdown(
                        f"- **Episode {r['episode']}** @ `{r['timestamp_hms']}` — "
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
    cpath = _uploads_dir() / f"clip_{up2.name}"
    with open(cpath, "wb") as f:
        f.write(up2.getbuffer())
    try:
        with st.spinner("Transcribing clip and matching against the playlist…"):
            # transcribe + translate to English (episodes have English captions)
            # without ffmpeg (not available on Streamlit Cloud)
            segs = _transcribe_upload(cpath, task="translate")
            clip_text = " ".join(t for _, _, t in segs)
            if not clip_text.strip():
                st.warning("No speech detected in the clip. Clips are matched by spoken dialogue.")
                st.stop()
            # align against the selected playlist (no ffmpeg needed)
            # Strategy: (1) precise word alignment, (2) fallback to semantic
            # search using the translated clip text as the query.
            from backend.align import align_against_playlist
            from backend import search as _search
            from backend.vtt import format_hms
            from backend.youtube import watch_url_at
            videos = db.get_videos(pid)
            pairs = []
            for v in videos:
                s = db.get_segments(v["id"])
                if s:
                    pairs.append((v, s))
            ranked = align_against_playlist(clip_text, pairs)
            # Fallback: if word alignment finds nothing, use semantic search
            # with the clip transcript as the query.
            _fallback_results = []
            if not ranked:
                _sr = _search.search(pid, clip_text, top_k=3)
                for r in _sr.get("results", []):
                    _fallback_results.append({
                        "episode": r.get("episode", "?"),
                        "title": r.get("title", ""),
                        "timestamp_hms": r.get("timestamp_hms", ""),
                        "url_at": r.get("url_at", ""),
                        "confidence": r.get("confidence", ""),
                        "score": r.get("score", 0),
                    })
    except Exception as e:
        st.error(f"Could not analyze the clip: {e}")
        st.stop()
    finally:
        try:
            cpath.unlink()
        except OSError:
            pass
    if not ranked and not _fallback_results:
        st.warning("No matching episode found. The clip's dialogue doesn't match any episode's transcript closely. "
                   "Try describing the scene in words instead.")
        with st.expander("Clip transcript (translated to English)"):
            st.write(clip_text)
        st.stop()

    def _to_result(score, video, a):
        conf = _search._confidence(score)
        return {
            "episode": video["position"], "title": video["title"],
            "timestamp_hms": format_hms(a["offset"]),
            "url_at": watch_url_at(video["youtube_id"], a["offset"]),
            "confidence": conf, "score": round(score, 3),
        }

    if ranked:
        results_clip = [_to_result(s, v, a) for s, v, a in ranked[:3]]
        st.success("✅ Found it! (precise dialogue match)")
    else:
        results_clip = _fallback_results
        st.success("✅ Found it! (semantic match)")
    with st.expander("Clip transcript (translated to English)"):
        st.write(clip_text)
    for r in results_clip:
        ep = r.get("episode", "?")
        title = r.get("title", "")
        ts = r.get("timestamp_hms", "")
        url_at = r.get("url_at", "")
        conf = r.get("confidence", "")
        score = r.get("score", 0)
        if url_at:
            st.markdown(f"**Episode {ep}** @ `{ts}` — [{title}]({url_at}) — {conf} (match {score:.0%})")
        else:
            st.markdown(f"**Episode {ep}** @ `{ts}` — {title} — {conf} (match {score:.0%})")

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
    db.mark_transcript(v["id"], True)
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
