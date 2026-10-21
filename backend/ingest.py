"""Background playlist ingestion: metadata -> transcripts -> embeddings."""
import os
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import db
from . import embeddings
from . import search as search_mod
from .config import MAX_VIDEOS_PER_PLAYLIST, REQUEST_DELAY, TRANSCRIPT_WORKERS
from .youtube import fetch_playlist, fetch_transcript


def _seeded_youtube_ids():
    """YouTube IDs covered by bundled verified transcripts (SEEDS_PATH).
    The ingest skips Whisper for these — the seeds are better quality."""
    sp = os.environ.get("SEEDS_PATH")
    if not sp:
        return set()
    try:
        import json
        return set(json.loads(open(sp).read()))
    except Exception:
        return set()


def transcribe_missing_episodes(pid, progress_fn=None):
    """Transcribe audio for videos that have no captions/transcript.
    Returns the number of videos newly transcribed. progress_fn(stage,
    current, total) is called for UI updates. Safe to call anytime;
    skips videos already transcribed or covered by bundled seeds."""
    from pathlib import Path as _P
    from . import analyze as _analyze
    from .youtube import download_audio as _dl_audio

    def _prog(stage, cur, tot):
        if progress_fn:
            progress_fn(stage, cur, tot)

    missing = [v for v in db.get_videos(pid) if not v["has_transcript"]]
    seeded_ids = _seeded_youtube_ids()
    missing = [v for v in missing if v["youtube_id"] not in seeded_ids]
    if not missing:
        return 0
    tmpdir = _P(tempfile.mkdtemp(prefix="sf_whisper_"))
    done = 0
    try:
        for i, v in enumerate(missing):
            yid = v["youtube_id"]
            _prog(f"Transcribing audio ({i+1}/{len(missing)})", i, len(missing))
            mp3 = tmpdir / f"{yid}.mp3"
            try:
                _dl_audio(yid, mp3)
                segs = _analyze.transcribe_audio_file(mp3)
                if segs:
                    db.add_segments(v["id"], segs)
                    db.mark_transcript(v["id"], True)
                    done += 1
                    print(f"[ingest] whisper ok: {yid} ({len(segs)} segs)")
                else:
                    print(f"[ingest] whisper empty: {yid}")
            except Exception as e:
                print(f"[ingest] whisper failed for {yid}: {e}")
            finally:
                try:
                    mp3.unlink()
                except OSError:
                    pass
    finally:
        try:
            tmpdir.rmdir()
        except OSError:
            pass
    if done:
        # re-index so the new transcripts are semantically searchable
        _index_embeddings(pid)
        search_mod.invalidate_cache(pid)
    _prog("done", len(missing), len(missing))
    return done


def run_playlist_job(jid: str, pid: str, url: str):
    def progress(stage, current, total):
        db.update_job(jid, status="running", stage=stage,
                      current=current, total=total)

    try:
        progress("Reading playlist", 0, 1)
        title, videos = fetch_playlist(url)
        videos = videos[:MAX_VIDEOS_PER_PLAYLIST]
        db.update_playlist(pid, title=title, video_count=len(videos),
                           status="ingesting")
        db.add_videos(pid, videos)
        stored = db.get_videos(pid)
        total = len(stored)

        # ---- transcripts (parallel, polite)
        progress("Downloading captions", 0, total)
        done = [0]

        def one(v):
            try:
                segs = fetch_transcript(v["youtube_id"])
                if segs:
                    db.add_segments(v["id"], segs)
                    db.mark_transcript(v["id"], True)
                else:
                    db.mark_transcript(v["id"], False)
            except Exception as e:
                print(f"[ingest] transcript failed for {v['youtube_id']}: {e}")
                db.mark_transcript(v["id"], False)
            finally:
                time.sleep(REQUEST_DELAY)
                done[0] += 1
                progress("Downloading captions", done[0], total)

        with ThreadPoolExecutor(max_workers=TRANSCRIPT_WORKERS) as ex:
            list(ex.map(one, stored))

        # ---- whisper transcription for videos with no captions at all.
        # This is what makes "100% analysis" possible for any playlist,
        # but it is slow (audio download + speech recognition per video),
        # so it only runs automatically when AUTO_TRANSCRIBE_MISSING=1.
        # Otherwise captions are indexed first (fast initial results) and
        # the user can start the same transcription from the UI, where it
        # runs in the background with progress.
        if os.environ.get("AUTO_TRANSCRIBE_MISSING") == "1":
            n_whisper = transcribe_missing_episodes(pid, progress_fn=progress)
            if n_whisper:
                print(f"[ingest] whisper transcribed {n_whisper} videos")
        else:
            n_missing = sum(1 for v in db.get_videos(pid)
                            if not v["has_transcript"])
            if n_missing:
                print(f"[ingest] {n_missing} videos have no captions; "
                      "Whisper backfill available from the UI "
                      "(or set AUTO_TRANSCRIBE_MISSING=1)")

        # ---- embeddings
        progress("Indexing for semantic search", 0, 1)
        _index_embeddings(pid)

        with_caption = sum(1 for v in db.get_videos(pid) if v["has_transcript"])
        db.update_playlist(pid, status="ready", analyzed_at=time.time())
        search_mod.invalidate_cache(pid)
        db.update_job(jid, status="done", stage="done", current=total, total=total)
        print(f"[ingest] playlist {pid}: {total} videos, {with_caption} with captions")
    except Exception as e:
        traceback.print_exc()
        db.update_playlist(pid, status="error")
        db.update_job(jid, status="error", error=str(e))


def _index_embeddings(pid: str):
    if not embeddings.available():
        print("[ingest] embeddings unavailable - keyword search only")
        return
    dim = embeddings.dim() or 384
    batch, seg_ids = [], []
    for v in db.get_videos(pid):
        for s in db.get_segments(v["id"]):
            # embed short windows (merge tiny segments implicitly by batching text)
            if len(s["text"]) < 3:
                continue
            seg_ids.append(s["id"])
            batch.append(s["text"][:500])
    B = 64
    for i in range(0, len(batch), B):
        try:
            vecs = embeddings.embed(batch[i:i + B])
        except Exception as e:
            print(f"[ingest] embedding batch failed: {e}")
            break
        pairs = [(seg_ids[i + j], int(vecs.shape[1]),
                  vecs[j].astype(np.float32).tobytes())
                 for j in range(vecs.shape[0])]
        db.save_embeddings(pairs)


def start_playlist_job(pid: str, url: str) -> str:
    jid = db.create_job("playlist", pid)
    t = threading.Thread(target=run_playlist_job, args=(jid, pid, url),
                         daemon=True)
    t.start()
    return jid
