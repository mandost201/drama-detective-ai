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
from .config import WHISPER_MODEL
from .youtube import fetch_playlist, fetch_transcript, fetch_transcript_meta
from .youtube import last_failure_reason


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


_tx_reports = {}


def last_transcription_report(pid):
    """The most recent transcription report for a playlist (in this
    server process): {"requested", "done", "failed": [{title,
    youtube_id, error}], "skipped_seeded"} or None."""
    return _tx_reports.get(pid)


def transcribe_missing_episodes(pid, progress_fn=None, video_ids=None):
    """Transcribe audio for videos that have no captions/transcript.

    Only videos the app could legitimately access are processed (the
    same public audio download used everywhere else - no DRM or
    access-control bypass). A video is marked searchable ONLY after
    its transcript segments are actually indexed. Every successful
    transcription records its provenance (detected language,
    language probability, Whisper model name/version) with the
    transcript; a detection probability under 0.5 is indexed but
    flagged in the report's "low_confidence" list - a shaky
    transcript is labelled, never silently trusted. Returns a
    report dict (see last_transcription_report); per-episode
    failures are recorded in the report, never silently swallowed.

    Re-transcription: when video_ids names episodes explicitly,
    already-transcribed episodes are included (this is how a
    transcript made by an older/smaller model gets refreshed). The
    replacement is safe: the new transcript is computed FIRST and
    db.add_segments swaps old segments for new ones in one step -
    if transcription fails, the previous transcript stays indexed
    and searchable, untouched."""
    from pathlib import Path as _P
    from . import analyze as _analyze
    from .youtube import download_audio as _dl_audio

    def _prog(stage, cur, tot):
        if progress_fn:
            progress_fn(stage, cur, tot)

    if video_ids is not None:
        wanted = set(video_ids)
        missing = [v for v in db.get_videos(pid) if v["id"] in wanted]
    else:
        missing = [v for v in db.get_videos(pid)
                   if not v["has_transcript"]]
    seeded_ids = _seeded_youtube_ids()
    skipped = [v for v in missing if v["youtube_id"] in seeded_ids]
    missing = [v for v in missing if v["youtube_id"] not in seeded_ids]
    report = {"requested": len(missing) + len(skipped), "done": 0,
              "failed": [], "low_confidence": [],
              "skipped_seeded": len(skipped)}
    if not missing:
        _tx_reports[pid] = report
        return report
    tmpdir = _P(tempfile.mkdtemp(prefix="sf_whisper_"))
    try:
        for i, v in enumerate(missing):
            yid = v["youtube_id"]
            _prog(f"Transcribing: {v['title']}", i, len(missing))
            mp3 = tmpdir / f"{yid}.mp3"
            try:
                _dl_audio(yid, mp3)
                segs, det_lang, det_prob = \
                    _analyze.transcribe_audio_file_meta(mp3)
                if segs:
                    db.add_segments(v["id"], segs)
                    db.mark_transcript(v["id"], True, source="whisper",
                                       lang=det_lang,
                                       model=WHISPER_MODEL,
                                       prob=det_prob)
                    report["done"] += 1
                    if det_prob is not None and det_prob < 0.5:
                        report["low_confidence"].append({
                            "title": v["title"], "youtube_id": yid,
                            "language": det_lang,
                            "probability": round(det_prob, 2)})
                    print(f"[ingest] whisper ok: {yid} ({len(segs)} segs, "
                          f"lang={det_lang}, p={det_prob})")
                else:
                    report["failed"].append({
                        "title": v["title"], "youtube_id": yid,
                        "error": "Whisper ne koi dialogue nahi pehchana "
                                 "(khaali transcript) — audio mein "
                                 "guftagu nahi mili."})
                    print(f"[ingest] whisper empty: {yid}")
            except Exception as e:
                report["failed"].append({
                    "title": v["title"], "youtube_id": yid,
                    "error": str(e)[:300]})
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
    if report["done"]:
        # re-index so the new transcripts are semantically searchable
        _index_embeddings(pid)
        search_mod.invalidate_cache(pid)
    _tx_reports[pid] = report
    _prog("done", len(missing), len(missing))
    return report


def start_transcription_job(pid, video_ids=None):
    """Run transcribe_missing_episodes as a tracked background job
    (progress + outcome visible in the UI via the jobs table)."""
    jid = db.create_job("transcribe", pid)

    def _run():
        def progress(stage, current, total):
            db.update_job(jid, status="running", stage=stage,
                          current=current, total=total)
        try:
            rep = transcribe_missing_episodes(
                pid, progress_fn=progress, video_ids=video_ids)
            db.update_job(jid, status="done", stage="done",
                          current=rep["done"] + len(rep["failed"]),
                          total=max(1, rep["requested"]))
        except Exception as e:
            db.update_job(jid, status="error", error=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jid


def _index_embeddings_for_video(pid, video_row_id):
    """Embed just one video's segments (used after a subtitle upload,
    so the episode becomes semantically searchable immediately)."""
    if not embeddings.available():
        return
    batch, seg_ids = [], []
    for s in db.get_segments(video_row_id):
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
    search_mod.invalidate_cache(pid)


def index_subtitle_text(pid, video_row_id, text, filename="subtitles.srt"):
    """Index a user-provided subtitle file (.srt/.vtt) for ONE
    episode. The episode counts as searchable only after its cues
    are stored. Returns the number of segments indexed; raises
    ValueError when the file yields no usable cues."""
    from .vtt import parse_subtitles
    segs = parse_subtitles(text, filename)
    if not segs:
        raise ValueError("Is file mein koi subtitle lines nahi "
                         "mileen — sahi .srt ya .vtt file dein.")
    db.add_segments(video_row_id, segs)
    db.mark_transcript(video_row_id, True, source="subtitle")
    _index_embeddings_for_video(pid, video_row_id)
    return len(segs)


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
                segs, cap_lang = fetch_transcript_meta(v["youtube_id"])
                if segs:
                    db.add_segments(v["id"], segs)
                    db.mark_transcript(v["id"], True, source="captions",
                                       lang=cap_lang)
                else:
                    # Never a silent skip: the reason is stored and
                    # shown per episode in the UI.
                    db.mark_transcript(
                        v["id"], False,
                        error=last_failure_reason(v["youtube_id"]))
            except Exception as e:
                print(f"[ingest] transcript failed for {v['youtube_id']}: {e}")
                db.mark_transcript(v["id"], False,
                                   error=f"Captions fetch mein "
                                         f"error: {type(e).__name__}")
            finally:
                time.sleep(REQUEST_DELAY)
                done[0] += 1
                progress("Downloading captions", done[0], total)

        with ThreadPoolExecutor(max_workers=TRANSCRIPT_WORKERS) as ex:
            list(ex.map(one, stored))

        # ---- caption retry sweep (serial, after the burst).
        # The parallel pass fires requests in bursts, which is
        # exactly what trips YouTube's rate limiter - the unlucky
        # episodes of a burst fail even though their captions
        # exist (live analyses lost DIFFERENT episodes per run:
        # Ep16 once, Ep03+04 another time). One calm serial retry
        # per failure, after the burst has cooled, recovers most
        # of them; whatever still fails keeps its recorded reason.
        failed = [v for v in db.get_videos(pid)
                  if not v["has_transcript"]]
        if failed:
            progress("Captions retry", 0, len(failed))
            for i, v in enumerate(failed):
                try:
                    segs, cap_lang = \
                        fetch_transcript_meta(v["youtube_id"])
                    if segs:
                        db.add_segments(v["id"], segs)
                        db.mark_transcript(v["id"], True,
                                           source="captions",
                                           lang=cap_lang)
                        print(f"[ingest] caption retry rescued "
                              f"{v['youtube_id']}")
                    else:
                        db.mark_transcript(
                            v["id"], False,
                            error=last_failure_reason(
                                v["youtube_id"]))
                except Exception as e:
                    print(f"[ingest] caption retry failed for "
                          f"{v['youtube_id']}: {e}")
                finally:
                    progress("Captions retry", i + 1, len(failed))

        # ---- whisper transcription for videos with no captions at all.
        # This is what makes "100% analysis" possible for any playlist,
        # but it is slow (audio download + speech recognition per video),
        # so it only runs automatically when AUTO_TRANSCRIBE_MISSING=1.
        # Otherwise captions are indexed first (fast initial results) and
        # the user can start the same transcription from the UI, where it
        # runs in the background with progress.
        if os.environ.get("AUTO_TRANSCRIBE_MISSING") == "1":
            rep = transcribe_missing_episodes(pid, progress_fn=progress)
            if rep["done"]:
                print(f"[ingest] whisper transcribed {rep['done']} videos")
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
