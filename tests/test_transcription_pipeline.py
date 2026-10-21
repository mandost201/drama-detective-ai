"""Missing-caption transcription pipeline (provenance edition).

The contract under test:
- captions first (with their language recorded);
- Whisper transcription records detected language, language
  probability and the model name/version WITH the transcript;
- an episode is searchable only after a real transcript lands;
- empty audio and processing failures are reported per episode
  and leave the episode unsearchable - never silently "recovered";
- a low-confidence (noisy / barely detectable language)
  transcription is indexed but flagged, never silently trusted;
- re-transcription is safe invalidation: the new transcript is
  computed first and swapped in only on success - a failed refresh
  leaves the previous transcript indexed and searchable.
"""
import tempfile
from pathlib import Path

import pytest

from backend import analyze, db, ingest, search, youtube


@pytest.fixture()
def pipe_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "pipe.db")
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=PIPE",
                             title="Pipe Drama")
    return pid


def _add(pid, pos, yid, title=None):
    db.add_videos(pid, [{
        "youtube_id": yid, "title": title or f"Pipe Episode {pos}",
        "position": pos, "duration": None,
        "url": f"https://www.youtube.com/watch?v={yid}",
        "thumbnail": None}])
    return {v["youtube_id"]: v for v in db.get_videos(pid)}[yid]


def _stub_audio(monkeypatch):
    monkeypatch.setattr(youtube, "download_audio",
                        lambda yid, out: Path(out).write_bytes(b"0" * 10)
                        or Path(out))


def _vid(pid, yid):
    return {v["youtube_id"]: v for v in db.get_videos(pid)}[yid]


def test_successful_transcription_records_provenance(pipe_db,
                                                     monkeypatch):
    pid = pipe_db
    v = _add(pid, 1, "OKVIDEO0001")
    _stub_audio(monkeypatch)
    monkeypatch.setattr(
        analyze, "transcribe_audio_file_meta",
        lambda p: ([(0.0, 2.0, "Rania Mirza aa gayi hain."),
                    (2.0, 4.0, "Wakeel sahab intezaar kar rahe hain.")],
                   "ur", 0.93))
    rep = ingest.transcribe_missing_episodes(pid)
    assert rep["done"] == 1 and rep["failed"] == []
    got = _vid(pid, "OKVIDEO0001")
    assert got["has_transcript"]
    assert got["transcript_source"] == "whisper"
    assert got["transcript_lang"] == "ur"
    assert got["transcript_model"] == ingest.WHISPER_MODEL
    assert got["transcript_prob"] == pytest.approx(0.93)
    assert got["transcript_at"]
    res = search.search(pid, "Rania Mirza")
    assert res["results"] and res["results"][0]["episode"] == 1


def test_empty_audio_is_a_failure_not_a_recovery(pipe_db, monkeypatch):
    pid = pipe_db
    _add(pid, 1, "EMPTY000001")
    _stub_audio(monkeypatch)
    monkeypatch.setattr(analyze, "transcribe_audio_file_meta",
                        lambda p: ([], "ur", 0.9))
    rep = ingest.transcribe_missing_episodes(pid)
    assert rep["done"] == 0 and len(rep["failed"]) == 1
    assert "koi dialogue nahi" in rep["failed"][0]["error"]
    got = _vid(pid, "EMPTY000001")
    assert not got["has_transcript"]
    assert db.playlist_stats(pid)["missing"] == 1


def test_processing_failure_reported_and_others_continue(pipe_db,
                                                        monkeypatch):
    pid = pipe_db
    _add(pid, 1, "BADVIDEO001")
    _add(pid, 2, "OKVIDEO0002")
    _stub_audio(monkeypatch)

    def meta(path):
        if "BADVIDEO" in str(path):
            raise RuntimeError("audio download corrupted")
        return ([(0.0, 2.0, "Sab theek hai.")], "ur", 0.9)

    monkeypatch.setattr(analyze, "transcribe_audio_file_meta", meta)
    rep = ingest.transcribe_missing_episodes(pid)
    assert rep["done"] == 1 and len(rep["failed"]) == 1
    assert "corrupted" in rep["failed"][0]["error"]
    assert not _vid(pid, "BADVIDEO001")["has_transcript"]
    assert _vid(pid, "OKVIDEO0002")["has_transcript"]


def test_noisy_low_confidence_indexed_but_flagged(pipe_db, monkeypatch):
    pid = pipe_db
    _add(pid, 1, "NOISY000001")
    _stub_audio(monkeypatch)
    monkeypatch.setattr(
        analyze, "transcribe_audio_file_meta",
        lambda p: ([(0.0, 3.0, "kuchh garbled se alfaz")], "ur", 0.31))
    rep = ingest.transcribe_missing_episodes(pid)
    assert rep["done"] == 1
    assert len(rep["low_confidence"]) == 1
    assert rep["low_confidence"][0]["probability"] == 0.31
    got = _vid(pid, "NOISY000001")
    assert got["has_transcript"]
    assert got["transcript_prob"] == pytest.approx(0.31)


def test_undetectable_language_is_honest(pipe_db, monkeypatch):
    """Speech Whisper cannot place in any language (probability 0,
    language unknown): the transcript exists, but the provenance
    says plainly that the language is unknown and confidence nil -
    it is flagged, not dressed up."""
    pid = pipe_db
    _add(pid, 1, "UNKNOWN0001")
    _stub_audio(monkeypatch)
    monkeypatch.setattr(
        analyze, "transcribe_audio_file_meta",
        lambda p: ([(0.0, 2.0, "unplaceable speech sounds")], None, 0.0))
    rep = ingest.transcribe_missing_episodes(pid)
    assert rep["done"] == 1
    assert len(rep["low_confidence"]) == 1
    got = _vid(pid, "UNKNOWN0001")
    assert got["transcript_lang"] is None
    assert got["transcript_prob"] == 0.0


def test_retranscription_replaces_only_on_success(pipe_db, monkeypatch):
    pid = pipe_db
    v = _add(pid, 1, "STALE000001")
    _stub_audio(monkeypatch)
    monkeypatch.setattr(ingest, "WHISPER_MODEL", "tiny")
    monkeypatch.setattr(
        analyze, "transcribe_audio_file_meta",
        lambda p: ([(0.0, 2.0, "PURANI LINE pehchan")], "ur", 0.9))
    assert ingest.transcribe_missing_episodes(pid)["done"] == 1
    assert _vid(pid, "STALE000001")["transcript_model"] == "tiny"

    # model upgraded; the refresh FAILS -> old transcript untouched
    monkeypatch.setattr(ingest, "WHISPER_MODEL", "base")

    def boom(path):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(analyze, "transcribe_audio_file_meta", boom)
    rep = ingest.transcribe_missing_episodes(pid, video_ids={v["id"]})
    assert rep["done"] == 0 and len(rep["failed"]) == 1
    got = _vid(pid, "STALE000001")
    assert got["has_transcript"] and got["transcript_model"] == "tiny"
    assert "PURANI LINE" in db.get_segments(v["id"])[0]["text"]
    assert search.search(pid, "PURANI LINE pehchan")["results"]

    # refresh succeeds -> segments swapped atomically, model updated
    monkeypatch.setattr(
        analyze, "transcribe_audio_file_meta",
        lambda p: ([(0.0, 2.0, "NAYI LINE pehchan")], "ur", 0.95))
    rep = ingest.transcribe_missing_episodes(pid, video_ids={v["id"]})
    assert rep["done"] == 1
    texts = [s["text"] for s in db.get_segments(v["id"])]
    assert texts == ["NAYI LINE pehchan"]
    assert _vid(pid, "STALE000001")["transcript_model"] == "base"
    # the replaced text is really gone from the index, the new text
    # is what search finds now
    assert not search.search(pid, "PURANI")["results"]
    assert search.search(pid, "NAYI")["results"]


def test_captions_provenance_records_track_language(pipe_db,
                                                    monkeypatch):
    pid = pipe_db
    monkeypatch.setattr(
        ingest, "fetch_playlist",
        lambda url: ("Pipe Drama", [{
            "youtube_id": "CAPVIDEO001", "title": "Pipe Episode 1",
            "position": 1, "duration": None,
            "url": "https://www.youtube.com/watch?v=CAPVIDEO001",
            "thumbnail": None}]))
    monkeypatch.setattr(
        ingest, "fetch_transcript_meta",
        lambda yid: ([(0.0, 2.0, "Caption wali line.")], "ur"))
    ingest.run_playlist_job("jobx", pid,
                            "https://youtube.com/playlist?list=PIPE")
    got = _vid(pid, "CAPVIDEO001")
    assert got["has_transcript"]
    assert got["transcript_source"] == "captions"
    assert got["transcript_lang"] == "ur"
    assert got["transcript_model"] is None


def test_fetch_transcript_wrapper_keeps_old_contract(monkeypatch):
    monkeypatch.setattr(youtube, "_transcript_ytdlp_meta",
                        lambda vid: ([(0.0, 1.0, "hi")], "en"))
    assert youtube.fetch_transcript("X") == [(0.0, 1.0, "hi")]
    assert youtube.fetch_transcript_meta("X") == ([(0.0, 1.0, "hi")],
                                                  "en")


def test_silence_hallucination_is_not_a_transcript(monkeypatch):
    """Measured on real Whisper (base): a pure-silence file yields
    one invented segment, "You", at no_speech_prob 0.68. When every
    segment is more likely no-speech than speech, the meta
    transcription must return EMPTY so the pipeline reports 'no
    dialogue found' instead of indexing the invention."""
    from types import SimpleNamespace

    class FakeModel:
        def __init__(self, segs):
            self._segs = segs

        def transcribe(self, path, beam_size=5, **kw):
            info = SimpleNamespace(language="en",
                                   language_probability=0.36)
            return iter(self._segs), info

    def seg(text, nsp):
        return SimpleNamespace(start=0.0, end=1.0, text=text,
                               no_speech_prob=nsp)

    monkeypatch.setattr(analyze, "_whisper",
                        lambda: FakeModel([seg("You", 0.68)]))
    segs, lang, prob = analyze.transcribe_audio_file_meta(
        Path("silence.wav"))
    assert segs == []

    monkeypatch.setattr(
        analyze, "_whisper",
        lambda: FakeModel([seg("You", 0.68),
                           seg("Asal dialogue yahan hai.", 0.05)]))
    segs, lang, prob = analyze.transcribe_audio_file_meta(
        Path("mixed.wav"))
    assert len(segs) == 2  # real speech present -> nothing dropped
