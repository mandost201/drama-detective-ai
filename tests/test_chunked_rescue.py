"""Chunked transcription rescue (2026-10-22, Problem 2): when
whole-clip decodes fail - the live app heard an Urdu clip as
fluent invented English - the clip is re-transcribed in short
chunks with per-chunk language detection, and the combined
transcript goes through the SAME rescue alignment gates. A
chunked pass that hears nothing must change nothing."""
from pathlib import Path

from backend import analyze, db

_EP_SEGS = [
    (2000.0, 2004.0, "उसे ठुकरा दी"),
    (2026.0, 2031.0, "आज तुमने अपनी जिंदगी में उसे एक शख्स को "
                     "भी खो दिया जिसने सिर्फ तुम्हें चाहा"),
    (2031.0, 2034.0, "सिर्फ तुम्हें चाहा बचपन से"),
    (2034.0, 2037.0, "शायद यही सजा थी तुम्हारी"),
    (2100.0, 2104.0, "मुझे अब घर जाना है खाना तैयार है"),
]

_CHUNK_SEGS = [
    (30.0, 35.0, "अज तुमने भी जिन्टकी में वो से एक शक्स को "
                 "भी खोटिया जिसने सर्फ तुमहे चाए"),
    (44.0, 48.0, "सर्फ तुमहे चाए बच्पन से यही सजा थी"),
]

_GIBBERISH = ("zxqwv plokm ijnhb ygtrf vbnhj qwert asdfg "
              "lkjhg mnbvc xzsdf qqq www eee rrr")


def _mk_playlist(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "c.db")
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=W",
                             title="Wehshi-like")
    db.add_videos(pid, [{"youtube_id": "y0q5xYZ8jZ8",
                         "title": "Last Episode", "position": 36,
                         "duration": None, "url": "",
                         "thumbnail": None}])
    v = db.get_videos(pid)[0]
    db.add_segments(v["id"], _EP_SEGS)
    db.mark_transcript(v["id"], True, source="captions")
    return pid


def _stub_visual(monkeypatch):
    monkeypatch.setattr(analyze, "_visual_verify",
                        lambda *a, **k: {"status": "unavailable",
                                         "best_distance": None,
                                         "detail": "test"})
    monkeypatch.setattr(analyze, "verify_boundaries",
                        lambda *a, **k: {"kind": "transcript_estimate",
                                         "status": "unavailable",
                                         "start_cut": None,
                                         "end_cut": None,
                                         "detail": "test"})


def _bad_whole_clip(p, task="transcribe"):
    return ([(0.0, 5.0, _GIBBERISH)], "en", 0.9)


def test_chunked_rescue_finds_scene(tmp_path, monkeypatch):
    pid = _mk_playlist(tmp_path, monkeypatch)
    _stub_visual(monkeypatch)
    res = analyze.analyze_clip_full(
        pid, Path("clip.mp4"),
        transcribe_fn=_bad_whole_clip,
        escalate_fn=_bad_whole_clip,
        chunked_fn=lambda p: (list(_CHUNK_SEGS), ["hi"]))
    assert res["verdict"] == "probable"
    assert res["failure_kind"] == "ok"
    diag = res["diagnostics"]
    assert diag["rescue_stage"] == "chunked"
    assert diag["chunked_words"] > 10
    assert diag["chunked_langs"] == ["hi"]
    top = res["results"][0]
    assert 1950 <= top["scene_start"] <= 2100


def test_chunked_empty_changes_nothing(tmp_path, monkeypatch):
    pid = _mk_playlist(tmp_path, monkeypatch)
    _stub_visual(monkeypatch)
    res = analyze.analyze_clip_full(
        pid, Path("clip.mp4"),
        transcribe_fn=_bad_whole_clip,
        escalate_fn=_bad_whole_clip,
        chunked_fn=lambda p: ([], []))
    assert res["verdict"] == "not_verified"
    assert res["failure_kind"] == "searched_no_match"
    assert "chunked_error" in res["diagnostics"]


def test_chunked_error_is_reported(tmp_path, monkeypatch):
    pid = _mk_playlist(tmp_path, monkeypatch)
    _stub_visual(monkeypatch)

    def _boom(p):
        raise RuntimeError("decoder exploded")

    res = analyze.analyze_clip_full(
        pid, Path("clip.mp4"),
        transcribe_fn=_bad_whole_clip,
        escalate_fn=_bad_whole_clip,
        chunked_fn=_boom)
    assert res["verdict"] == "not_verified"
    assert "decoder exploded" in res["diagnostics"]["chunked_error"]
