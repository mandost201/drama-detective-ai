"""Rescue alignment (2026-10-21, Wehshi case): when the primary
clip transcription cannot match - episode captions in Devanagari,
clip heard in Urdu script by the base model, code-switched noisy
audio - a stronger re-transcription plus chunked, script-bridged
alignment must still find the scene, under the SAME verdict
discipline. Fixtures are modelled on the real measured transcripts
and the episode's real published captions."""
from pathlib import Path

from backend import align, analyze, db

_EP_SEGS = [
    (2000.0, 2004.0, "उसे ठुकरा दी"),
    (2026.0, 2031.0, "आज तुमने अपनी जिंदगी में उसे एक शख्स को "
                     "भी खो दिया जिसने सिर्फ तुम्हें चाहा"),
    (2031.0, 2034.0, "सिर्फ तुम्हें चाहा बचपन से"),
    (2034.0, 2037.0, "शायद यही सजा थी तुम्हारी"),
    (2100.0, 2104.0, "मुझे अब घर जाना है खाना तैयार है"),
]

_ESC_SEGS = [
    (30.0, 35.0, "अज तुमने भी जिन्टकी में वो से एक शक्स को "
                 "भी खोटिया जिसने सर्फ तुमहे चाए"),
    (44.0, 48.0, "सर्फ तुमहे चाए बच्पन से यही सजा थी"),
]

_GIBBERISH = ("zxqwv plokm ijnhb ygtrf vbnhj qwert asdfg "
              "lkjhg mnbvc xzsdf qqq www eee rrr")


def _mk_playlist(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "r.db")
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=W",
                             title="Wehshi-like")
    db.add_videos(pid, [{"youtube_id": "y0q5xYZ8jZ8",
                         "title": "Last Episode", "position": 36,
                         "duration": None, "url": "", "thumbnail": None}])
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


def test_align_fragmented_blocks_do_not_crash():
    """Regression: biggest matching block of 2 with >= 3 total
    matched words crashed fuzzy_align (min() on empty)."""
    clip = ["khaana", "paani", "ghar", "darwaaza",
            "kitaab", "qalam", "kursi", "mez"]
    video = ["khaana", "paani", "x1x", "ghar", "darwaaza", "x2x",
             "kitaab", "qalam", "x3x", "kursi", "mez"]
    times = [float(i * 2) for i in range(len(video))]
    out = align.fuzzy_align(clip, video, times)
    assert out is not None
    assert out["score"] >= 0.5


def test_collapse_loops():
    assert analyze._collapse_loops("a a a a a a a b c") == "a a b c"
    assert analyze._collapse_loops("a a b b c") == "a a b b c"


def test_rescue_finds_devanagari_scene(tmp_path, monkeypatch):
    pid = _mk_playlist(tmp_path, monkeypatch)
    _stub_visual(monkeypatch)
    res = analyze.analyze_clip_full(
        pid, Path("clip.mp4"),
        transcribe_fn=lambda p, task="transcribe": (
            [(0.0, 5.0, _GIBBERISH)], "en", 0.9),
        escalate_fn=lambda p, task="transcribe": (
            _ESC_SEGS, "hi", 0.88))
    assert res["verdict"] == "probable"
    assert res["failure_kind"] == "ok"
    top = res["results"][0]
    assert "rescue" in top["method"]
    assert 1950 <= top["scene_start"] <= 2100
    assert res["diagnostics"]["rescue"]["distinct_segments"] >= 2
    # Timestamp fine-tune: the raw anchor line sits at ~2028s, but
    # the matched clip segments start 30s/44s into the clip, so the
    # refined clip start must land well BEFORE the anchor offset.
    assert res["diagnostics"]["rescue"]["clip_start_in_episode"]
    assert 1950 <= top["match_start"] <= 2025
    assert top["match_start"] < 2028


def test_rescue_single_weak_chunk_not_promoted(tmp_path, monkeypatch):
    pid = _mk_playlist(tmp_path, monkeypatch)
    _stub_visual(monkeypatch)
    res = analyze.analyze_clip_full(
        pid, Path("clip.mp4"),
        transcribe_fn=lambda p, task="transcribe": (
            [(0.0, 5.0, _GIBBERISH)], "en", 0.9),
        escalate_fn=lambda p, task="transcribe": (
            [_ESC_SEGS[1]], "hi", 0.88))
    assert res["verdict"] == "not_verified"
    assert res["failure_kind"] == "searched_no_match"


_FORMULAIC = ("I swear to God, there is no god but God. "
              "I swear to God, there is no god but God. "
              "I love you, my son. I love you, too.")
_NATURAL = ("You used to say that he loves you a lot and my father "
            "used to say never leave your enemy alive behind you "
            "because winning is everything in this world")


def test_pass_is_weak():
    assert analyze._pass_is_weak(_FORMULAIC) is True
    assert analyze._pass_is_weak(_NATURAL) is False
    assert analyze._pass_is_weak("you are the one who is " * 12) is True
    assert analyze._pass_is_weak("short text") is False


def test_weak_formulaic_pass_cannot_align(tmp_path, monkeypatch):
    """A formulaic pass must not produce a dialogue 'match' even
    when an episode contains the exact same formula - such
    phrases occur in countless scenes and identify none."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "f.db")
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=F",
                             title="Formulaic")
    db.add_videos(pid, [{"youtube_id": "fv1", "title": "Ep 1",
                         "position": 1, "duration": None, "url": "",
                         "thumbnail": None}])
    v = db.get_videos(pid)[0]
    db.add_segments(v["id"], [
        (100.0, 106.0, "I swear to God, there is no god but God. "
                       "I love you, my son."),
        (106.0, 112.0, "I swear to God, there is no god but God. "
                       "I love you, too."),
    ])
    db.mark_transcript(v["id"], True, source="captions")
    _stub_visual(monkeypatch)

    def tf(p, task="transcribe"):
        if task == "translate":
            return [(0.0, 29.0, _FORMULAIC)], "tr", 0.9
        return [(0.0, 29.0, _GIBBERISH)], "tr", 0.9

    res = analyze.analyze_clip_full(pid, Path("clip.mp4"),
                                    transcribe_fn=tf)
    assert res["verdict"] == "not_verified"
    assert res["diagnostics"]["weak_passes"] == ["english"]
