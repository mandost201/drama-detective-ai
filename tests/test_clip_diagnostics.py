"""Clip-pipeline hardening (2026-10-21 critical bug): deterministic
decoding, diagnostics + failure kinds, combined dialogue+visual
evidence. The root cause these guard: faster-whisper's default
temperature-fallback sampling made the SAME clip transcribe
differently run to run, so same-drama clips intermittently returned
"No verified match"."""
import wave
from pathlib import Path

from backend import analyze, db


# ------------------------------------------------------------ pinning
class _FakeInfo:
    language = "ur"
    language_probability = 0.9


class _FakeSeg:
    def __init__(self, text):
        self.start, self.end, self.text = 0.0, 1.0, text
        self.no_speech_prob = 0.0


def test_backend_transcribe_calls_pin_decoding(monkeypatch):
    calls = []

    class FakeModel:
        def transcribe(self, path, **kw):
            calls.append(kw)
            return iter([_FakeSeg("kuch dialogue hai yahan")]), _FakeInfo()

    monkeypatch.setattr(analyze, "_whisper", lambda: FakeModel())
    segs, lang, prob = analyze._whisper_transcribe(
        Path("x.wav"), task="transcribe")
    assert segs and lang == "ur"
    analyze.transcribe_audio_file_meta(Path("x.wav"))
    assert len(calls) == 2
    for kw in calls:
        assert kw.get("temperature") == 0.0
        assert kw.get("condition_on_previous_text") is False


# ------------------------------------------------------- failure kinds
def _tf_text(text, lang="en"):
    def tf(path, task="transcribe"):
        return ([(0.0, 5.0, text)], lang, 0.99)
    return tf


def _one_video_playlist(tmp_path, monkeypatch, with_segments):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=X",
                             title="X")
    db.add_videos(pid, [{"youtube_id": "YDu9akegEHU", "title": "EP25",
                         "position": 25, "duration": None, "url": "",
                         "thumbnail": None}])
    v = db.get_videos(pid)[0]
    if with_segments:
        from backend.vtt import parse_vtt
        fx = Path(__file__).parent / "fixtures" / "raja_ep25.en.vtt"
        db.add_segments(v["id"], parse_vtt(fx.read_text()))
        db.mark_transcript(v["id"], True, source="captions")
    return pid


def test_failure_kind_no_searchable_episodes(tmp_path, monkeypatch):
    pid = _one_video_playlist(tmp_path, monkeypatch, with_segments=False)
    res = analyze.analyze_clip_full(
        pid, Path("clip.wav"),
        transcribe_fn=_tf_text(
            "wakil sahab ye shadi nahi ho sakti rania mirza aa gayi"),
        visual_check=False)
    assert res["verdict"] == "not_verified"
    assert res["failure_kind"] == "no_searchable_episodes"
    d = res["diagnostics"]
    assert d["episodes_searched"] == 0
    assert d["playlist_videos_total"] == 1
    assert d["transcript_words_original"] >= 5


def test_failure_kind_searched_no_match(built_db):
    gibberish = ("zxqwv plokm ijnhb ygtrf vbnhj qwert asdfg "
                 "lkjhg mnbvc xzsdf qqq www")
    res = analyze.analyze_clip_full(
        built_db["raja"], Path("clip.wav"),
        transcribe_fn=_tf_text(gibberish), visual_check=False)
    assert res["verdict"] == "not_verified"
    assert res["failure_kind"] == "searched_no_match"
    assert res["diagnostics"]["episodes_searched"] >= 1


def test_failure_kind_no_speech(tmp_path, monkeypatch, built_db):
    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    monkeypatch.setattr(analyze, "_visual_verify",
                        lambda *a, **k: {"status": "unavailable",
                                         "best_distance": None,
                                         "detail": "test"})
    res = analyze.analyze_clip_full(
        built_db["raja"], silent,
        transcribe_fn=lambda p, task="transcribe": ([], "en", 0.9),
        visual_check=False)
    assert res["failure_kind"] == "no_speech"
    assert res["diagnostics"]["audio_extraction"] == \
        "no_speech_detected"


# ------------------------------------------- combined dialogue+visual
_WEAK = ("this marriage cannot take place "
         "zxqwv plokm ijnhb ygtrf vbnhj qwert asdfg lkjhg mnbvc "
         "xzsdf qplokm ijnbh ygtrf deswq")  # 5 real + 20 filler = 0.20


def test_degenerate_loop_transcript_is_rejected(built_db, monkeypatch):
    """Measured false positive (2026-10-21): a music/wailing clip
    from an UNRELATED drama transcribed to a repetition loop; the
    semantic fallback promoted the generic loop text to probable
    in the wrong drama. A loop is not evidence - it must be
    rejected as unhearable, and semantic search must not run."""
    loop = "you are the one who is the one who is " * 12
    called = []
    monkeypatch.setattr(analyze.search_mod, "search",
                        lambda *a, **k: called.append(1) or
                        {"results": []})
    res = analyze.analyze_clip_full(
        built_db["raja"], Path("clip.wav"),
        transcribe_fn=_tf_text(loop), visual_check=False)
    assert res["verdict"] == "not_verified"
    assert res["failure_kind"] == "transcript_degenerate"
    assert res["diagnostics"]["diversity_original"] < 0.15
    assert called == []  # semantic fallback never consulted


_SEM_CLIP = ("raniya mirza refuses the marriage proposal angrily "
             "in the courtroom before everyone present today")


def _fake_sem_result(vid):
    return {"video_id": vid, "youtube_id": "YDu9akegEHU",
            "title": "Raja EP25", "episode": 25, "timestamp": 865.0,
            "timestamp_hms": "00:14:25", "scene_start": 865.0,
            "scene_end": 900.0, "scene_start_hms": "00:14:25",
            "scene_end_hms": "00:15:00", "score": 0.74,
            "confidence": "medium", "url": "", "url_at": "",
            "thumbnail": None, "matched_lines": [], "reasons": []}


def test_semantic_fallback_needs_distinctive_evidence(built_db,
                                                      monkeypatch):
    """Two measured false positives (2026-10-21): unrelated clips
    promoted to probable on semantic score alone while only 5/19
    and 7/16 of their distinctive terms existed in the scenes.
    The fallback must demand the clip's own identifying terms."""
    vid = db.get_videos(built_db["raja"])[0]["id"]
    monkeypatch.setattr(analyze, "align_against_playlist",
                        lambda *a, **k: [])
    monkeypatch.setattr(analyze.search_mod, "search",
                        lambda *a, **k: {"results": [_fake_sem_result(vid)]})
    # scene contains only 2 of the clip's distinctive terms
    monkeypatch.setattr(analyze.search_mod, "_scene_text",
                        lambda *a, **k: "the courtroom is quiet today")
    res = analyze.analyze_clip_full(
        built_db["raja"], Path("clip.wav"),
        transcribe_fn=_tf_text(_SEM_CLIP), visual_check=False)
    assert res["verdict"] == "not_verified"
    assert res["failure_kind"] == "searched_no_match"
    assert res["diagnostics"]["semantic_rejected"]


def test_semantic_fallback_promotes_with_distinctive_evidence(
        built_db, monkeypatch):
    vid = db.get_videos(built_db["raja"])[0]["id"]
    monkeypatch.setattr(analyze, "align_against_playlist",
                        lambda *a, **k: [])
    monkeypatch.setattr(analyze.search_mod, "search",
                        lambda *a, **k: {"results": [_fake_sem_result(vid)]})
    monkeypatch.setattr(
        analyze.search_mod, "_scene_text",
        lambda *a, **k: "raniya mirza refuses the marriage proposal "
                        "in the courtroom angrily today")
    res = analyze.analyze_clip_full(
        built_db["raja"], Path("clip.wav"),
        transcribe_fn=_tf_text(_SEM_CLIP), visual_check=False)
    assert res["verdict"] == "probable"
    assert res["results"][0]["method"] == \
        "semantic search (clip transcript)"


def test_combined_evidence_upgrades_weak_dialogue(built_db, monkeypatch):
    monkeypatch.setattr(analyze.search_mod, "search",
                        lambda *a, **k: {"results": []})
    monkeypatch.setattr(
        analyze, "_visual_verify",
        lambda *a, **k: {"status": "similar", "best_distance": 6,
                         "detail": "test"})
    monkeypatch.setattr(
        analyze, "verify_boundaries",
        lambda *a, **k: {"kind": "transcript_estimate",
                         "status": "unavailable", "start_cut": None,
                         "end_cut": None, "detail": "test"})
    res = analyze.analyze_clip_full(
        built_db["raja"], Path("clip.wav"), transcribe_fn=_tf_text(_WEAK))
    assert res["verdict"] == "probable"
    assert res["failure_kind"] == "ok"
    top = res["results"][0]
    assert "combined evidence" in top["method"]
    assert top["visual"]["status"] == "similar"


def test_weak_dialogue_without_visual_stays_unverified(built_db,
                                                       monkeypatch):
    monkeypatch.setattr(analyze.search_mod, "search",
                        lambda *a, **k: {"results": []})
    monkeypatch.setattr(
        analyze, "_visual_verify",
        lambda *a, **k: {"status": "different", "best_distance": 40,
                         "detail": "test"})
    res = analyze.analyze_clip_full(
        built_db["raja"], Path("clip.wav"), transcribe_fn=_tf_text(_WEAK))
    assert res["verdict"] == "not_verified"
    assert res["failure_kind"] == "searched_no_match"
    # the suppressed evidence is REPORTED, not hidden
    assert res["diagnostics"]["best_alignment"] is not None
    assert res["diagnostics"]["best_alignment"]["matched_words"] >= 5
