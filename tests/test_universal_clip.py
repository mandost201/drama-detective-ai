"""Universal multilingual clip finder tests.

Ground truth: the ~55s fan clip of Raja London Ka's lawyer scene is
from Episode 25 (video YDu9akegEHU), starting at 00:13:48 (828s) -
verified by hand against the official subtitles on 2026-10-21.
The clip's speech is Urdu; the episode's captions are English. The
old pipeline transcribed ONLY a translation and tokenized words
with an [a-z0-9] regex, so original-language matching - and any
non-Latin script at all - was impossible.
"""
from pathlib import Path

import pytest
from PIL import Image

from backend import analyze, db
from backend.align import align_against_playlist, normalize

# What Whisper's translate pass produces for the real Urdu clip
# (built from Episode 25's own subtitle lines at 828-885s).
CLIP_EN = ("Lawyer, the groom and bridegroom are here. I was saying, "
           "get them signed on the papers. Then let's go to the court "
           "to do the rest of the work. Rania Mirza. This marriage "
           "cannot take place. What are you saying?")
CLIP_UR_SEGS = [(0.0, 3.0, "وکیل صاحب دلہا دلہن آ گئے ہیں"),
                (3.0, 6.0, "میں کہہ رہا تھا کاغذات پر سائن کروا لیں")]
CLIP_EN_SEGS = [(0.0, 20.0, CLIP_EN)]


def _stub_tf(original, translated=None, lang="ur", prob=0.93):
    def tf(path, task="transcribe"):
        if task == "translate":
            return (translated or []), lang, prob
        return original, lang, prob
    return tf


# ------------------------------------------------------- unicode align

def test_normalize_keeps_non_latin_scripts():
    assert normalize("وکیل صاحب آ گئے")  # Urdu words survive
    assert normalize("مرحبا بالعالم")     # Arabic
    assert normalize("वकील साहब")         # Hindi (Devanagari)
    assert normalize("Lawyer साहब")       # mixed scripts


def test_urdu_clip_aligns_against_urdu_captions():
    segs = [{"start": 100.0, "end": 104.0,
             "text": "وکیل صاحب دلہا دلہن آ گئے ہیں"},
            {"start": 104.0, "end": 108.0,
             "text": "میں کہہ رہا تھا کاغذات پر سائن کروا لیں"},
            {"start": 108.0, "end": 112.0,
             "text": "پھر عدالت جا کر باقی کام کریں گے"}]
    video = {"id": 1, "youtube_id": "URDU0000001", "title": "Ep 1",
             "position": 1, "url": "", "thumbnail": None}
    ranked = align_against_playlist(
        "وکیل صاحب دلہا دلہن آ گئے ہیں میں کہہ رہا تھا", [(video, segs)])
    assert ranked, "Urdu clip must align against Urdu captions"
    score, _v, a = ranked[0]
    assert score > 0.5
    assert abs(a["offset"] - 100.0) < 5


# ------------------------------------------- cross-language clip match

def test_urdu_clip_matches_english_captions_via_translate(built_db,
                                                           monkeypatch):
    pid = built_db["raja"]
    # no storyboard network in tests: visual check reports honestly
    monkeypatch.setattr(analyze, "_visual_verify",
                        lambda *a, **k: {"status": "unavailable",
                                         "best_distance": None,
                                         "detail": "test stub"})
    res = analyze.analyze_clip_full(
        pid, Path("clip.wav"),
        transcribe_fn=_stub_tf(CLIP_UR_SEGS, CLIP_EN_SEGS))
    assert res["ok"] is True
    assert res["language"] == "ur"
    assert "وکیل" in res["clip_transcript"]  # original preserved
    assert res["clip_transcript_en"] == CLIP_EN
    assert res["results"], res
    top = res["results"][0]
    assert top["youtube_id"] == "YDu9akegEHU"
    assert top["episode"] == 25
    assert abs(top["timestamp"] - 828) <= 25, top["timestamp_hms"]
    assert "translated" in top["method"]
    assert top["visual"]["status"] == "unavailable"
    cov = res["coverage"]
    assert cov["searchable"] == 3 and cov["total"] == 3


def test_unrelated_clip_says_no_verified_match(built_db):
    pid = built_db["raja"]
    gibberish = [(0.0, 5.0, "Quantum bananas orbit the silent "
                             "watermelon factory at midnight")]
    res = analyze.analyze_clip_full(
        pid, Path("clip.wav"),
        transcribe_fn=_stub_tf(gibberish, lang="en"))
    assert res["ok"] is True
    assert res["results"] == []
    assert res["note"] == "No verified match in this playlist."


# ------------------------------------------------------ visual compare

def test_visual_compare_identical_and_different(tmp_path):
    img = Image.new("RGB", (320, 180), (200, 30, 30))
    for x in range(0, 320, 40):
        for y in range(0, 180, 40):
            img.paste((x, y, 90), (x, y, x + 20, y + 20))
    p = tmp_path / "f.png"
    img.save(p)
    h = [analyze._dhash(img)]
    same = analyze._visual_compare(
        h, {"available": True, "frames": [{"path": str(p)}]})
    assert same["status"] == "similar" and same["best_distance"] == 0
    noise = Image.effect_noise((320, 180), 90).convert("RGB")
    np_ = tmp_path / "n.png"
    noise.save(np_)
    diff = analyze._visual_compare(
        h, {"available": True, "frames": [{"path": str(np_)}]})
    assert diff["status"] in ("different", "similar")
    assert diff["best_distance"] is not None
    un = analyze._visual_compare(
        h, {"available": False, "frames": [], "reason": "no previews"})
    assert un["status"] == "unavailable"


# --------------------------------------------------------- silent clip

def test_silent_clip_visual_scan_labels_hints(built_db, monkeypatch,
                                              tmp_path):
    pid = built_db["raja"]
    img = Image.new("RGB", (320, 180), (40, 80, 160))
    fp = tmp_path / "sb.png"
    img.save(fp)
    monkeypatch.setattr(analyze, "_clip_frames", lambda p: [img])
    from backend import visual
    monkeypatch.setattr(
        visual, "episode_frames",
        lambda yid, count=6: {"available": True,
                              "frames": [{"timestamp": 10.0,
                                          "hms": "00:00:10",
                                          "path": str(fp)}],
                              "reason": "stub"})
    silent = _stub_tf([], lang="ur", prob=0.4)
    res = analyze.analyze_clip_full(pid, Path("silent.mp4"),
                                    transcribe_fn=silent)
    assert res["ok"] is True
    assert res["results"], "identical frames should yield a hint"
    top = res["results"][0]
    assert top["confidence"] == "low"
    assert "HINT" in " ".join(top["reasons"])
    assert "verified" in (res["note"] or "").lower()


def test_silent_clip_no_frames_no_match(built_db, monkeypatch):
    pid = built_db["raja"]
    monkeypatch.setattr(analyze, "_clip_frames", lambda p: [])
    res = analyze.analyze_clip_full(
        pid, Path("silent.mp4"), transcribe_fn=_stub_tf([], lang="ur"))
    assert res["ok"] is False  # honest failure, not a fake match
    assert "describe" in res["error"]


# ------------------------------------------------------------ stats

def test_playlist_stats_counts_sources(tmp_path):
    old = db.DB_PATH
    db.DB_PATH = tmp_path / "stats.db"
    db.init_db()
    try:
        pid = db.create_playlist("https://youtube.com/playlist?list=ST",
                                 title="Stats")
        rows = [
            ("STAT0000001", "Ep 1", "captions"),
            ("STAT0000002", "Ep 2", "whisper"),
            ("STAT0000003", "Ep 3", "subtitle"),
            ("STAT0000004", "Private video", None),
            ("STAT0000005", "Ep 5", None),
        ]
        db.add_videos(pid, [
            {"youtube_id": y, "title": t, "position": i + 1,
             "duration": None, "url": "", "thumbnail": None}
            for i, (y, t, _s) in enumerate(rows)])
        for v, (_y, _t, src) in zip(db.get_videos(pid), rows):
            if src:
                db.mark_transcript(v["id"], True, source=src)
        st = db.playlist_stats(pid)
        assert st["total"] == 5 and st["searchable"] == 3
        assert st["missing"] == 2 and st["skipped"] == 1
        assert st["transcribed"] == 2
        assert st["by_source"] == {"captions": 1, "whisper": 1,
                                   "subtitle": 1}
    finally:
        db.DB_PATH = old


# ------------------------------------------------------------- app UI

def test_app_clip_section_renders(tmp_path):
    """The clip finder (uploader + button) renders for a ready
    playlist with the new universal pipeline wired in."""
    old = db.DB_PATH
    db.DB_PATH = tmp_path / "ui3.db"
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=UICLIP",
                             title="Clip Drama")
    db.update_playlist(pid, drama_name="Clip Drama")
    db.add_videos(pid, [{
        "youtube_id": "CLIP0000001", "title": "Clip Drama Episode 1",
        "position": 1, "duration": None,
        "url": "https://www.youtube.com/watch?v=CLIP0000001",
        "thumbnail": None}])
    v = db.get_videos(pid)[0]
    db.add_segments(v["id"], [(0.0, 2.0, "Some indexed dialogue line.")])
    db.mark_transcript(v["id"], True, source="captions")
    db.update_playlist(pid, status="ready", video_count=1)
    try:
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parent.parent
        at = AppTest.from_file(str(root / "streamlit" / "app.py"),
                               default_timeout=180)
        at.session_state["active_pid"] = pid
        at.session_state["playlist_picker"] = pid
        at.run()
        assert not at.exception, at.exception
        assert len(at.file_uploader) >= 1  # the clip uploader
        labels = [b.label for b in at.button]
        assert any("Find episode for this clip" in (l or "")
                   for l in labels), labels
    finally:
        db.DB_PATH = old
