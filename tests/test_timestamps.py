"""Timestamp & scene-boundary verification.

The contract under test:
- the reported scene start is the scene's start (dialogue block /
  focus-trimmed estimate), NEVER just the first keyword hit - the
  matched line's own time is kept separately as evidence;
- the watch link opens at the scene start, and the matched line
  lies inside the reported range;
- scenes spanning many segments, scenes at the very start/end of
  a video, and repeated dialogue all resolve to the right block;
- a clip that starts mid-scene says so, and reports the complete
  scene range separately from its matched window;
- boundaries are labelled transcript estimates unless storyboard
  frames actually show a visual cut at both edges - no invented
  precision.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend import analyze, db, search

SEGS = [
    (0.0, 4.0, "pehli line shuru ki"),
    (5.0, 9.0, "dusri line yahin"),
    # 30s silence -> scene break
    (39.0, 43.0, "Wakeel sahab aa gaye"),
    (45.0, 49.0, "Rania Mirza aa gayi hain"),
    (51.0, 55.0, "ye shaadi nahi ho sakti"),
    (58.0, 62.0, "sab hairan reh gaye"),
    # 38s silence -> scene break
    (100.0, 104.0, "aakhri line video ki"),
]

SEGS_REPEAT = [
    (10.0, 14.0, "tum yahan kya kar rahe ho"),
    (16.0, 20.0, "mein bas guzar raha tha"),
    # long gap -> another scene, same generic line
    (200.0, 204.0, "tum yahan kya kar rahe ho"),
    (206.0, 210.0, "Rania Mirza tum yahan"),
]


@pytest.fixture()
def ts_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "ts.db")
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=TS",
                             title="TS Drama")
    db.add_videos(pid, [
        {"youtube_id": "TSVIDEO0001", "title": "TS Episode 3",
         "position": 3, "duration": None,
         "url": "https://www.youtube.com/watch?v=TSVIDEO0001",
         "thumbnail": None},
        {"youtube_id": "TSVIDEO0002", "title": "TS Episode 4",
         "position": 4, "duration": None,
         "url": "https://www.youtube.com/watch?v=TSVIDEO0002",
         "thumbnail": None}])
    vids = {v["youtube_id"]: v for v in db.get_videos(pid)}
    db.add_segments(vids["TSVIDEO0001"]["id"], SEGS)
    db.mark_transcript(vids["TSVIDEO0001"]["id"], True,
                       source="captions")
    db.add_segments(vids["TSVIDEO0002"]["id"], SEGS_REPEAT)
    db.mark_transcript(vids["TSVIDEO0002"]["id"], True,
                       source="captions")
    return pid, vids


def test_block_multi_segment_and_edges(ts_db):
    _pid, vids = ts_db
    v1 = vids["TSVIDEO0001"]["id"]
    # a scene spanning four segments with small pauses is ONE block
    assert search.scene_block(v1, 47.0) == (39.0, 62.0)
    # scene at the very beginning of the video
    assert search.scene_block(v1, 2.0) == (0.0, 9.0)
    # scene at the very end of the video
    assert search.scene_block(v1, 102.0) == (100.0, 104.0)


def test_scene_start_is_not_the_keyword_hit(ts_db):
    """Regression for the old behaviour: the result's clock used to
    point at the matched segment (45s). Now the link and the scene
    range start at the scene's start (39s), and the matched line
    keeps its own timestamp as evidence, inside the range."""
    pid, _vids = ts_db
    res = search.search(pid, "Rania Mirza")
    # both test episodes contain the name; examine episode 3's result
    top = next(r for r in res["results"] if r["episode"] == 3)
    assert top["matched_at"] == 45.0
    assert top["scene_start"] == 39.0
    assert top["scene_end"] == 62.0
    # the focus window around the distinctive line is tighter and
    # reported separately - it is not passed off as the whole scene
    assert top["focus_start"] == 39.0
    assert top["focus_end"] == 55.0
    assert top["block_start"] == 39.0 and top["block_end"] == 62.0
    assert top["scene_start"] <= top["matched_at"] <= top["scene_end"]
    assert top["url_at"].endswith("t=39s")
    assert top["boundary"]["kind"] == "transcript_estimate"


def test_repeated_dialogue_picks_the_right_block(ts_db):
    pid, _vids = ts_db
    res = search.search(pid, "Rania Mirza yahan")
    top = res["results"][0]
    assert top["episode"] == 4
    # the generic line also occurs at 10s; the block containing the
    # distinctive name (200s) is the one that must be reported
    assert top["block_start"] == 200.0
    assert top["matched_at"] == 206.0


def _noise_img(path, seed):
    arr = np.random.RandomState(seed).randint(
        0, 255, (64, 64, 3), dtype="uint8")
    Image.fromarray(arr).save(path)
    return str(path)


def test_verify_boundaries_frame_checked(tmp_path):
    frames = [
        {"timestamp": 92.0, "path": _noise_img(tmp_path / "a.png", 1)},
        {"timestamp": 102.0, "path": _noise_img(tmp_path / "b.png", 2)},
        {"timestamp": 158.0, "path": _noise_img(tmp_path / "c.png", 3)},
        {"timestamp": 168.0, "path": _noise_img(tmp_path / "d.png", 4)},
    ]
    out = analyze.verify_boundaries("X", 100.0, 160.0, frames=frames)
    assert out["start_cut"] is True and out["end_cut"] is True
    assert out["kind"] == "frame_checked"


def test_verify_boundaries_continuity_stays_estimate(tmp_path):
    same = _noise_img(tmp_path / "same.png", 7)
    frames = [
        {"timestamp": 92.0, "path": same},
        {"timestamp": 102.0, "path": same},   # same shot across start
        {"timestamp": 158.0, "path": _noise_img(tmp_path / "e.png", 5)},
        {"timestamp": 168.0, "path": _noise_img(tmp_path / "f.png", 6)},
    ]
    out = analyze.verify_boundaries("X", 100.0, 160.0, frames=frames)
    assert out["start_cut"] is False
    assert out["end_cut"] is True
    # one continuous edge is enough to deny the frame-verified label
    assert out["kind"] == "transcript_estimate"
    assert "shuruaat" in out["detail"]


def test_verify_boundaries_unavailable_is_honest(monkeypatch):
    from backend import visual
    monkeypatch.setattr(visual, "boundary_frames",
                        lambda *a: {"available": False, "frames": [],
                                    "reason": "no storyboard"})
    out = analyze.verify_boundaries("X", 100.0, 160.0)
    assert out["kind"] == "transcript_estimate"
    assert out["status"] == "unavailable"
    assert out["start_cut"] is None and out["end_cut"] is None


def test_clip_starting_mid_scene_says_so(built_db, monkeypatch):
    monkeypatch.setattr(analyze, "_visual_verify",
                        lambda *a, **k: {"status": "unavailable",
                                         "best_distance": None,
                                         "detail": "test stub"})
    monkeypatch.setattr(analyze, "verify_boundaries",
                        lambda *a, **k: {"kind": "transcript_estimate",
                                         "status": "unavailable",
                                         "start_cut": None,
                                         "end_cut": None,
                                         "detail": "test stub"})
    late = [(0.0, 18.0,
             "Rania Mirza this marriage cannot take place what are "
             "you saying I will not let this wedding happen")]

    def tf(path, task="transcribe"):
        return late, "en", 0.99

    res = analyze.analyze_clip_full(built_db["raja"], Path("c.wav"),
                                    transcribe_fn=tf)
    assert res["results"], res
    top = res["results"][0]
    assert top["youtube_id"] == "YDu9akegEHU"
    # the matched window is the precise claim; the scene around it
    # starts materially earlier and is reported separately
    assert top["match_start"] == top["timestamp"]
    assert top["scene_start"] < top["timestamp"] - 20
    assert top["block_start"] <= top["scene_start"]
    assert top["url_at"].endswith(f"t={int(top['timestamp'])}s")
    assert any("darmiyan" in r for r in top["reasons"]), top["reasons"]
    assert top["boundary"]["kind"] == "transcript_estimate"
