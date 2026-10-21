"""Missing-captions workflow tests.

The limitation being fixed: episodes whose YouTube captions are not
in English were declared "no captions" (the fetcher asked YouTube
for en.* only), and the Whisper fallback ran in an opaque thread
with no progress, no errors and no per-episode outcome. These tests
cover the real fallbacks: multilingual caption fetch, user subtitle
files (.srt/.vtt) indexed per episode, tracked Whisper transcription
with a truthful report, and the visual frame fallback.
"""
import subprocess
from pathlib import Path

import pytest

from backend import db, ingest, search, visual, youtube
from backend.vtt import parse_srt, parse_subtitles

SRT_SAMPLE = """1
00:00:01,000 --> 00:00:03,500
Neelam dupatta wali larki darwaze par khari hai.

2
00:00:04,000 --> 00:00:06,000
Woh kehti hai, main tum se mohabbat karti hoon.

3
00:00:06,500 --> 00:00:08,000
Haan.
"""

URDU_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
Ye pehli line hai

00:00:04.000 --> 00:00:06.000
Ye doosri line hai
"""


# ------------------------------------------------------------- parsers

def test_parse_srt_timestamps_and_text():
    segs = parse_srt(SRT_SAMPLE)
    assert len(segs) == 3
    assert segs[0][0] == 1.0 and segs[0][1] == 3.5
    assert "Neelam dupatta" in segs[0][2]
    assert segs[2][2] == "Haan."  # short repeats are real dialogue


def test_parse_subtitles_dispatch():
    assert len(parse_subtitles(SRT_SAMPLE, "ep1.srt")) == 3
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello world\n"
    assert parse_subtitles(vtt, "ep1.vtt")[0][2] == "Hello world"
    # no extension -> content sniffing still finds the SRT cues
    assert len(parse_subtitles(SRT_SAMPLE, "")) == 3


# ------------------------------------------------- caption fetch (i18n)

def test_track_language_preference():
    assert youtube._track_lang("YDu9akegEHU.en-US.vtt", "YDu9akegEHU") == "en-US"
    assert youtube._track_lang("YDu9akegEHU.ur.vtt", "YDu9akegEHU") == "ur"
    files = [Path("YDu9akegEHU.hi.vtt"), Path("YDu9akegEHU.ur.vtt"),
             Path("YDu9akegEHU.en.vtt")]
    assert youtube._pick_track(files, "YDu9akegEHU").name.endswith(".en.vtt")
    files = [Path("YDu9akegEHU.hi.vtt"), Path("YDu9akegEHU.ur.vtt")]
    assert youtube._pick_track(files, "YDu9akegEHU").name.endswith(".ur.vtt")
    assert youtube._pick_track([], "x") is None


def test_fetch_transcript_uses_non_english_tracks(monkeypatch):
    """An episode whose only captions are Urdu must NOT be reported
    as caption-less (the old fetcher requested en.* only)."""
    seen_cmds = []

    def fake_run(cmd, **kwargs):
        seen_cmds.append(cmd)
        out = cmd[cmd.index("-o") + 1]
        target = out.replace("%(id)s", "URDUVID0001").replace("%(ext)s", "vtt")
        langs = cmd[cmd.index("--sub-langs") + 1]
        if "ur" in langs:  # YouTube "has" an Urdu track for this video
            Path(target).write_text(URDU_VTT, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(youtube.subprocess, "run", fake_run)
    segs = youtube.fetch_transcript("URDUVID0001")
    assert segs and len(segs) == 2
    assert "pehli line" in segs[0][2]
    # and the request really asked for more than English
    first = seen_cmds[0]
    assert "ur" in first[first.index("--sub-langs") + 1]


LIST_SUBS_SAMPLE = """[youtube] abc: Downloading webpage
Available automatic captions for abc:
Language Name                  Formats
en       English               vtt, srt
ur       Urdu                  vtt, srt
ar       Arabic                vtt, srt
fa       Persian               vtt, srt

Available subtitles for abc:
Language Name       Formats
hi       Hindi      vtt, srt
"""


def test_probe_other_language_picks_beyond_pref(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, LIST_SUBS_SAMPLE, "")

    monkeypatch.setattr(youtube.subprocess, "run", fake_run)
    # en/ur/hi are pass-1 languages; the probe must offer another
    assert youtube._probe_other_language("abc") == "ar"


def test_probe_other_language_none_when_only_preferred(monkeypatch):
    only = ("Available subtitles for abc:\n"
            "Language Name       Formats\n"
            "en       English    vtt\nur       Urdu       vtt\n")
    monkeypatch.setattr(
        youtube.subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, only, ""))
    assert youtube._probe_other_language("abc") is None


# ------------------------------------------- subtitle upload indexing

@pytest.fixture(scope="module")
def missing_db(built_db):
    pid = db.create_playlist("https://youtube.com/playlist?list=SUBTEST",
                             title="Subtitle Drama")
    db.update_playlist(pid, drama_name="Subtitle Drama")
    db.add_videos(pid, [{
        "youtube_id": "SUBVID00001", "title": "Subtitle Drama Episode 1",
        "position": 1, "duration": None,
        "url": "https://www.youtube.com/watch?v=SUBVID00001",
        "thumbnail": None}])
    v = db.get_videos(pid)[0]
    db.update_playlist(pid, status="ready", video_count=1)
    return {"pid": pid, "video": v}


def test_subtitle_upload_makes_episode_searchable(missing_db):
    pid, v = missing_db["pid"], missing_db["video"]
    assert not v["has_transcript"]
    assert search.playlist_coverage(pid)["searchable"] == 0
    n = ingest.index_subtitle_text(pid, v["id"], SRT_SAMPLE, "ep1.srt")
    assert n == 3
    v2 = [x for x in db.get_videos(pid) if x["id"] == v["id"]][0]
    assert v2["has_transcript"]
    assert search.playlist_coverage(pid)["searchable"] == 1
    res = search.search(pid, "neelam dupatta wali larki darwaze par")
    assert res["results"]
    assert res["results"][0]["youtube_id"] == "SUBVID00001"
    assert res["results"][0]["timestamp"] <= 3.5


def test_bad_subtitle_file_rejected_and_not_searchable(missing_db):
    pid = missing_db["pid"]
    db.add_videos(pid, [{
        "youtube_id": "SUBVID00002", "title": "Subtitle Drama Episode 2",
        "position": 2, "duration": None,
        "url": "https://www.youtube.com/watch?v=SUBVID00002",
        "thumbnail": None}])
    v = [x for x in db.get_videos(pid)
         if x["youtube_id"] == "SUBVID00002"][0]
    with pytest.raises(ValueError):
        ingest.index_subtitle_text(pid, v["id"], "no cues here", "x.srt")
    v2 = [x for x in db.get_videos(pid) if x["id"] == v["id"]][0]
    assert not v2["has_transcript"]


# -------------------------------------------- whisper job orchestration

def test_transcription_report_done_failed_and_marking(missing_db,
                                                      monkeypatch):
    pid = missing_db["pid"]
    for pos, yid in ((3, "TXOK0000001"), (4, "TXBAD000001")):
        db.add_videos(pid, [{
            "youtube_id": yid, "title": f"Subtitle Drama Episode {pos}",
            "position": pos, "duration": None,
            "url": f"https://www.youtube.com/watch?v={yid}",
            "thumbnail": None}])

    def fake_download(yid, out_path):
        Path(out_path).write_bytes(b"0" * 60000)
        return Path(out_path)

    def fake_transcribe(path):
        if "TXBAD" in str(path):
            raise RuntimeError("boom: no audio stream")
        return [(0.0, 2.0, "Transcribed dialogue line.")]

    monkeypatch.setattr(youtube, "download_audio", fake_download)
    from backend import analyze
    monkeypatch.setattr(analyze, "transcribe_audio_file", fake_transcribe)

    ids = {v["id"] for v in db.get_videos(pid)
           if v["youtube_id"] in ("TXOK0000001", "TXBAD000001")}
    rep = ingest.transcribe_missing_episodes(pid, video_ids=ids)
    by_id = {v["youtube_id"]: v for v in db.get_videos(pid)}
    assert rep["done"] == 1
    assert len(rep["failed"]) == 1
    assert "boom" in rep["failed"][0]["error"]
    # marked searchable ONLY after a real transcript was indexed
    assert by_id["TXOK0000001"]["has_transcript"]
    assert not by_id["TXBAD000001"]["has_transcript"]
    assert ingest.last_transcription_report(pid)["done"] == 1


# ------------------------------------------------------- visual fallback

def test_episode_frames_evenly_spaced(monkeypatch, tmp_path):
    from PIL import Image
    monkeypatch.setattr(visual, "_SB_DIR", tmp_path)
    monkeypatch.setattr(visual, "_metadata", lambda vid: (100.0, 320, 180))
    monkeypatch.setattr(visual, "_download_storyboard",
                        lambda vid: tmp_path / "sb.mhtml")
    sheet = Image.new("RGB", (960, 540), (30, 60, 90))
    monkeypatch.setattr(visual, "_parse_sheets",
                        lambda p: ([sheet], [(0.0, 100.0)]))
    out = visual.episode_frames("SOMEVID0001", count=4)
    assert out["available"] is True
    assert [f["timestamp"] for f in out["frames"]] == [12.5, 37.5, 62.5, 87.5]
    assert all(Path(f["path"]).exists() for f in out["frames"])


def test_episode_frames_unavailable_is_honest(monkeypatch):
    monkeypatch.setattr(visual, "_metadata", lambda vid: None)
    out = visual.episode_frames("SOMEVID0002")
    assert out["available"] is False and out["frames"] == []


# ------------------------------------------------------------- app panel

@pytest.fixture()
def missing_ui_db(tmp_path):
    old = db.DB_PATH
    db.DB_PATH = tmp_path / "ui2.db"
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=UICAP",
                             title="Panel Drama")
    db.update_playlist(pid, drama_name="Panel Drama")
    db.add_videos(pid, [{
        "youtube_id": "PANEL000001", "title": "Panel Drama Episode 1",
        "position": 1, "duration": None,
        "url": "https://www.youtube.com/watch?v=PANEL000001",
        "thumbnail": None}])
    db.update_playlist(pid, status="ready", video_count=1)
    yield pid
    db.DB_PATH = old


def test_app_missing_captions_panel_renders(missing_ui_db):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parent.parent
    at = AppTest.from_file(str(root / "streamlit" / "app.py"),
                           default_timeout=180)
    at.session_state["active_pid"] = missing_ui_db
    at.session_state["playlist_picker"] = missing_ui_db
    at.run()
    assert not at.exception, at.exception
    button_labels = [b.label for b in at.button]
    assert any("transcribe karein" in (l or "") for l in button_labels), \
        button_labels
    warnings = [w.value for w in at.warning]
    assert any("captions YouTube par" in (w or "") for w in warnings), \
        warnings
