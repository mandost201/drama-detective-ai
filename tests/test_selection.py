"""Regression tests for the wrong-playlist bug (2026-10-21).

Incident: the user submitted a new drama's playlist, but the app's
playlist picker kept its previous value (Streamlit ignores a
selectbox `index` once a value exists, and the old code keyed options
by display label), so `active_pid` was overwritten with the old
playlist and the clip finder matched a clip against the WRONG drama,
presenting a 40% semantic hit as "Found it!".

These tests lock the fixed data flow:
  * selection policy is by stable playlist ID (backend.selection);
  * search/alignment are strictly scoped to the selected playlist;
  * the app UI itself selects the just-analyzed playlist and shows
    the clip banner for THAT drama.
"""
from pathlib import Path

import pytest

from backend import db, search, selection
from backend.align import align_against_playlist
from backend.vtt import parse_vtt

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent
EP25_CLIP_TEXT = ("Lawyer, the groom and bridegroom are here. "
                  "Rania Mirza, this marriage cannot take place.")
EP25_QUERY = ("Rania Mirza, this marriage cannot take place. Lawyer, "
              "no judge will come here. This is the money for your "
              "fees and your tea.")


# ---------------------------------------------------------------- policy

def _p(pid, name, count):
    return {"id": pid, "drama_name": name, "title": None,
            "video_count": count, "url": "https://youtube.com/playlist?list=X"}


def test_unique_labels_never_collapse_playlists():
    labels = selection.unique_labels([_p(1, "Wehshi", 20), _p(2, "Wehshi", 20),
                                      _p(3, "Raja London Ka", 1)])
    assert len(labels) == 3
    assert labels[1] != labels[2]  # same name+count -> disambiguated
    assert labels[3] == "Raja London Ka (1 videos)"


def test_default_selection_prefers_active_then_largest():
    ready = [_p(1, "Raja London Ka", 1), _p(2, "Wehshi", 32)]
    assert selection.default_selection(ready, active_pid=1) == 1
    assert selection.default_selection(ready, active_pid=None) == 2
    assert selection.default_selection(ready, active_pid=999) == 2
    assert selection.default_selection([], active_pid=1) is None


def test_stale_results_detection():
    assert selection.stale_results(results_pid=7, selected_pid=7) is False
    assert selection.stale_results(results_pid=7, selected_pid=9) is True
    assert selection.stale_results(results_pid=None, selected_pid=9) is False


def test_clip_block_reason():
    # nothing submitted -> never blocked
    assert selection.clip_block_reason(None, 1, {1, 2}) is None
    # submitted IS the selection -> fine
    assert selection.clip_block_reason({"pid": 2, "name": "Wehshi"}, 2, {1, 2}) is None
    # submitted ready, user deliberately switched away -> allowed
    assert selection.clip_block_reason({"pid": 2, "name": "Wehshi"}, 1, {1, 2}) is None
    # submitted never became ready, selection is another drama -> BLOCK
    msg = selection.clip_block_reason({"pid": 2, "name": "Wehshi"}, 1, {1})
    assert msg and "Wehshi" in msg


# ------------------------------------------------------- scope isolation

@pytest.fixture(scope="module")
def wehshi_db(built_db):
    """A second drama's playlist whose episodes must never leak into
    (or be leaked from) the Raja London Ka search scope."""
    ep24 = parse_vtt((FIXTURES / "raja_ep24.en.vtt").read_text("utf-8"))
    pid = db.create_playlist("https://youtube.com/playlist?list=SELTEST",
                             title="Wehshi")
    db.update_playlist(pid, drama_name="Wehshi")
    db.add_videos(pid, [{
        "youtube_id": "WEHSHI00001", "title": "Wehshi Episode 1",
        "position": 1, "duration": None,
        "url": "https://www.youtube.com/watch?v=WEHSHI00001",
        "thumbnail": None}])
    v = db.get_videos(pid)[0]
    db.add_segments(v["id"], ep24)  # different dialogue entirely
    db.mark_transcript(v["id"], True)
    db.update_playlist(pid, status="ready", video_count=1)
    return pid


def test_search_scope_is_the_selected_playlist(built_db, wehshi_db):
    res = search.search(wehshi_db, EP25_QUERY)
    ids = [r["youtube_id"] for r in res["results"] + res["alternatives"]]
    assert "YDu9akegEHU" not in ids  # Raja EP25 is out of scope here


def test_clip_alignment_scope_is_the_selected_playlist(wehshi_db):
    pairs = [(v, db.get_segments(v["id"])) for v in db.get_videos(wehshi_db)]
    ranked = align_against_playlist(EP25_CLIP_TEXT, pairs)
    # EP25's dialogue does not exist in this playlist: no strong match
    assert not ranked or ranked[0][0] < 0.5


# ------------------------------------------------------------- app UI

@pytest.fixture()
def two_playlists(tmp_path):
    """A throwaway DB with two ready dramas, Raja + Wehshi."""
    old = db.DB_PATH
    db.DB_PATH = tmp_path / "ui.db"
    db.init_db()
    ep25 = parse_vtt((FIXTURES / "raja_ep25.en.vtt").read_text("utf-8"))
    ep24 = parse_vtt((FIXTURES / "raja_ep24.en.vtt").read_text("utf-8"))

    def make(url, name, yid, segs, count):
        pid = db.create_playlist(url, title=name)
        db.update_playlist(pid, drama_name=name)
        db.add_videos(pid, [{
            "youtube_id": yid, "title": f"{name} Episode 1",
            "position": 1, "duration": None,
            "url": f"https://www.youtube.com/watch?v={yid}",
            "thumbnail": None}])
        v = db.get_videos(pid)[0]
        db.add_segments(v["id"], segs)
        db.mark_transcript(v["id"], True)
        db.update_playlist(pid, status="ready", video_count=count)
        return pid

    raja = make("https://youtube.com/playlist?list=UIA", "Raja London Ka",
                "YDu9akegEHU", ep25, 1)
    wehshi = make("https://youtube.com/playlist?list=UIB", "Wehshi",
                  "WEHSHI00001", ep24, 2)
    yield {"raja": raja, "wehshi": wehshi}
    db.DB_PATH = old


def _banner_texts(at):
    texts = []
    for attr in ("info", "markdown", "caption", "text", "warning"):
        for el in getattr(at, attr, []) or []:
            val = getattr(el, "value", None)
            if isinstance(val, str):
                texts.append(val)
    return texts


def test_app_picker_selects_just_analyzed_playlist(two_playlists):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "streamlit" / "app.py"),
                           default_timeout=180)
    # what the Analyze handler leaves behind for a fresh analysis
    at.session_state["active_pid"] = two_playlists["wehshi"]
    at.session_state["playlist_picker"] = two_playlists["wehshi"]
    at.run()
    assert not at.exception, at.exception
    pickers = [s for s in at.selectbox if s.label == "📚 Playlist"]
    assert pickers, "playlist picker not rendered"
    assert pickers[0].value == two_playlists["wehshi"]
    banners = _banner_texts(at)
    assert any("Clip will be matched against" in t and "Wehshi" in t
               for t in banners), banners


def test_app_picker_respects_explicit_choice(two_playlists):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "streamlit" / "app.py"),
                           default_timeout=180)
    # user deliberately picked Raja although Wehshi was analyzed last
    at.session_state["playlist_picker"] = two_playlists["raja"]
    at.session_state["active_pid"] = two_playlists["wehshi"]
    at.run()
    assert not at.exception, at.exception
    pickers = [s for s in at.selectbox if s.label == "📚 Playlist"]
    assert pickers[0].value == two_playlists["raja"]
    banners = _banner_texts(at)
    assert any("Clip will be matched against" in t
               and "Raja London Ka" in t for t in banners), banners
