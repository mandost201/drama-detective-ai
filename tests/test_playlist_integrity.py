"""Playlist integrity tests (Command 1).

Identity rules under test:
  * A playlist's identity is its canonical source id (YouTube
    playlist/video id) + its own pid - never its display title.
  * Every search / transcript / cache entry is bound to those ids;
    two unrelated dramas can be switched between, refreshed over,
    and reopened in new sessions without any data bleeding across.
  * Invalid or unreachable sources end in an explicit error state,
    never in a silent fall-back to another drama's data.
"""
from pathlib import Path

import pytest

from backend import db, ingest, search, selection as sel, youtube


# ------------------------------------------------------------- identity

def test_source_id_canonical_forms():
    assert db.source_id_for(
        "https://youtube.com/playlist?list=PLabc123_-X") == "pl:PLabc123_-X"
    assert db.source_id_for(
        "https://www.youtube.com/watch?v=YDu9akegEHU&t=9s"
    ) == "vi:YDu9akegEHU"
    assert db.source_id_for("https://youtu.be/YDu9akegEHU") == \
        "vi:YDu9akegEHU"
    assert db.source_id_for("https://example.com/nope") is None
    assert db.source_id_for("") is None


def test_find_by_source_and_duplicate_titles(tmp_path):
    old = db.DB_PATH
    db.DB_PATH = tmp_path / "ident.db"
    db.init_db()
    try:
        url = "https://youtube.com/playlist?list=PLDUPTITLE1"
        p1 = db.create_playlist(url, title="Fetched Title One")
        db.update_playlist(p1, drama_name="Same Name")
        p2 = db.create_playlist(
            "https://youtube.com/playlist?list=PLDUPTITLE2",
            title="Fetched Title Two")
        db.update_playlist(p2, drama_name="Same Name")
        # same display name, different sources: identities stay apart
        assert db.get_playlist(p1)["source_id"] == "pl:PLDUPTITLE1"
        assert db.get_playlist(p2)["source_id"] == "pl:PLDUPTITLE2"
        assert [p["id"] for p in db.find_by_source("pl:PLDUPTITLE1")] == [p1]
        labels = sel.unique_labels(db.list_playlists())
        assert labels[p1] != labels[p2]
        # a second analysis of the SAME source is findable as such,
        # and is a separate row - never silently merged
        p3 = db.create_playlist(url)
        assert p3 != p1
        assert {p["id"] for p in db.find_by_source("pl:PLDUPTITLE1")} == \
            {p1, p3}
    finally:
        db.DB_PATH = old


def test_confirmed_identity_uses_fetched_title():
    row = {"title": "Raja London Ka | Full Episodes",
           "source_id": "pl:PLeb83", "video_count": 32,
           "drama_name": "User Typed Name"}
    line = sel.confirmed_identity(row)
    assert "Raja London Ka | Full Episodes" in line
    assert "32 videos" in line and "pl:PLeb83" in line
    assert sel.confirmed_identity({"title": None}) == ""


# ---------------------------------------------------------- invalid URL

def test_fetch_playlist_rejects_invalid_urls():
    for bad in ("", "not a url", "https://example.com/foo",
                "https://www.youtube.com/watch",
                "https://www.youtube.com/feed/subscriptions"):
        with pytest.raises(RuntimeError):
            youtube.fetch_playlist(bad)


def test_failed_analysis_lands_in_error_state(tmp_path, monkeypatch):
    """An unreachable/private source must end as status='error' -
    never ready, never searchable, never a silent substitute."""
    old = db.DB_PATH
    db.DB_PATH = tmp_path / "err.db"
    db.init_db()
    try:
        pid = db.create_playlist(
            "https://youtube.com/playlist?list=PLPRIVATE00")
        jid = db.create_job("playlist", pid)

        def boom(url):
            raise RuntimeError("Could not read that playlist: "
                               "Private video. It may be private, "
                               "deleted, or region-restricted.")
        monkeypatch.setattr(ingest, "fetch_playlist", boom)
        ingest.run_playlist_job(
            jid, pid, "https://youtube.com/playlist?list=PLPRIVATE00")
        assert db.get_playlist(pid)["status"] == "error"
        assert db.get_job(jid)["status"] == "error"
        assert search.playlist_coverage(pid)["searchable"] == 0
    finally:
        db.DB_PATH = old


def test_identity_notice_only_when_submitted_unready():
    ready = {"id": "A", "status": "ready", "drama_name": "Raja",
             "title": "Raja Full"}
    failed = {"id": "B", "status": "error", "drama_name": "Wehshi",
              "title": None}
    by_id = {"A": ready, "B": failed}
    # submitted failed, selection is the other (ready) one -> notice
    n = sel.identity_notice({"pid": "B", "name": "Wehshi"}, "A", by_id)
    assert n and "Wehshi" in n and "Raja" in n
    # submitted is ready -> switching away is explicit, no notice
    assert sel.identity_notice({"pid": "A", "name": "Raja"}, "B",
                               by_id) is None
    # nothing submitted / submitted == selected -> no notice
    assert sel.identity_notice(None, "A", by_id) is None
    assert sel.identity_notice({"pid": "B", "name": "Wehshi"}, "B",
                               by_id) is None


# ------------------------------------------- switching / scope isolation

def test_search_and_clip_never_cross_playlists(built_db):
    raja, ert = built_db["raja"], built_db["ert"]
    # Ertugrul's distinctive dialogue is invisible inside Raja scope
    r = search.search(raja, "Suleman Shah ki death Ertugrul")
    assert all(x["youtube_id"] != "qA4r7LDmnqY" for x in
               r["results"] + r["alternatives"])
    # ...and Raja's lawyer scene is invisible inside Ertugrul scope
    r2 = search.search(ert, "Rania Mirza ye shaadi nahi ho sakti vakil")
    assert all(x["youtube_id"] == "qA4r7LDmnqY" for x in
               r2["results"] + r2["alternatives"])
    # coverage is reported per playlist, from its own rows only
    assert search.playlist_coverage(raja)["total"] == 3
    assert search.playlist_coverage(ert)["total"] == 1


def test_stale_cache_cleared_on_data_change(tmp_path):
    """On a DEDICATED database (never the shared fixtures): after a
    video's segments are replaced and the cache invalidated - the
    contract every data-changing path in the app follows - searches
    must reflect the new data, not the stale transcript."""
    from backend.vtt import parse_vtt
    old = db.DB_PATH
    db.DB_PATH = tmp_path / "stale.db"
    db.init_db()
    try:
        pid = db.create_playlist(
            "https://youtube.com/playlist?list=PLSTALE000", title="Stale")
        db.add_videos(pid, [{
            "youtube_id": "YDu9akegEHU", "title": "Ep 25", "position": 25,
            "duration": None,
            "url": "https://www.youtube.com/watch?v=YDu9akegEHU",
            "thumbnail": None}])
        v = db.get_videos(pid)[0]
        fixtures = Path(__file__).resolve().parent / "fixtures"
        segs = parse_vtt((fixtures / "raja_ep25.en.vtt")
                         .read_text(encoding="utf-8"))
        db.add_segments(v["id"], segs)
        db.mark_transcript(v["id"], True, source="captions")
        db.update_playlist(pid, status="ready", video_count=1)
        before = search.search(pid, "Rania Mirza")
        assert before["results"]
        assert before["results"][0]["youtube_id"] == "YDu9akegEHU"
        db.add_segments(v["id"], [(0.0, 3.0, "Completely different "
                                             "dialogue about trains.")])
        search.invalidate_cache(pid)
        after = search.search(pid, "Rania Mirza")
        assert all(x["youtube_id"] != "YDu9akegEHU" for x in
                   after["results"] + after["alternatives"]), \
            "stale transcript data leaked into a new search"
    finally:
        db.DB_PATH = old


# ------------------------------------------- refresh / new session (UI)

def _two_playlist_db(tmp_path):
    old = db.DB_PATH
    db.DB_PATH = tmp_path / "ui.db"
    db.init_db()
    made = {}
    for key, yid, name, n in (("small", "SMALL000001", "Small Drama", 1),
                              ("big", "BIG00000001", "Big Drama", 3)):
        pid = db.create_playlist(
            f"https://youtube.com/playlist?list=PL{key.upper()}",
            title=f"{name} | Full")
        db.update_playlist(pid, drama_name=name)
        db.add_videos(pid, [{
            "youtube_id": yid, "title": f"{name} Episode 1",
            "position": 1, "duration": None,
            "url": f"https://www.youtube.com/watch?v={yid}",
            "thumbnail": None}])
        v = db.get_videos(pid)[0]
        db.add_segments(v["id"], [(0.0, 3.0, f"{name} khaas dialogue "
                                             "line number one.")])
        db.mark_transcript(v["id"], True, source="captions")
        db.update_playlist(pid, status="ready", video_count=n)
        made[key] = pid
    return old, made


def test_app_new_session_defaults_to_largest_ready(tmp_path):
    """A brand-new session (page refresh / reopened app) must land
    on a deterministic selection - the largest ready playlist -
    with the identity of that playlist shown, and searches bound
    to it."""
    old, made = _two_playlist_db(tmp_path)
    try:
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parent.parent
        at = AppTest.from_file(str(root / "streamlit" / "app.py"),
                               default_timeout=180)
        at.run()  # NO preset session state: a truly fresh session
        assert not at.exception, at.exception
        assert at.session_state["playlist_picker"] == made["big"]
        captions = [c.value for c in at.caption]
        assert any("Confirmed:" in (c or "") and "Big Drama" in c
                   for c in captions), captions
    finally:
        db.DB_PATH = old


def test_app_switching_playlists_rebinds_search(tmp_path):
    old, made = _two_playlist_db(tmp_path)
    try:
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parent.parent
        at = AppTest.from_file(str(root / "streamlit" / "app.py"),
                               default_timeout=180)
        at.session_state["active_pid"] = made["small"]
        at.session_state["playlist_picker"] = made["small"]
        at.run()
        assert not at.exception, at.exception
        assert at.session_state["playlist_picker"] == made["small"]
        # run a search in this scope; results must belong to it
        at.text_area[0].set_value("Small Drama khaas dialogue")
        [b for b in at.button if "Find scene" in (b.label or "")][0] \
            .click().run()
        assert not at.exception, at.exception
        assert at.session_state["results_pid"] == made["small"]
        res = at.session_state["last_results"]
        assert res["coverage"]["total"] == 1  # Small Drama has 1 video
        assert all(r["youtube_id"] == "SMALL000001"
                   for r in res["results"] + res["alternatives"])
    finally:
        db.DB_PATH = old
