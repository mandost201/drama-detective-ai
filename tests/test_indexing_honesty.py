"""Indexing honesty (2026-10-22): a failed caption fetch is never
a silent skip. The failure reason is classified (bot-check / rate
limit / no captions exist), stored on the video row, shown in the
UI, and one serial retry sweep after the parallel burst recovers
episodes that were only rate-limited."""
from backend import db, ingest, youtube


def test_classify_bot_check_and_rate_limit():
    r = youtube._classify_fetch_failure(
        "ERROR: Sign in to confirm you're not a bot", None, "x")
    assert "bot-check" in r
    r = youtube._classify_fetch_failure(
        "HTTP Error 429: Too Many Requests", None, "x")
    assert "429" in r


def test_classify_no_captions_exist(monkeypatch):
    monkeypatch.setattr(youtube, "_has_any_caption_tracks",
                        lambda vid: False)
    r = youtube._classify_fetch_failure("some other error", None, "x")
    assert "mojood hi nahi" in r


def test_mark_transcript_records_and_clears_error(tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=T",
                             title="T")
    db.add_videos(pid, [{"youtube_id": "v1", "title": "Ep 1",
                         "position": 1, "duration": None, "url": "",
                         "thumbnail": None}])
    v = db.get_videos(pid)[0]
    assert "transcript_error" in v  # migration added the column
    db.mark_transcript(v["id"], False,
                       error="YouTube rate limit (HTTP 429)")
    v = db.get_videos(pid)[0]
    assert not v["has_transcript"]
    assert "429" in (v["transcript_error"] or "")
    db.mark_transcript(v["id"], True, source="captions", lang="hi")
    v = db.get_videos(pid)[0]
    assert v["has_transcript"]
    assert v["transcript_error"] is None


def test_retry_sweep_recovers_rate_limited_episode(tmp_path,
                                                   monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "s.db")
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=S",
                             title="S")
    db.add_videos(pid, [{"youtube_id": "ok1", "title": "Ep OK",
                         "position": 1, "duration": None, "url": "",
                         "thumbnail": None},
                        {"youtube_id": "rl1", "title": "Ep RL",
                         "position": 2, "duration": None, "url": "",
                         "thumbnail": None}])
    calls = {"rl1": 0}

    def fake_fetch(yid):
        if yid == "ok1":
            return [(0.0, 1.0, "hello")], "en"
        calls["rl1"] += 1
        if calls["rl1"] == 1:
            return None, None
        return [(0.0, 1.0, "recovered")], "hi"

    monkeypatch.setattr(ingest, "fetch_transcript_meta", fake_fetch)
    monkeypatch.setattr(ingest, "last_failure_reason",
                        lambda yid: "YouTube rate limit (HTTP 429)")
    monkeypatch.setattr(ingest, "fetch_playlist",
                        lambda url: ("S", [
                            {"youtube_id": "ok1", "title": "Ep OK",
                             "position": 1, "duration": None,
                             "url": "", "thumbnail": None},
                            {"youtube_id": "rl1", "title": "Ep RL",
                             "position": 2, "duration": None,
                             "url": "", "thumbnail": None}]))
    # embeddings are not under test here
    monkeypatch.setattr(ingest, "_index_embeddings",
                        lambda pid: None)
    jid = db.create_job("ingest", pid)
    ingest.run_playlist_job(jid, pid,
                            "https://youtube.com/playlist?list=S")
    vids = {v["youtube_id"]: v for v in db.get_videos(pid)}
    assert vids["ok1"]["has_transcript"]
    # the whole point: the rate-limited episode was RECOVERED by
    # the serial sweep, not silently skipped
    assert vids["rl1"]["has_transcript"]
    assert vids["rl1"]["transcript_error"] is None
    assert calls["rl1"] == 2
