"""Exact Clip Finder tests: scene ranges, playlist coverage,
confidence caps, missing captions, duplicates, boundaries.

Synthetic playlists are built next to the real-data ones from
conftest.py; each concern gets its own playlist so tests stay
independent. Semantic embeddings are disabled (see conftest), so the
keyword path + guards are what is verified.
"""
from pathlib import Path

import pytest

from backend import db, search
from backend.vtt import parse_vtt

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EP25_QUERY = ("Rania Mirza, this marriage cannot take place. Lawyer, "
              "no judge will come here. This is the money for your "
              "fees and your tea.")


def _add_video(pid, yid, pos, segs=None):
    db.add_videos(pid, [{
        "youtube_id": yid, "title": f"Synthetic Episode {pos}",
        "position": pos, "duration": None,
        "url": f"https://www.youtube.com/watch?v={yid}",
        "thumbnail": None}])
    v = [x for x in db.get_videos(pid) if x["youtube_id"] == yid][0]
    if segs is not None:
        db.add_segments(v["id"], segs)
        db.mark_transcript(v["id"], True)
    return v


@pytest.fixture(scope="module")
def synth(built_db):
    ep25 = parse_vtt((FIXTURES / "raja_ep25.en.vtt").read_text("utf-8"))
    ep24 = parse_vtt((FIXTURES / "raja_ep24.en.vtt").read_text("utf-8"))
    smalltalk = [(0.0, 2.0, "Hello there, how are you today?"),
                 (3.0, 5.0, "I am fine, thank you very much."),
                 (6.0, 8.0, "Let us go to the market now.")]

    # A: only 1 of 4 episodes searchable -> 25% coverage (cap test)
    pa = db.create_playlist("https://youtube.com/playlist?list=CAPTEST",
                            title="Cap Drama")
    _add_video(pa, "SYNTHA00001", 1, ep25)
    _add_video(pa, "SYNTHA00002", 2)
    _add_video(pa, "SYNTHA00003", 3)
    _add_video(pa, "SYNTHA00004", 4)

    # B: 2 of 3 searchable; one episode has no captions at all
    pb = db.create_playlist("https://youtube.com/playlist?list=MISSTEST",
                            title="Missing Drama")
    _add_video(pb, "SYNTHB00001", 1, smalltalk)
    _add_video(pb, "SYNTHB00002", 2, ep24)
    b3 = _add_video(pb, "SYNTHB00003", 3)

    # C: hand-built segment blocks for scene-range boundary tests
    pc = db.create_playlist("https://youtube.com/playlist?list=RANGE",
                            title="Range Drama")
    blocks = [(0.0, 2.0, "alpha one"), (2.5, 4.5, "alpha two"),
              (5.0, 7.0, "alpha three"),
              (40.0, 42.0, "beta one keyword"), (42.5, 44.5, "beta two"),
              (45.0, 47.0, "beta three keyword"),
              (80.0, 82.0, "gamma one"), (82.5, 84.0, "gamma two")]
    c1 = _add_video(pc, "SYNTHC00001", 1, blocks)
    dense = [(float(i * 5), float(i * 5) + 4.0, "filler line here")
             for i in range(121)]  # 0..604s of unbroken dialogue
    c2 = _add_video(pc, "SYNTHC00002", 2, dense)

    return {"cap": pa, "missing": pb, "range": pc,
            "b3": b3, "c1": c1, "c2": c2}


# --- scene ranges -----------------------------------------------------

def test_scene_spans_multiple_dialogue_segments(built_db):
    res = search.search(built_db["raja"], EP25_QUERY)
    top = res["results"][0]
    assert top["youtube_id"] == "YDu9akegEHU"
    # The described scene spans 00:13:48 -> ~00:15:20 (many segments);
    # the range must cover the whole exchange, not one line.
    assert top["scene_start"] <= 836, top["scene_start_hms"]
    assert top["scene_end"] >= 916, top["scene_end_hms"]
    assert top["scene_start"] >= 720, top["scene_start_hms"]
    assert top["scene_end"] - top["scene_start"] <= 300
    assert top["scene_start"] <= top["timestamp"] <= top["scene_end"]
    # the watch link now opens at the scene start
    assert top["url_at"].endswith(f"t={int(top['scene_start'])}s")
    assert top["match_type"] == "transcript"


def test_roman_urdu_variant_scene_range(built_db):
    res = search.search(
        built_db["raja"],
        "vakil sahab dulha dulhan aa gaye hain, Rania Mirza ye shaadi "
        "nahi ho sakti, koi judge bhi nahi aayega, fees aur chai ke "
        "paise diye jate hain")
    top = res["results"][0]
    assert top["youtube_id"] == "YDu9akegEHU"
    assert top["scene_start"] <= 850, top["scene_start_hms"]
    assert top["scene_end"] >= 905, top["scene_end_hms"]
    assert top["scene_start"] <= top["timestamp"] <= top["scene_end"]


def test_scene_range_block_boundaries(synth):
    c1 = synth["c1"]["id"]
    # anchor inside block B (40..47), big silence on both sides
    s, e = search.scene_range(c1, 42.5, ["keyword"])
    assert (s, e) == (40.0, 47.0), (s, e)
    # no terms -> pure dialogue-block bounds, same answer here
    s, e = search.scene_range(c1, 42.5, [])
    assert (s, e) == (40.0, 47.0), (s, e)
    # anchor in block A: term trim finds no 'keyword' in A -> block A
    s, e = search.scene_range(c1, 2.5, ["keyword"])
    assert (s, e) == (0.0, 7.0), (s, e)


def test_scene_range_span_is_capped(synth):
    s, e = search.scene_range(synth["c2"]["id"], 300.0, [])
    assert e - s <= 300.0, (s, e)
    assert s <= 300.0 <= e


# --- coverage ----------------------------------------------------------

def test_full_coverage_reported(built_db):
    res = search.search(built_db["raja"], EP25_QUERY)
    assert res["coverage"] == {"total": 3, "searchable": 3,
                               "missing": 0, "pct": 100.0}


def test_missing_captions_counted_and_excluded(synth):
    cov = search.playlist_coverage(synth["missing"])
    assert cov == {"total": 3, "searchable": 2, "missing": 1,
                   "pct": 66.7}
    # the un-captioned episode is reported as unprocessed...
    vids = {v["youtube_id"]: v for v in db.get_videos(synth["missing"])}
    assert not vids["SYNTHB00003"]["has_transcript"]
    # ...and can never appear as a search result (nothing indexed)
    res = search.search(synth["missing"], EP25_QUERY)
    assert "SYNTHB00003" not in [r["youtube_id"] for r in
                                 res["results"] + res["alternatives"]]
    assert res["coverage"]["missing"] == 1


def test_partial_coverage_caps_confidence(synth):
    # Exact dialogue, but only 1 of 4 episodes is searchable: the
    # scene might be in an unchecked episode, so "high" is not honest.
    res = search.search(synth["cap"], EP25_QUERY)
    assert res["coverage"]["pct"] == 25.0
    top = res["results"][0]
    assert top["youtube_id"] == "SYNTHA00001"
    assert top["confidence"] == "medium"
    assert any("capped" in r for r in top["reasons"])


# --- honesty guards ----------------------------------------------------

def test_incorrect_character_name_never_high(built_db):
    res = search.search(
        built_db["raja"],
        "Hamza Mirza says this marriage cannot take place and no "
        "judge will come here")
    if res["results"]:
        assert res["results"][0]["confidence"] != "high"


def test_duplicate_episode_not_double_counted():
    pid = db.create_playlist("https://youtube.com/playlist?list=DUPTEST",
                             title="Duplicate Drama")
    _add_video(pid, "SYNTHD00001", 1, [(0.0, 2.0, "only copy")])
    # adding the same episode again (re-import) must not duplicate it
    db.add_videos(pid, [{
        "youtube_id": "SYNTHD00001", "title": "Synthetic Episode 1",
        "position": 1, "duration": None,
        "url": "https://www.youtube.com/watch?v=SYNTHD00001",
        "thumbnail": None}])
    ids = [v["youtube_id"] for v in db.get_videos(pid)]
    assert ids.count("SYNTHD00001") == 1
    assert search.playlist_coverage(pid)["total"] == 1
