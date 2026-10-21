"""End-to-end tests of the matching engine on real subtitle data.

Ground truth (verified by hand against the episodes' official English
subtitles on 2026-10-21):
- Raja London Ka EP25, lawyer scene: starts 00:13:48; "The groom and
  bridegroom are here" 00:13:55; "Rania Mirza / This marriage cannot
  take place" 00:14:25-00:14:34; fees + tea money 00:14:51-00:14:55;
  "No judge will come here either" 00:15:16.
- Ertugrul EP76: Suleiman Shah death anchor at 700s (00:11:40) in the
  bundled verified transcript.
"""
from backend import search

EP25_YID = "YDu9akegEHU"
ERT_YID = "qA4r7LDmnqY"


def test_english_dialogue_finds_ep25(built_db):
    res = search.search(
        built_db["raja"],
        "Rania Mirza, this marriage cannot take place. Lawyer, no judge "
        "will come here. This is the money for your fees and your tea.")
    assert res["results"], "expected at least one result"
    top = res["results"][0]
    assert top["youtube_id"] == EP25_YID
    assert top["episode"] == 25
    assert 800 <= top["timestamp"] <= 940, top["timestamp_hms"]
    assert top["confidence"] in ("high", "medium")
    # Exact Clip Finder: the watch link opens at the estimated scene
    # start, and the anchor timestamp sits inside the scene range.
    assert top["url_at"].endswith(f"t={int(top['scene_start'])}s")
    assert top["scene_start"] <= top["timestamp"] <= top["scene_end"]


def test_roman_urdu_description_finds_ep25(built_db):
    # The same scene, described the way the user actually writes.
    res = search.search(
        built_db["raja"],
        "vakil sahab dulha dulhan aa gaye hain, Rania Mirza ye shaadi "
        "nahi ho sakti, koi judge bhi nahi aayega, fees aur chai ke "
        "paise diye jate hain")
    assert res["results"], "expected at least one result"
    top = res["results"][0]
    assert top["youtube_id"] == EP25_YID
    assert 800 <= top["timestamp"] <= 940, top["timestamp_hms"]
    assert top["confidence"] in ("high", "medium")


def test_generic_query_never_high_confidence(built_db):
    # Historical false positive: a generic death query with a name that
    # appears nowhere must never come back as a confident match.
    res = search.search(built_db["raja"], "Shahbudin dies scene")
    if res["results"]:
        assert res["results"][0]["confidence"] == "low"


def test_suleman_shah_death_finds_ep76_anchor(built_db):
    res = search.search(built_db["ert"], "Suleman shah ki death")
    assert res["results"], "expected the seeded episode to match"
    top = res["results"][0]
    assert top["youtube_id"] == ERT_YID
    assert abs(top["timestamp"] - 700) <= 120, top["timestamp_hms"]
    assert top["confidence"] == "high"


def test_wrong_name_on_ertugrul_never_high(built_db):
    res = search.search(built_db["ert"], "Shahbudin dies seen")
    if res["results"]:
        assert res["results"][0]["confidence"] != "high"
