"""Unit tests: normalization, URL parsing, history & feedback storage."""
from backend import db
from backend import normalize as nz
from backend.youtube import extract_playlist_id, extract_video_id


def test_detect_script():
    assert nz.detect_script("ye shaadi nahi ho sakti") == "latin"
    assert nz.detect_script("یہ شادی نہیں ہو سکتی") == "ur"
    assert nz.detect_script("यह शादी नहीं हो सकती") == "hi"


def test_canonical_variants():
    assert nz.canonical("shadi") == "shaadi"
    assert nz.canonical("wakeel") == "vakil"
    assert nz.canonical("nhi") == "nahi"
    assert nz.canonical("suleiman") == "suleman"


def test_concept_expansions():
    assert "marriage" in nz.expansions_for("shaadi")
    assert "groom" in nz.expansions_for("dulha")
    assert "lawyer" in nz.expansions_for("vakil")
    assert "lawyer" in nz.expansions_for("وکیل")
    assert "marriage" in nz.expansions_for("शादी")
    assert "suleiman" in nz.variants_for("suleman")


def test_expand_query_keeps_user_words_first():
    q = nz.expand_query("vakil sahab dulha aa gaya")
    assert q.startswith("vakil sahab dulha")
    assert "lawyer" in q and "groom" in q


def test_term_matches_cross_language():
    line = "This marriage cannot take place."
    assert nz.term_matches("shaadi", line)
    assert nz.term_matches("suleman", "Then Suleiman Shah spoke.")
    assert not nz.term_matches("shahbudin", "The groom and bride are here.")
    assert not nz.term_matches("vakil", "Nothing relevant in this line.")


def test_url_parsing():
    assert extract_playlist_id(
        "https://youtube.com/playlist?list=PLeb83ChrfOzns7wEnZvKrKrR3a-c_bK1x"
    ) == "PLeb83ChrfOzns7wEnZvKrKrR3a-c_bK1x"
    assert extract_video_id("https://www.youtube.com/watch?v=YDu9akegEHU") == "YDu9akegEHU"
    assert extract_video_id("https://youtu.be/YDu9akegEHU") == "YDu9akegEHU"
    assert extract_playlist_id("https://example.com/nope") is None
    assert extract_video_id("https://example.com/nope") is None


def test_history_and_feedback_roundtrip(built_db):
    pid = built_db["raja"]
    vids = db.get_videos(pid)
    db.log_search(pid, "test query", vids[2]["id"], 835.0, "high")
    recent = db.recent_searches(pid, limit=5)
    assert recent and recent[0]["query"] == "test query"
    assert recent[0]["confidence"] == "high"

    db.log_feedback(pid, "test query", vids[2]["id"], "wrong",
                    correct_video_id=vids[1]["id"], note="ep24 hai")
    fb = db.list_feedback(pid)
    assert fb and fb[0]["verdict"] == "wrong"
    assert fb[0]["correct_video_id"] == vids[1]["id"]

    pl = db.get_playlist(pid)
    assert pl["drama_name"] == "Raja London Ka"
