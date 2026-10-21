"""Evidence-verdict tests for the clip finder (Command 2).

The rule under test: a clip result is a claim, and claims are
tiered by evidence. Clean extensive dialogue -> "verified".
Real but imperfect evidence (loose STT translation, an edited
clip) -> "probable", with the imperfection stated. Anything weaker
is not a match at all: no candidate is selected and the verdict is
"not verified" - a merely similar scene is never dressed up as an
answer.
"""
from pathlib import Path

from backend import analyze

# Modelled on the REAL Whisper-base output for the ground-truth
# clip (observed 2026-10-21): the Urdu speech comes back garbled -
# names mangled, sentences reordered - yet the episode is right.
NOISY_ORIG = [(0.0, 22.0,
               "Waki Sahe Dhula Dhula I came to you. I was saying "
               "you can finish the task and then you can go to the "
               "rest of the work. Waki Sahe Dhula is here.")]
NOISY_EN = [(0.0, 22.0,
             "Mr Vakil the bride is here. I was telling you. She "
             "is going to do the rest of the rest of the work. Mr "
             "Vakil the bride is here. Raniya I can't marry him.")]

# An EDITED clip: two chunks of the EP25 lawyer scene spliced
# together with the middle cut out and foreign filler inserted -
# exactly what fan edits do.
EDITED_EN = [(0.0, 26.0,
              "Lawyer the groom and bridegroom are here I was "
              "saying get them signed on the papers meanwhile "
              "somewhere across the city a train was leaving the "
              "station Rania Mirza this marriage cannot take "
              "place what are you saying")]


def _tf(original, translated=None, lang="ur", prob=0.9):
    def tf(path, task="transcribe"):
        if task == "translate":
            return (translated or []), lang, prob
        return original, lang, prob
    return tf


def _no_visual(monkeypatch):
    monkeypatch.setattr(analyze, "_visual_verify",
                        lambda *a, **k: {"status": "unavailable",
                                         "best_distance": None,
                                         "detail": "test stub"})


def test_verdict_tiers_are_evidence_bound():
    v = analyze._alignment_verdict
    assert v(0.9, 30) == "verified"
    assert v(0.55, 10) == "verified"
    assert v(0.54, 30) == "probable"   # score just under the bar
    assert v(0.9, 9) == "probable"     # too few matched words
    assert v(0.337, 31) == "probable"  # the real clip's evidence
    assert v(0.28, 8) == "probable"
    assert v(0.8, 6) == "probable"     # short but near-perfect phrase
    assert v(0.74, 6) is None
    assert v(0.9, 4) is None           # too little material at all
    assert v(0.27, 30) is None
    assert v(0.9, 7) == "probable"     # short distinctive phrase
    assert v(0.23, 12) is None         # the old fuzzy_align floor


def test_noisy_real_world_clip_is_probable_not_verified(built_db,
                                                        monkeypatch):
    _no_visual(monkeypatch)
    res = analyze.analyze_clip_full(
        built_db["raja"], Path("clip.wav"),
        transcribe_fn=_tf(NOISY_ORIG, NOISY_EN))
    assert res["ok"] is True
    assert res["results"], res
    top = res["results"][0]
    assert top["youtube_id"] == "YDu9akegEHU"
    assert abs(top["timestamp"] - 828) <= 45, top["timestamp_hms"]
    # the evidence is real but imperfect: probable, NEVER verified
    assert res["verdict"] == "probable"
    assert top["matched_lines"], "timestamped evidence must be kept"


def test_edited_clip_matches_dominant_segment_with_warning(built_db,
                                                           monkeypatch):
    _no_visual(monkeypatch)
    res = analyze.analyze_clip_full(
        built_db["raja"], Path("edit.mp4"),
        transcribe_fn=_tf(EDITED_EN, lang="en"))
    assert res["results"], res
    top = res["results"][0]
    assert top["youtube_id"] == "YDu9akegEHU"
    assert abs(top["timestamp"] - 828) <= 30, top["timestamp_hms"]
    assert res["verdict"] in ("probable", "verified")
    reasons = " ".join(top["reasons"])
    assert "edited" in reasons or "hissa" in reasons, reasons


def test_short_generic_clip_is_not_a_match(built_db):
    """Six generic words that surely occur in some episode are NOT
    evidence: too few matched words, whatever the local similarity."""
    tiny = [(0.0, 3.0, "please come here right now okay")]
    res = analyze.analyze_clip_full(
        built_db["raja"], Path("tiny.wav"),
        transcribe_fn=_tf(tiny, lang="en"))
    assert res["results"] == []
    assert res["verdict"] == "not_verified"
    assert res["note"] == "No verified match in this playlist."


def test_wrong_playlist_clip_is_not_verified(built_db):
    """The EP25 clip searched inside the ERTUGRUL playlist must not
    produce a 'match' from another drama's transcripts."""
    from tests.test_universal_clip import CLIP_EN_SEGS, CLIP_UR_SEGS
    res = analyze.analyze_clip_full(
        built_db["ert"], Path("clip.wav"),
        transcribe_fn=_tf(CLIP_UR_SEGS, CLIP_EN_SEGS))
    assert res["results"] == []
    assert res["verdict"] == "not_verified"
