"""Script-query regression (final audit, 2026-10-21).

Measured before the fix: an Urdu-script name query ("رانیہ مرزا")
and a Devanagari one returned NO results against English
subtitles - names have no concept-map entries and there was no
transliteration path. Latin/Roman-Urdu queries were unaffected.
The fix: mechanical transliteration to Latin candidates, retried
only when the script query itself finds nothing.
"""
from backend import normalize, search


def test_latin_candidates_only_for_script_queries():
    assert normalize.latin_candidates("Rania Mirza") == []
    ur = normalize.latin_candidates("رانیہ مرزا")
    assert ur and "rania" in ur[0], ur
    assert any("mirza" in c for c in ur), ur
    hi = normalize.latin_candidates("रानिया मिर्ज़ा")
    assert hi and "rania" in hi[0] and "mirza" in hi[0], hi


def test_urdu_script_name_query_finds_the_episode(built_db):
    res = search.search(built_db["raja"], "رانیہ مرزا")
    assert res["results"], res
    assert res["results"][0]["episode"] == 25
    assert res.get("effective_query")


def test_devanagari_query_finds_the_episode(built_db):
    res = search.search(built_db["raja"], "रानिया मिर्ज़ा शादी")
    assert res["results"], res
    assert res["results"][0]["episode"] == 25


def test_latin_query_path_unchanged(built_db):
    res = search.search(built_db["raja"], "Rania Mirza")
    assert res["results"][0]["episode"] == 25
    assert "effective_query" not in res  # no retry happened


def test_urdu_query_with_semantic_enabled_still_uses_transliteration(
        built_db, monkeypatch):
    """Production repro (found live, 2026-10-21): with semantic
    search ENABLED, the Urdu name query returns a weak semantic
    hit (no distinctive-term coverage) instead of nothing - the
    transliteration retry must fire on that evidence, not only
    on an empty page, and must surface the real dialogue match."""
    import numpy as np

    from backend import embeddings
    monkeypatch.setattr(embeddings, "available", lambda: True)

    def fake_embed(texts):
        # every text maps to the same unit vector: semantic search
        # then returns hits for any query, as it did live
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0]),
                       (len(texts), 1)).astype("float32")

    monkeypatch.setattr(embeddings, "embed", fake_embed)
    res = search.search(built_db["raja"], "رانیہ مرزا")
    assert res["results"], res
    top = res["results"][0]
    assert top["episode"] == 25
    assert top["matched_at"] == 865.0  # the real Rania Mirza line
    assert res.get("effective_query")  # transliteration was used
