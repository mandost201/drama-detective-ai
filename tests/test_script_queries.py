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
