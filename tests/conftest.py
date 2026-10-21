"""Test fixtures: a throwaway SQLite DB filled with REAL subtitle data.

- Raja London Ka episodes 23/24/25: official English subtitles, fetched
  from the public YouTube caption tracks (tests/fixtures/*.vtt).
- Dirilis Ertugrul episode 76: the human-verified transcript bundled
  with the app (streamlit/seeds.json).

Semantic embeddings are disabled in tests (deterministic, offline):
the keyword path + coverage guard are what these tests verify. On the
live app, semantic search adds a second signal on top.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import db, embeddings  # noqa: E402
from backend.vtt import parse_vtt  # noqa: E402

embeddings.available = lambda: False  # noqa: E731 - deterministic tests

FIXTURES = Path(__file__).resolve().parent / "fixtures"

RAJA_EPS = [
    (23, "zW4vWJjSrpM", "Raja London Ka Episode 23 [ENG SUB]", "raja_ep23.en.vtt"),
    (24, "nwQZtlocHBo", "Raja London Ka Episode 24 [ENG SUB]", "raja_ep24.en.vtt"),
    (25, "YDu9akegEHU", "Raja London Ka Episode 25 [ENG SUB]", "raja_ep25.en.vtt"),
]
ERT_YID = "qA4r7LDmnqY"  # Ertugrul Ghazi Episode 76


@pytest.fixture(scope="session")
def built_db(tmp_path_factory):
    db.DB_PATH = tmp_path_factory.mktemp("ddb") / "test.db"
    db.init_db()

    # --- Raja London Ka (3 real episodes) ---
    pid_raja = db.create_playlist(
        "https://youtube.com/playlist?list=PLeb83ChrfOzns7wEnZvKrKrR3a-c_bK1x",
        title="Raja London Ka | Full Episodes")
    db.update_playlist(pid_raja, drama_name="Raja London Ka")
    db.add_videos(pid_raja, [
        {"youtube_id": yid, "title": title, "position": pos,
         "duration": None, "url": f"https://www.youtube.com/watch?v={yid}",
         "thumbnail": None}
        for pos, yid, title, _f in RAJA_EPS])
    by_yid = {v["youtube_id"]: v for v in db.get_videos(pid_raja)}
    for _pos, yid, _title, fname in RAJA_EPS:
        segs = parse_vtt((FIXTURES / fname).read_text(encoding="utf-8"))
        assert len(segs) > 100, f"fixture {fname} parsed too few segments"
        db.add_segments(by_yid[yid]["id"], segs)
        db.mark_transcript(by_yid[yid]["id"], True)
    db.update_playlist(pid_raja, status="ready", video_count=len(RAJA_EPS))

    # --- Ertugrul Ep76 (bundled verified seed transcript) ---
    pid_ert = db.create_playlist("https://youtube.com/playlist?list=ERT",
                                 title="Ertugrul Ghazi Season 1 Urdu")
    db.update_playlist(pid_ert, drama_name="Ertugrul Ghazi")
    db.add_videos(pid_ert, [
        {"youtube_id": ERT_YID, "title": "Ertugrul Ghazi Episode 76",
         "position": 76, "duration": None,
         "url": f"https://www.youtube.com/watch?v={ERT_YID}",
         "thumbnail": None}])
    v = db.get_videos(pid_ert)[0]
    seeds = json.loads((ROOT / "streamlit" / "seeds.json").read_text())
    segs = [(float(s[0]), float(s[1]), str(s[2])) for s in seeds[ERT_YID]]
    db.add_segments(v["id"], segs)
    db.mark_transcript(v["id"], True)
    db.update_playlist(pid_ert, status="ready", video_count=1)

    return {"raja": pid_raja, "ert": pid_ert}
