"""Ground-truth clip battery — REAL clips, REAL Whisper, production
pipeline. ONE database holding THREE playlists (exactly like the
live app), so cross-playlist isolation is exercised for real.

Positives (independently verified source episode + timestamp):
  Raja London Ka EP25  - clipwork_1021/clip.wav (user's TikTok edit;
      verified vs official subtitles: scene starts 828s).
  Ertugrul Ghazi EP76  - clipwork_1021/ert_ep76_dialogue.mp3 (audio
      section 120-210s of the source video, a dialogue/narration
      dense window of the bundled verified seed transcript).
  Nashtar EP21         - nashtar_work/clip.wav (user's fan edit;
      verified word-for-word vs official subtitles: boat dialogue
      from 647.6s). DB transcript: clipwork_1021/
      nashtar_ep21_segs.json (parsed from the saved subtitle page).

Rejections: every positive clip against the OTHER two playlists
must be not_verified.

Degraded case: clipwork_1021/ert_ep76_death.mp3 (665-755s section:
music + wailing, Whisper produces a repetition loop). It must be
rejected EVERYWHERE - including its own playlist - as
transcript_degenerate / not_verified, never promoted. (This clip
produced the 2026-10-21 false positive that motivated the gate.)

Run:  WHISPER_MODEL=base .venv/bin/python tools/ground_truth_clips.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WS = ROOT.parent

from backend import db, analyze  # noqa: E402
from backend.vtt import parse_vtt  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


def build_all():
    """One DB, three playlists - the production shape."""
    db.DB_PATH = Path(tempfile.mkdtemp()) / "gt.db"
    db.init_db()
    pids = {}

    def pl(key, title, vids):
        pid = db.create_playlist(f"https://youtube.com/playlist?list={key}",
                                 title=title)
        db.add_videos(pid, [{"youtube_id": y, "title": t, "position": p,
                             "duration": None, "url": "", "thumbnail": None}
                            for y, t, p, _s in vids])
        by = {v["youtube_id"]: v for v in db.get_videos(pid)}
        for y, _t, _p, segs in vids:
            db.add_segments(by[y]["id"], segs)
            db.mark_transcript(by[y]["id"], True, source="captions")
        pids[key] = pid

    pl("raja", "Raja London Ka", [
        (yid, f"Raja London Ka Episode {pos}", pos,
         parse_vtt((FIX / f).read_text()))
        for pos, yid, f in ((23, "raja23", "raja_ep23.en.vtt"),
                            (24, "raja24", "raja_ep24.en.vtt"),
                            (25, "YDu9akegEHU", "raja_ep25.en.vtt"))])
    seeds = json.loads((ROOT / "streamlit" / "seeds.json").read_text())
    pl("ert", "Ertugrul Ghazi", [
        ("qA4r7LDmnqY", "Ertugrul Ghazi Episode 76", 76,
         [(float(a), float(b), str(t)) for a, b, t in
          seeds["qA4r7LDmnqY"]])])
    raw = json.loads((WS / "clipwork_1021" /
                      "nashtar_ep21_segs.json").read_text())
    pl("nashtar", "Nashtar", [
        ("5vlp0YFzoRc", "Nashtar Episode 21", 21,
         [(float(a), float(b), str(t)) for a, b, t in raw])])
    return pids


POSITIVES = [
    ("Raja clip", "raja", WS / "clipwork_1021" / "clip.wav",
     "YDu9akegEHU", 828),
    ("Ertugrul dialogue clip", "ert",
     WS / "clipwork_1021" / "ert_ep76_dialogue.mp3",
     "qA4r7LDmnqY", 120),
    ("Nashtar clip", "nashtar", WS / "nashtar_work" / "clip.wav",
     "5vlp0YFzoRc", 648),
]
DEGRADED = ("Ertugrul death-section clip (music/wailing)",
            WS / "clipwork_1021" / "ert_ep76_death.mp3")


def main():
    pids = build_all()
    pos_ok = 0
    for label, key, clip, yid, start in POSITIVES:
        res = analyze.analyze_clip_full(pids[key], clip)
        tops = res.get("results") or []
        top = tops[0] if tops else None
        ok_ep = bool(top) and top["youtube_id"] == yid
        pos_ok += bool(ok_ep)
        got = top.get("match_start", top.get("scene_start")) if top else None
        print(f"[POS {label}] verdict={res.get('verdict')} "
              f"ep_ok={ok_ep} start={got} (truth ~{start}) "
              f"score={top['score'] if top else '-'}", flush=True)
    rej_ok = rej_total = 0
    for label, key, clip, _yid, _s in POSITIVES:
        for other in pids:
            if other == key:
                continue
            rej_total += 1
            res = analyze.analyze_clip_full(pids[other], clip)
            good = res.get("verdict") == "not_verified"
            rej_ok += good
            print(f"[REJ {label} vs {other}] "
                  f"verdict={res.get('verdict')} "
                  f"fk={res.get('failure_kind')} "
                  f"({'OK' if good else 'FALSE POSITIVE'})", flush=True)
    dlabel, dclip = DEGRADED
    deg_ok = deg_total = 0
    for key in pids:
        deg_total += 1
        res = analyze.analyze_clip_full(pids[key], dclip)
        good = res.get("verdict") == "not_verified"
        deg_ok += good
        print(f"[DEG {dlabel} vs {key}] verdict={res.get('verdict')} "
              f"fk={res.get('failure_kind')} "
              f"({'OK' if good else 'FALSE POSITIVE'})", flush=True)
    print(f"\nGROUND TRUTH: positives {pos_ok}/{len(POSITIVES)} | "
          f"rejections {rej_ok}/{rej_total} | degraded {deg_ok}/{deg_total}")


if __name__ == "__main__":
    main()
