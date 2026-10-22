"""Wehshi regression battery (2026-10-21).

Ground truth - established INDEPENDENTLY of the clip pipeline,
from the episode's own published captions: the user's 59.6s clip
is the hospital confrontation in Wehshi's LAST EPISODE
(video y0q5xYZ8jZ8). At ~00:33:55 (2035s) the episode's official
auto-captions read "आज तुमने अपनी जिंदगी में उसे एक शख्स को भी
खो दिया जिसने सिर्फ तुम्हें चाहा / सिर्फ तुम्हें चाहा बचपन से /
शायद यही सजा थी तुम्हारी" - the same lines the clip's audio
carries ("...you have lost the one person who only loved you
since childhood... maybe this is the punishment for you").

The case that exposed the bug: episodes carry ONLY Hindi
(Devanagari) auto-captions; the site's base-model clip hearing
came out in Urdu script at 57% confidence with a repetition chunk,
and whole-text token alignment found nothing in any script space.

Cases:
  1. Wehshi clip  vs Wehshi (36 eps)  -> probable/verified,
     Last Episode, refined clip start ~1986-2010s (the clip's
     ~30s point carries the line the episode captions stamp at
     2016.4s), scene containing the verified lines
  2. Raja clip    vs Wehshi           -> not_verified
  3. Ertugrul clip vs Wehshi          -> not_verified
  4. Wehshi clip  vs Raja EP25        -> not_verified

Run: .venv/bin/python tools/wehshi_regression.py
Captions are cached in ~/workspace/clipwork_1021/wehshi_subs
(downloaded once with yt-dlp --write-auto-subs hi-orig)."""
import subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CW = Path.home() / "workspace/clipwork_1021"
SUBS = CW / "wehshi_subs"
PLAYLIST = "https://youtube.com/playlist?list=" \
           "PLbVdwtmx18svLwGeRuOqc5lTKSnhzGOV9"
WEHSHI_CLIP = Path.home() / "workspace/user/media_library/video/40" \
    "/4093ee6f07cc00392f7a4282a99577eb7eb51957ca87dbe0a548ccbd0d952b57.mp4"
TRUTH_VID, TRUTH_T = "y0q5xYZ8jZ8", 2035


def _titles():
    out = subprocess.run(
        [str(Path.home() / ".clipvenv/bin/yt-dlp"), "--flat-playlist",
         "--print", "%(id)s | %(title)s",
         "--extractor-args", "youtube:player_client=android",
         PLAYLIST], capture_output=True, text=True).stdout
    return {l.split(" | ", 1)[0].strip(): l.split(" | ", 1)[1].strip()
            for l in out.splitlines() if " | " in l}


def main():
    from backend import analyze, db
    from backend.vtt import parse_vtt
    db.DB_PATH = Path(tempfile.mkdtemp()) / "w.db"
    db.init_db()
    titles = _titles()
    pid = db.create_playlist(PLAYLIST, title="Wehshi")
    files = sorted(SUBS.glob("*.vtt"))
    assert len(files) == 36, f"expected 36 caption files, got {len(files)}"
    db.add_videos(pid, [{"youtube_id": f.name.split(".")[0],
                         "title": titles.get(f.name.split(".")[0], "?"),
                         "position": i + 1, "duration": None,
                         "url": "", "thumbnail": None}
                        for i, f in enumerate(files)])
    for v, f in zip(db.get_videos(pid), files):
        segs = parse_vtt(f.read_text(encoding="utf-8", errors="ignore"))
        if segs:
            db.add_segments(v["id"], segs)
            db.mark_transcript(v["id"], True, source="captions")
    _nv = len(db.get_videos(pid))
    _ns = sum(len(db.get_segments(v["id"])) for v in db.get_videos(pid))
    print(f"Wehshi DB: {_nv} videos, {_ns} segments", flush=True)
    wehshi_db = db.DB_PATH

    # Cross check: build the 3-playlist ground-truth DB (its own
    # file) and take its Raja playlist.
    sys.path.insert(0, str(ROOT / "tools"))
    import ground_truth_clips as gt
    raja = gt.build_all()["raja"]
    gt_db = db.DB_PATH
    db.DB_PATH = wehshi_db  # cases 1-3 run against the Wehshi DB
    print(f"Raja DB ready: pid={raja}", flush=True)

    def tf_base(p, task="transcribe"):
        return analyze._whisper_transcribe(p, task=task,
                                           model_name="base")

    def tf_small(p, task="transcribe"):
        return analyze._whisper_transcribe(p, task=task,
                                           model_name="small")

    def run(pid_, clip):
        return analyze.analyze_clip_full(
            pid_, clip, transcribe_fn=tf_base, escalate_fn=tf_small,
            visual_check=False)

    ok = True

    r = run(pid, WEHSHI_CLIP)
    top = r["results"][0] if r["results"] else {}
    good = (r["verdict"] in ("probable", "verified")
            and top.get("youtube_id") == TRUTH_VID
            and 1950 <= top.get("match_start", -1) <= 2030
            and top.get("scene_start", 99999) <= 2016
            and top.get("scene_end", -1) >= 2016)
    ok &= good
    print(f"CASE1 wehshi->wehshi: {r['verdict']} "
          f"vid={top.get('youtube_id')} match_start={top.get('match_start')} "
          f"scene={top.get('scene_start')}-{top.get('scene_end')} "
          f"method={top.get('method')} -> {'PASS' if good else 'FAIL'}",
          flush=True)

    for name, clip in (("raja", CW / "clip.wav"),
                       ("ertugrul", CW / "ert_ep76_dialogue.mp3")):
        r = run(pid, clip)
        good = r["verdict"] == "not_verified"
        ok &= good
        print(f"CASE {name}->wehshi: {r['verdict']} "
              f"({r['failure_kind']}) -> {'PASS' if good else 'FAIL'}",
              flush=True)

    # CASE audio (2026-10-22): the sound channel. (a) An
    # ORIGINAL-audio slice cut from ep36 at the true scene must
    # be found by sound alone - at ~1986 s, within seconds -
    # even when the transcript is the live-style English
    # confabulation. (b) The real fan-edit clip (whose sound an
    # editor replaced) must NOT be force-matched by sound: the
    # search runs, reports its best score, and stays honest.
    ep_audio_dir = CW / "ep_audio"
    ep36_file = ep_audio_dir / "y0q5xYZ8jZ8.mp4"
    if ep36_file.exists():
        import wave as _wave
        import numpy as _np
        from backend import audio_match as _am
        slice_path = CW / "wehshi_ep36_slice_1986.wav"
        if not slice_path.exists():
            sig = _am.decode_mono(ep36_file)
            seg = sig[int(1986 * _am.SR):int(2046 * _am.SR)]
            pcm = _np.clip(seg * 32767, -32768, 32767) \
                .astype(_np.int16)
            with _wave.open(str(slice_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(_am.SR)
                w.writeframes(pcm.tobytes())

        def _audio_prov(yid):
            p = ep_audio_dir / f"{yid}.mp4"
            return p if p.exists() else None

        _CONFAB = ("Pihu, you talk about yourself, okay? Now, we "
                   "will be together, no one can stop us. We will "
                   "start with a new story. This will be fun, "
                   "okay? No, Asim. You have made the biggest "
                   "mistake of your life. That girl used to go "
                   "away with you and you took her away.")

        def _tf_confab(p, task="transcribe"):
            return ([(0.0, 5.0, _CONFAB)], "en", 0.9)

        def _run_audio(clip):
            return analyze.analyze_clip_full(
                pid, clip, transcribe_fn=_tf_confab,
                escalate_fn=_tf_confab,
                chunked_fn=lambda p: ([], []),
                audio_fn=_audio_prov, visual_check=False)

        r = _run_audio(slice_path)
        top = r["results"][0] if r["results"] else {}
        good = (r["verdict"] in ("probable", "verified")
                and top.get("youtube_id") == TRUTH_VID
                and abs(top.get("match_start", -9999) - 1986) <= 3)
        ok &= good
        print(f"CASE audio-slice->wehshi: {r['verdict']} "
              f"vid={top.get('youtube_id')} "
              f"match_start={top.get('match_start')} "
              f"score={top.get('score')} "
              f"-> {'PASS' if good else 'FAIL'}", flush=True)

        r = _run_audio(WEHSHI_CLIP)
        _as = r["diagnostics"].get("audio_search") or {}
        good = (r["verdict"] == "not_verified"
                and _as.get("checked", 0) >= 1
                and (_as.get("best_score") or 0) < 0.80)
        ok &= good
        print(f"CASE audio-fanedit->wehshi: {r['verdict']} "
              f"audio_checked={_as.get('checked')} "
              f"best_score={_as.get('best_score')} "
              f"-> {'PASS' if good else 'FAIL'}", flush=True)
    else:
        print("CASE audio: SKIP (ep36 audio not downloaded)",
              flush=True)

    db.DB_PATH = gt_db  # cross check runs against the GT DB
    r = run(raja, WEHSHI_CLIP)
    good = r["verdict"] == "not_verified"
    ok &= good
    print(f"CASE wehshi->raja: {r['verdict']} ({r['failure_kind']}) "
          f"-> {'PASS' if good else 'FAIL'}", flush=True)

    print("WEHSHI BATTERY:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
