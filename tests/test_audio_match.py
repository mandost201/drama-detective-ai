"""Audio fingerprint channel (2026-10-22): a Whisper-independent
search that matches the clip's SOUND against episode sound.

Calibration behind the bars (tools/audio_calibrate.py, real
data): an original-audio slice cut from an episode matched
itself at 0.95-0.96 with the offset recovered to ~0.01 s; wrong
episodes and a re-scored fan-edit clip peaked at 0.50-0.66.
These tests use synthetic tone signals (deterministic, offline):
the same relationships must hold - an original slice is found
at its true offset, an unrelated sound is not.
"""
import wave
from pathlib import Path

import numpy as np

from backend import analyze, audio_match, db

_GIBBERISH = ("zxqwv plokm ijnhb ygtrf vbnhj qwert asdfg "
              "lkjhg mnbvc xzsdf qqq www eee rrr")


def _signal(seed, seconds=90.0, sr=22050):
    """Distinctive sound: per-1.5 s blocks of tone pairs whose
    frequencies come from a seeded RNG, plus light noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = np.zeros(n, dtype=np.float32)
    t = np.arange(n) / sr
    block = int(1.5 * sr)
    for i in range(0, n, block):
        f1, f2 = rng.uniform(300, 3000, 2)
        seg = slice(i, min(n, i + block))
        x[seg] += (0.5 * np.sin(2 * np.pi * f1 * t[seg])
                   + 0.3 * np.sin(2 * np.pi * f2 * t[seg]))
    return x + 0.02 * rng.normal(size=n).astype(np.float32)


def _write_wav(path, x, sr=22050):
    pcm = np.clip(x * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return Path(path)


def test_matcher_recovers_exact_offset():
    rng = np.random.default_rng(0)
    ep = rng.normal(size=(5000, audio_match.BANDS)).astype(np.float32)
    clip = ep[1000:1060].copy()
    m = audio_match.best_match(ep, clip)
    assert abs(m["offset_s"] / audio_match.HOP_S - 1000) < 1.0
    assert m["score"] > 0.99


def test_matcher_on_real_signals(tmp_path):
    ep = _signal(7)
    clip = ep[int(30 * 22050):int(60 * 22050)]
    ep_f = audio_match.features_from_signal(
        audio_match.decode_mono(_write_wav(tmp_path / "ep.wav", ep)))
    clip_f = audio_match.features_from_signal(
        audio_match.decode_mono(_write_wav(tmp_path / "clip.wav", clip)))
    m = audio_match.best_match(ep_f, clip_f)
    assert abs(m["offset_s"] - 30.0) < 0.25
    assert m["score"] >= 0.90


def _mk_playlist(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "a.db")
    db.init_db()
    pid = db.create_playlist("https://youtube.com/playlist?list=A",
                             title="AudioTest")
    db.add_videos(pid, [
        {"youtube_id": "EP_A", "title": "Episode A", "position": 1,
         "duration": None, "url": "", "thumbnail": None},
        {"youtube_id": "EP_B", "title": "Episode B", "position": 2,
         "duration": None, "url": "", "thumbnail": None}])
    segs = [(float(s), float(s + 3), f"line number {s} here now")
            for s in range(0, 90, 10)]
    for v in db.get_videos(pid):
        db.add_segments(v["id"], segs)
        db.mark_transcript(v["id"], True, source="captions")
    return pid


def _gibberish_tf(p, task="transcribe"):
    return ([(0.0, 5.0, _GIBBERISH)], "en", 0.9)


def _run(pid, clip_path, audio_fn):
    return analyze.analyze_clip_full(
        pid, clip_path, transcribe_fn=_gibberish_tf,
        escalate_fn=_gibberish_tf, chunked_fn=lambda p: ([], []),
        audio_fn=audio_fn, visual_check=False)


def test_audio_search_finds_original_clip(tmp_path, monkeypatch):
    pid = _mk_playlist(tmp_path, monkeypatch)
    ep_a = _write_wav(tmp_path / "ep_a.wav", _signal(11))
    ep_b = _write_wav(tmp_path / "ep_b.wav", _signal(23))
    full = _signal(11)
    clip = _write_wav(tmp_path / "clip.wav",
                      full[int(30 * 22050):int(60 * 22050)])
    fn = lambda yid: ep_a if yid == "EP_A" else ep_b  # noqa: E731
    res = _run(pid, clip, fn)
    assert res["verdict"] in ("probable", "verified")
    top = res["results"][0]
    assert top["youtube_id"] == "EP_A"
    assert abs(top["match_start"] - 30.0) < 1.0
    assert top["method"].startswith("audio fingerprint")
    assert res["diagnostics"]["audio_search"]["checked"] == 2


def test_audio_search_rejects_unrelated_sound(tmp_path, monkeypatch):
    pid = _mk_playlist(tmp_path, monkeypatch)
    ep_a = _write_wav(tmp_path / "ep_a.wav", _signal(11))
    ep_b = _write_wav(tmp_path / "ep_b.wav", _signal(23))
    clip = _write_wav(tmp_path / "clip.wav", _signal(99, seconds=30.0))
    fn = lambda yid: ep_a if yid == "EP_A" else ep_b  # noqa: E731
    res = _run(pid, clip, fn)
    assert res["verdict"] == "not_verified"
    assert res["diagnostics"]["audio_search"]["best_score"] < \
        analyze.AUDIO_MATCH_MIN_SCORE


def test_audio_refine_corrects_dialogue_start(tmp_path, monkeypatch):
    pid = _mk_playlist(tmp_path, monkeypatch)
    full = _signal(11)
    ep_a = _write_wav(tmp_path / "ep_a.wav", full)
    clip = _write_wav(tmp_path / "clip.wav",
                      full[int(30 * 22050):int(60 * 22050)])
    v = db.get_videos(pid)[0]
    top = {"video_id": v["id"], "youtube_id": "EP_A",
           "title": v["title"], "episode": 1,
           "timestamp": 90.0, "timestamp_hms": "00:01:30",
           "match_start": 90.0, "match_end": 120.0,
           "match_start_hms": "00:01:30", "match_end_hms": "00:02:00",
           "scene_start": 80.0, "scene_end": 130.0,
           "scene_start_hms": "00:01:20", "scene_end_hms": "00:02:10",
           "block_start": 80.0, "block_end": 130.0,
           "block_start_hms": "00:01:20", "block_end_hms": "00:02:10",
           "url": "", "url_at": "", "thumbnail": None,
           "reasons": ["dialogue estimate"], "matched_lines": []}
    diag = {}
    analyze._audio_refine(top, clip, lambda yid: ep_a, diag)
    assert diag["audio_refine"]["applied"] is True
    assert abs(top["match_start"] - 30.0) < 1.0
