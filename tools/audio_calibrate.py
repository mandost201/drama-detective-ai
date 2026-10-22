"""Audio fingerprint calibration (2026-10-22).

Matches the real Wehshi clip against downloaded episode audios:
the true episode (y0q5xYZ8jZ8, truth: clip starts ~1986s) plus
same-drama wrong episodes and one foreign-drama episode (Raja 25).
Also a self-consistency check: a slice cut from ep36 at 700s must
be found at ~700s. Prints offset/score/z/second-peak + timings so
verdict thresholds come from measurements, not wishes.
"""
import sys
import time
from pathlib import Path

ROOT = Path.home() / "workspace" / "drama-detective-ai"
sys.path.insert(0, str(ROOT))
import av  # noqa: E402  (av.open patch parity with backend use)
from backend import audio_match as am  # noqa: E402

CW = Path.home() / "workspace" / "clipwork_1021"
AUD = CW / "ep_audio"
CLIP = (Path.home() / "workspace" / "user" / "media_library" / "video"
        / "40" / "4093ee6f07cc00392f7a4282a99577eb7eb51957"
        "ca87dbe0a548ccbd0d952b57.mp4")
FEATS = AUD / "feats"


def feats_for(path, key):
    t = time.time()
    f = am.audio_features(path, cache_dir=FEATS, cache_key=key)
    return f, time.time() - t


def main():
    clip_feats, dt = feats_for(CLIP, "wehshi_clip")
    print(f"clip feats {clip_feats.shape} in {dt:.1f}s", flush=True)

    targets = sorted(AUD.glob("*.mp4"))
    for p in targets:
        vid = p.stem
        ef, dt = feats_for(p, vid)
        t = time.time()
        m = am.best_match(ef, clip_feats)
        dtm = time.time() - t
        print(f"{vid}: offset={m['offset_s']:.2f}s score={m['score']:.3f} "
              f"z={m['z']:.1f} second={m['second_score']:.3f} "
              f"(feats {dt:.1f}s, match {dtm:.2f}s)", flush=True)

    # self-consistency: slice ep36 [700, 760] -> must land at ~700
    sig = am.decode_mono(AUD / "y0q5xYZ8jZ8.mp4")
    sl = sig[int(700 * am.SR):int(760 * am.SR)]
    sf = am.features_from_signal(sl)
    ef, _ = feats_for(AUD / "y0q5xYZ8jZ8.mp4", "y0q5xYZ8jZ8")
    m = am.best_match(ef, sf)
    print(f"SELF slice@700: offset={m['offset_s']:.2f}s "
          f"score={m['score']:.3f} z={m['z']:.1f}", flush=True)
    # and a slice at the TRUE scene [1986, 2046]
    sl2 = sig[int(1986 * am.SR):int(2046 * am.SR)]
    m2 = am.best_match(ef, am.features_from_signal(sl2))
    print(f"SELF slice@1986: offset={m2['offset_s']:.2f}s "
          f"score={m2['score']:.3f} z={m2['z']:.1f}", flush=True)


if __name__ == "__main__":
    main()
