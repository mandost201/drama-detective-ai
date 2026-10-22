"""Audio fingerprint matching - a Whisper-independent channel.

Why this exists (measured 2026-10-22): every other channel in the
pipeline depends on Whisper HEARING the clip correctly. On clips
with music under dialogue the recognizer sometimes invents English
dialogue or loops one phrase, in the local runtime AND on Streamlit
- and then no text channel can honestly match. The clip's SOUND,
however, is the same recording as the source episode's sound.
This module compares sound directly:

  audio -> 8 kHz mono -> mel-band log-energy frames (64 ms hop)
        -> per-band standardised, per-frame L2-normalised vectors
        -> FFT cross-correlation of the clip's frame sequence
           against the episode's, for every possible offset.

The best offset is where the clip's sound sits on the episode's
timeline. A true match (same recording, different upload/codec)
scores far above wrong episodes and wrong offsets - calibration
numbers live in tools/audio_calibrate.py output and in the tests.
Scores are means of per-frame cosine similarities in [-1, 1].

Nothing here transcribes or invents dialogue: it aligns sound.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SR = 8000                 # decode/sample rate
FRAME = 1024              # STFT window (128 ms)
HOP = 512                 # STFT hop (64 ms)
BANDS = 24                # mel bands
FMIN, FMAX = 200.0, 3800.0
HOP_S = HOP / SR


def decode_mono(path, sr=SR):
    """Decode any audio/video file to mono float32 at `sr` Hz via
    PyAV (no ffmpeg binary needed - Streamlit Cloud safe)."""
    import av
    chunks = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams
                       if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError(f"no audio stream in {path}")
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError(f"no audio decoded from {path}")
    return (np.concatenate(chunks).astype(np.float32) / 32768.0)


def _mel_filterbank():
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), BANDS + 2))
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)
    fb = np.zeros((BANDS, len(freqs)), dtype=np.float32)
    for b in range(BANDS):
        lo, ce, hi = pts[b], pts[b + 1], pts[b + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        dn = (hi - freqs) / max(hi - ce, 1e-9)
        fb[b] = np.clip(np.minimum(up, dn), 0.0, None)
    return fb


_FB = _mel_filterbank()
_WIN = np.hanning(FRAME).astype(np.float32)


def features_from_signal(x):
    """(T, BANDS) RAW mel log-energy features for a mono float32
    signal. Standardisation is NOT applied here: it must use ONE
    reference (the episode's band statistics) for both sides of a
    comparison - standardising a 60 s clip by its own statistics
    and a 40 min episode by its own made identical sounds compare
    as different (measured 2026-10-22: a slice cut from an episode
    did not match itself). See prepare_pair."""
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n_frames = 1 + (len(x) - FRAME) // HOP
    feats = np.empty((n_frames, BANDS), dtype=np.float32)
    block = 4096
    for i0 in range(0, n_frames, block):
        i1 = min(n_frames, i0 + block)
        idx = (np.arange(i0, i1)[:, None] * HOP
               + np.arange(FRAME)[None, :])
        frames = x[idx] * _WIN
        power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2)
        feats[i0:i1] = np.log10(power @ _FB.T + 1e-9)
    return feats


def _standardize(raw, mu, sd):
    z = np.clip((raw - mu) / sd, -4.0, 4.0)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norms, 1e-9)


def prepare_pair(ep_raw, clip_raw):
    """Standardise episode AND clip features with the EPISODE's
    per-band mean/std, then per-frame L2-normalise. Identical
    sounds then produce identical vectors on both sides."""
    mu = ep_raw.mean(axis=0, keepdims=True)
    sd = ep_raw.std(axis=0, keepdims=True) + 1e-6
    return _standardize(ep_raw, mu, sd), _standardize(clip_raw, mu, sd)


def audio_features(path, cache_dir=None, cache_key=None):
    """Features for a file, with an on-disk .npz cache (keyed by an
    explicit key or the file's path+size+mtime) so repeat searches
    never recompute - or re-download - an episode's sound."""
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_key is None:
            p = Path(path)
            raw = f"{p}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
            cache_key = hashlib.sha1(raw.encode()).hexdigest()[:16]
        cache_path = cache_dir / f"{cache_key}.npz"
        if cache_path.exists():
            try:
                with np.load(cache_path) as z:
                    return z["feats"]
            except Exception:
                pass  # corrupt cache: recompute below
    feats = features_from_signal(decode_mono(path))
    if cache_path is not None:
        try:
            np.savez_compressed(cache_path, feats=feats)
        except Exception:
            pass
    return feats


def correlation_curve(ep_raw, clip_raw):
    """Mean per-frame cosine similarity of the clip sequence placed
    at every frame offset in the episode. Both sides are first
    standardised with the episode's band statistics (prepare_pair).
    Returns a float32 array of length T_ep - T_clip + 1 (empty if
    the clip is longer)."""
    T, S = len(ep_raw), len(clip_raw)
    if S == 0 or T < S:
        return np.zeros(0, dtype=np.float32)
    ep_feats, clip_feats = prepare_pair(ep_raw, clip_raw)
    n = 1 << int(np.ceil(np.log2(T + S)))
    # Correlation theorem: irfft(rfft(E) * conj(rfft(C)))[o]
    #   = sum_s E[o+s] * C[s]   (no wraparound for o <= T-S
    #   since both sequences are zero-padded to n >= T+S).
    # (An earlier version ALSO time-reversed C - reversal plus
    # conjugation computes the wrong alignment entirely.)
    F_e = np.fft.rfft(ep_feats, n=n, axis=0)
    F_c = np.fft.rfft(clip_feats, n=n, axis=0)
    corr = np.fft.irfft(F_e * np.conj(F_c), n=n, axis=0)
    return (corr[:T - S + 1].sum(axis=1) / S).astype(np.float32)


def _parabolic(curve, i):
    if 0 < i < len(curve) - 1:
        a, b, c = float(curve[i - 1]), float(curve[i]), float(curve[i + 1])
        d = a - 2 * b + c
        if d < 0:
            return float(np.clip(0.5 * (a - c) / d, -1.0, 1.0))
    return 0.0


def best_match(ep_feats, clip_feats, lo_s=0.0, hi_s=None):
    """Best placement of the clip inside [lo_s, hi_s) of the
    episode. Returns dict(offset_s, score, z, second_score) or
    None when the clip cannot fit / the window is empty."""
    curve = correlation_curve(ep_feats, clip_feats)
    if len(curve) == 0:
        return None
    lo = max(0, int(lo_s / HOP_S))
    hi = len(curve) if hi_s is None else min(len(curve),
                                             int(hi_s / HOP_S) + 1)
    if hi <= lo:
        return None
    seg = curve[lo:hi]
    i = int(np.argmax(seg)) + lo
    score = float(curve[i])
    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med))) + 1e-9
    z = (score - med) / (1.4826 * mad)
    # strongest peak FAR from the winner (outside one clip length)
    S = len(clip_feats)
    far = curve.copy()
    far[max(0, i - S):i + S + 1] = -np.inf
    second = float(far.max()) if np.isfinite(far.max()) else -1.0
    return {"offset_s": (i + _parabolic(curve, i)) * HOP_S,
            "score": score, "z": float(z), "second_score": second,
            "curve_median": med}
