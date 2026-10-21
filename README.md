# 🔎 Drama Detective AI

Find the **exact episode and timestamp** of a scene from a drama playlist.

1. Enter the drama name and paste a public YouTube **playlist URL** (or a single video URL).
2. The app indexes every episode's metadata and **publicly accessible captions/transcripts** in the background, with live progress.
3. Describe the scene in **English, Roman Urdu, Urdu, or Hindi** — dialogue, characters, clothing, location, events.
4. Get the best-matching **episode, timestamp (HH:MM:SS), a clickable link that opens the video at that exact second**, the matching dialogue as evidence, an explanation of *why* it matches, a plain **confidence level (High / Medium / Low)**, and alternative candidates.

Also included: **clip finder** (upload a short clip → its episode + timestamp), **video upload** (any video file → transcribed → searchable at the exact second), **recent searches**, and **feedback/correction** when a suggested episode is wrong.

## When captions are missing

An episode counts as searchable **only** when transcript segments are actually indexed for it — the app never fakes coverage. Three real fallbacks, all in the panel under the playlist:

1. **Broader caption fetch (automatic).** The fetcher no longer asks YouTube for English only: it tries English, Urdu and Hindi tracks (manual or auto), then one further language chosen from the video's track list. Downloads go through the android player client first (YouTube bot-walls the web path from server IPs), with one paused retry, because YouTube also rate-limits (HTTP 429) subtitle bursts — both are transient, and re-analyzing later fills in episodes that were blocked. (The English-only request was the actual root cause for playlists where most episodes showed as caption-less.)

## Playlist integrity

Every playlist has two identities, and the app never confuses them:

- an internal id (`pid`) that owns its videos, transcripts, embeddings and search results — everything is bound to it, so one drama's data can never surface inside another's;
- a canonical **source id** (`pl:<YouTube playlist id>` or `vi:<video id>`) recorded when the URL is submitted.

What this means in practice:

- A submitted URL is validated before anything is created; an unreadable/private source ends in an explicit **error state**, never in a silent fall-back to a previous drama.
- After an analysis, the app displays the **confirmed** identity — the title YouTube actually returned, the real video count, and the source id — not just the name you typed. Re-submitting the same source creates a separate, clearly-labelled entry (fresh data for re-checks); it is never silently merged with, or served from, the earlier one.
- If your last submitted playlist never became ready, Step 2 and the clip finder say so plainly and name the playlist searches *will* run against, until you correct the link. Clip search in that situation is blocked outright.
- Search caches are keyed by playlist id and invalidated whenever that playlist's data changes (new transcripts, subtitle uploads, re-analysis), so stale results cannot leak into a new search.

## Universal clip finder

Upload a short clip of a scene — from a drama of any country or language — and the app works out which episode of the **selected playlist** it came from:

- **Original language first.** Whisper transcribes the clip in its detected original language (the detected language and detector confidence are shown; the original text is preserved). When the clip isn't English, a second pass translates it to English, because many episodes carry English subtitles.
- **Unicode-aware dialogue alignment.** Word alignment tokenizes any script (Urdu, Arabic, Hindi, Turkish, Korean, Chinese…), so an Urdu clip can match Urdu captions directly, and the translated text can match English captions. Both passes run; the better evidence wins and the method is labelled on the result.
- **Separate visual verification.** The top dialogue candidate is additionally checked against real storyboard frames of its source video (perceptual frame comparison). The visual status is reported separately — it is a coarse supporting signal (storyboard previews are sparse), never proof on its own.
- **Silent clips.** A clip with no detectable speech goes to a bounded visual-only scan; anything it returns is labelled a low-confidence *hint*, never a verified match.
- **Honest negatives.** If nothing verifies, the app says **"No verified match in this playlist."** Search never leaks across playlists: another drama's transcripts or cached results are never used.
- **Evidence-tiered verdicts.** Every clip result carries a verdict computed from the actual evidence (alignment score *and* the number of matched words — a similarity score alone is never verification):
  - 🟢 **EXACT MATCH VERIFIED** — extensive, clean dialogue evidence (score ≥ 0.55, ≥ 10 matched words) for this specific clip and source segment.
  - 🟡 **PROBABLE MATCH** — real but imperfect evidence (e.g. a loose speech-to-text translation, or a short but distinctive phrase). The episode/scene is likely; the matched dialogue lines are shown so a human can confirm in seconds.
  - **Not verified** — anything weaker is suppressed entirely: the app returns "No verified match in this playlist." instead of dressing up a merely similar scene. Semantic-search candidates can never exceed *probable*, and visual-only hints never count as verification.
- **Embeddings note.** Semantic search uses `EMBED_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`, English-centric). For stronger non-English semantic recall you can set `EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` in the app secrets — at a real RAM cost on the free tier, which is why it isn't the default. Keyword/FTS search, transliteration expansion and dialogue alignment are not limited to English.
2. **Whisper transcription (optional, free, slow).** Per episode or for all missing episodes, as a tracked background job with visible progress and per-episode errors. Audio comes from the same public download path as everything else (no DRM/access bypass). On the free server a 35–40 minute episode can take roughly as long to transcribe, and the app must stay awake — the panel says so, and an episode is marked searchable only after its transcript is indexed. No API key is required.
3. **Your own subtitle file.** Upload a `.srt`/`.vtt` for a specific episode; its cues are parsed, indexed and embedded immediately.
4. **Visual fallback.** For episodes with no transcript at all, the panel can show real storyboard frames from across the episode so a scene can be spotted by eye and its approximate time noted. (Frames are evidence for the user; the app does not claim to recognize their content.)

## Exact Clip Finder (upgraded search results)

Every Step-2 result now also carries:

- **Estimated scene range** `HH:MM:SS – HH:MM:SS`, not just one second. The range is an *estimate* derived from the matched dialogue: the anchor segment's dialogue block (pauses up to 20s count as the same scene — real subtitle tracks pause inside scenes), trimmed to the outermost segments matching your distinctive terms plus a short lead-in/out, capped at 5 minutes. The watch link opens at the scene start.
- **Playlist coverage** — how many episodes were actually searched (`searchable/total`, %). Episodes without captions are counted and reported, never silently skipped. When coverage is under 50%, confidence is capped at Medium: the scene may be in an episode that could not be checked.
- **Playable preview** — the top result embeds YouTube's official player cued to the scene's start/end. It is a player, **not** a downloadable clip.
- **Visual check** — on demand, the app extracts real frames from YouTube's public storyboard preview images at the scene's start/middle/end (downloaded via yt-dlp, cropped locally; cached per video). The frames are evidence for *your* eyes next to the dialogue — the app runs no vision model and never claims to have recognized characters or actions in a frame. If a video has no storyboards, the app says so instead of pretending.
- The top three candidates are ranked; you can confirm or correct the match (feedback is stored locally), and weaker alternatives are listed with their scene ranges.

---

## Evidence rules (never faked)

- Matches are computed **only** from real caption/transcript text that is legitimately accessible (public caption tracks via yt-dlp / youtube-transcript-api, or the official YouTube Data API for metadata when a key is provided).
- If an episode has **no captions**, the app says so plainly and offers **Whisper transcription** (background, from the UI) — it never pretends to have watched or heard a video it could not access.
- Private, deleted, or region-blocked videos are skipped, never invented.
- **Confidence is derived from explicit matching signals** (dialogue match strength + distinctive-term coverage of the scene) and is **never shown as 100%**. A query whose distinctive terms (names, places) appear nowhere can never be labelled High — this guard is covered by tests (it exists because of a real false positive in a predecessor app: a generic "…dies" query once matched the wrong episode with high confidence).

## Architecture (chosen for this environment)

Environment inspection (Oct 2026): Python 3.12 VM, no paid hosting, no credit card available for cloud billing. A previous app for the same user (**SceneFinder**) is live on **Streamlit Community Cloud** (free, no card) using this same backend design — so Drama Detective AI keeps the proven deployment shape instead of inventing an unproven one:

| Layer | Choice | Why |
|---|---|---|
| Frontend | **Streamlit** (`streamlit/app.py`), mobile-first, custom CSS | Deploys free on Streamlit Cloud; responsive; app-like |
| Backend/API | Shared Python package (`backend/`) + optional **FastAPI** app (`backend/app.py`) exposing the same features over HTTP | One codebase, two front doors |
| Database | **SQLite** + **FTS5** full-text index (`data/drama_detective.db`) | Zero-ops, persistent, ships with Python |
| Background jobs | In-process worker threads + `jobs` table with stage/current/total progress | Works inside Streamlit's model; progress is real |
| Metadata | YouTube Data API v3 if `YOUTUBE_API_KEY` set, else **yt-dlp** | Official API preferred; keyless fallback |
| Captions | Public caption tracks only (manual first, then auto) via yt-dlp / youtube-transcript-api | Authorized access only; no bypassing |
| Missing captions | **faster-whisper** transcription (opt-in per playlist, background) | Makes caption-less episodes searchable honestly |
| Semantic search | **fastembed** (all-MiniLM-L6-v2, ONNX, CPU) | No torch, no paid API; degrades gracefully to keyword-only |
| Clip matching | Whisper translate → word alignment (`backend/align.py`) + semantic fallback | Proven on real fan-edit clips |

### Matching pipeline (spec A–H)

Validate URL → fetch playlist metadata → fetch/index transcripts (cached in DB; re-analysis reuses them) → normalize the query (script detection; Roman Urdu spelling variants; Roman Urdu/Urdu/Hindi → English concept expansion, e.g. *vakil → lawyer, dulha → groom, shaadi → marriage*) → extract distinctive terms → FTS5 BM25 dialogue search + semantic search → blend scores → **distinctive-term coverage measured over the scene window (±60s)** → hard caps (zero coverage ⇒ never High; partial ⇒ at most Medium) → ranked results with evidence lines, reasons, and alternatives.

Concurrency & politeness are configurable: `TRANSCRIPT_WORKERS`, `REQUEST_DELAY`, retries inside the caption fetcher, `MAX_VIDEOS_PER_PLAYLIST`.

### Database schema

`playlists(id, url, title, drama_name, video_count, status, created_at, analyzed_at)` · `videos(id, playlist_id, youtube_id, title, position, duration, url, thumbnail, has_transcript)` · `segments(id, video_id, start, end, text)` + `segments_fts` (FTS5) · `segment_embeddings(segment_id, dim, embedding)` · `jobs(id, kind, playlist_id, status, stage, current, total, error, …)` · `searches(id, playlist_id, query, result_video_id, result_timestamp, confidence, created_at)` · `feedback(id, playlist_id, query, result_video_id, verdict, correct_video_id, note, created_at)`

Search history and feedback are stored **only in the app's own database** — nothing is sent anywhere.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r streamlit/requirements.txt
.venv/bin/streamlit run streamlit/app.py        # http://localhost:8501
# optional API layer:
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.app:app --port 8000
```

Copy `.env.example` to `.env` for configuration (all values optional — the app works with **no API keys**).

## Tests

```bash
.venv/bin/pip install pytest numpy
.venv/bin/python -m pytest tests/ -q     # 12 tests, offline, real subtitle fixtures
.venv/bin/python tools/smoke_test.py     # boots the Streamlit UI headlessly
```

The search tests run against **real subtitle data** with hand-verified ground truth:

- *Raja London Ka* EP25 lawyer scene (official English subtitles): an English dialogue query **and** a Roman Urdu description both rank EP25 first at ~00:14:25 — *"Rania Mirza / this marriage cannot take place"*, fees + tea money, *"no judge will come here either"*.
- *Ertugrul* EP76 (bundled verified transcript): "Suleman shah ki death" → EP76 @ **00:11:40**, High confidence.
- Negative controls: "Shahbudin dies scene" (a name that appears nowhere) can never return High confidence.

Live-path verification (Oct 21, 2026): playlist fetch returned all 32 *Raja London Ka* episodes; EP25 caption fetch returned 493 segments with the scene lines at 836s and 916s, matching the ground truth above.

## Deploy (Streamlit Community Cloud, free)

1. Push this repo to GitHub.
2. streamlit.io → **New app** → pick the repo → main file: `streamlit/app.py`.
3. (Optional) add `YOUTUBE_API_KEY` in the app's Secrets if you have one.
4. Deploy. First analysis of a playlist downloads captions; the semantic model (~90 MB) downloads on first search.

## Honest limitations

- Only **public** playlists/videos; only captions that YouTube exposes publicly. The YouTube Data API does **not** provide arbitrary transcripts — captions come from the caption tracks themselves.
- Episodes without captions need Whisper transcription (slow: roughly real-time or slower on free CPU) before they can be searched.
- Roman Urdu / Urdu / Hindi descriptions match English subtitles through concept expansion + semantic search; exact dialogue quotes in the subtitle language remain the strongest signal.
- On Streamlit Cloud's free tier the app sleeps when idle and its disk is ephemeral — analyzed data may need re-indexing after a cold restart (transcripts are re-fetched from YouTube, which is the source of truth).

## Project layout

```
backend/    config, db (SQLite/FTS5), youtube, ingest (jobs), search,
            normalize (multilingual), embeddings, analyze (Whisper/clip),
            align, vtt, app (FastAPI)
streamlit/  app.py (UI), requirements.txt, seeds.json (verified transcripts
            for caption-less episodes)
tests/      pytest suite + real subtitle fixtures
tools/      smoke_test.py (headless UI boot check)
```
