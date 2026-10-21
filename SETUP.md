# Drama Detective AI — Setup Guide

Complete instructions to run the app on your own computer and to deploy
it later. Everything the app needs is in this package — no file is
missing. (Not included on purpose: `.venv/` and `data/` — both are
**generated automatically**: the virtual environment by pip, and the
SQLite database on first run.)

## What you need

- **Python 3.11 or newer** (developed/tested on 3.12) — python.org
- An internet connection (first run downloads the search model ~90 MB;
  playlist analysis downloads captions from YouTube)
- No API keys, no credit card. A YouTube Data API key is *optional*
  (see `.env.example`).

## Run it locally — Windows

```bat
cd drama-detective-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r streamlit\requirements.txt
streamlit run streamlit\app.py
```

## Run it locally — macOS / Linux

```bash
cd drama-detective-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r streamlit/requirements.txt
streamlit run streamlit/app.py
```

Then open **http://localhost:8501** in your browser.

1. In **Step 1**, type the drama name and paste a public YouTube
   playlist URL (or a single video URL) → **Analyze playlist**.
   Wait for the progress bar (captions are being indexed).
2. In **Step 2**, describe the scene — English, Roman Urdu, Urdu or
   Hindi — and press **Find scene**.
3. Episodes without captions show a warning and a
   **Transcribe episodes** button (Whisper runs in the background;
   slow on CPU, but it makes those episodes searchable too).

## Optional configuration

Copy `.env.example` to `.env` and edit. All values are optional:

| Variable | Purpose |
|---|---|
| `YOUTUBE_API_KEY` | Official YouTube Data API for playlist metadata (app falls back to yt-dlp without it) |
| `WHISPER_MODEL` | `tiny` / `base` / `small` — transcription quality vs speed |
| `AUTO_TRANSCRIBE_MISSING=1` | Auto-transcribe caption-less episodes during analysis |
| `TRANSCRIPT_WORKERS`, `REQUEST_DELAY` | Download parallelism / politeness |
| `MAX_VIDEOS_PER_PLAYLIST` | Playlist size cap (default 500) |

## Run the tests (optional)

```bash
pip install pytest numpy
python -m pytest tests/ -q        # 12 tests on real subtitle fixtures
python tools/smoke_test.py        # boots the UI headlessly
```

## Deploy later (free, Streamlit Community Cloud)

1. Create a GitHub repository and push this whole folder to it.
2. Go to **share.streamlit.io** → sign in with GitHub → **New app**.
3. Pick the repository, branch `main` (or `master`), and set the main
   file path to **`streamlit/app.py`**.
4. (Optional) In **Advanced settings → Secrets**, add
   `YOUTUBE_API_KEY = "..."` if you have one.
5. **Deploy.** The app gets a public URL like
   `https://<name>.streamlit.app`.

Notes: on the free tier the app sleeps when idle (first visit wakes it)
and its disk is temporary, so playlists may need re-analyzing after a
cold restart. Captions always come from YouTube, the source of truth.

## Folder map

```
streamlit/app.py          the web app (UI)
streamlit/requirements.txt  app dependencies
streamlit/seeds.json      verified transcripts for caption-less episodes
backend/                  search engine, database, YouTube, Whisper, API
backend/app.py            optional FastAPI layer (uvicorn backend.app:app)
tests/                    automated tests + real subtitle fixtures
tools/smoke_test.py       headless UI boot check
.env.example              environment variables template
README.md                 full documentation (architecture, schema, limits)
```
