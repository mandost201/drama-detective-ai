"""SQLite storage: playlists, videos, transcript segments, embeddings, jobs."""
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH

_lock = threading.RLock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS playlists (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    video_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    created_at REAL,
    analyzed_at REAL
);
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_id TEXT NOT NULL,
    youtube_id TEXT NOT NULL,
    title TEXT,
    position INTEGER,
    duration REAL,
    url TEXT,
    thumbnail TEXT,
    has_transcript INTEGER DEFAULT 0,
    UNIQUE(playlist_id, youtube_id)
);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS segment_embeddings (
    segment_id INTEGER PRIMARY KEY,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(text);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT,
    playlist_id TEXT,
    status TEXT DEFAULT 'queued',
    stage TEXT DEFAULT '',
    current INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    error TEXT,
    created_at REAL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_id TEXT,
    query TEXT,
    result_video_id INTEGER,
    result_timestamp REAL,
    confidence TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_id TEXT,
    query TEXT,
    result_video_id INTEGER,
    verdict TEXT,
    correct_video_id INTEGER,
    note TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);
CREATE INDEX IF NOT EXISTS idx_videos_playlist ON videos(playlist_id);
"""


def _connect():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _lock, _connect() as con:
        con.executescript(SCHEMA)
        # migration: playlists.drama_name (user-supplied drama title,
        # kept separate from the YouTube playlist title)
        cols = [r[1] for r in con.execute("PRAGMA table_info(playlists)")]
        if "drama_name" not in cols:
            con.execute("ALTER TABLE playlists ADD COLUMN drama_name TEXT")
        # migration: videos.transcript_source (captions / whisper /
        # subtitle / seed / upload) - powers honest coverage stats
        vcols = [r[1] for r in con.execute("PRAGMA table_info(videos)")]
        if "transcript_source" not in vcols:
            con.execute("ALTER TABLE videos ADD COLUMN transcript_source TEXT")
        # migration: playlists.source_id - the canonical identity of
        # the source (YouTube playlist id, or a single video id),
        # independent of the user-typed drama name and of this app's
        # internal pid. Two entries can share a title; they can
        # never share a (source_id, pid) identity by accident.
        pcols = [r[1] for r in con.execute("PRAGMA table_info(playlists)")]
        if "source_id" not in pcols:
            con.execute("ALTER TABLE playlists ADD COLUMN source_id TEXT")
        # backfill source_id for rows created before it existed
        for r in con.execute(
                "SELECT id, url FROM playlists WHERE source_id IS NULL"
        ).fetchall():
            sid = source_id_for(r[1])
            if sid:
                con.execute("UPDATE playlists SET source_id=? WHERE id=?",
                            (sid, r[0]))


@contextmanager
def get_con():
    with _lock:
        con = _connect()
        try:
            yield con
            con.commit()
        finally:
            con.close()


# ---- playlists ----
def source_id_for(url):
    """Canonical source identity for a submitted URL:
    'pl:<youtube playlist id>' or 'vi:<youtube video id>', else
    None (identity uncertain - the caller must not proceed as if
    the source were known)."""
    from .youtube import extract_playlist_id, extract_video_id
    pl = extract_playlist_id(url or "")
    if pl:
        return f"pl:{pl}"
    vi = extract_video_id(url or "")
    if vi:
        return f"vi:{vi}"
    return None


def find_by_source(source_id):
    """All playlist rows whose canonical source matches (there can
    be several analyses of the same source; each keeps its own pid
    and its own data - they are never merged silently)."""
    if not source_id:
        return []
    with get_con() as con:
        rows = con.execute("SELECT * FROM playlists WHERE source_id=?",
                           (source_id,)).fetchall()
        return [dict(r) for r in rows]


def create_playlist(url, title=None):
    pid = uuid.uuid4().hex[:12]
    now = time.time()
    with get_con() as con:
        con.execute(
            "INSERT INTO playlists(id,url,title,created_at,status,source_id)"
            " VALUES(?,?,?,?,?,?)",
            (pid, url, title, now, "pending", source_id_for(url)),
        )
    return pid


def update_playlist(pid, **fields):
    sets = ", ".join(f"{k}=?" for k in fields)
    with get_con() as con:
        con.execute(f"UPDATE playlists SET {sets} WHERE id=?", (*fields.values(), pid))


def get_playlist(pid):
    with get_con() as con:
        row = con.execute("SELECT * FROM playlists WHERE id=?", (pid,)).fetchone()
        return dict(row) if row else None


def list_playlists():
    with get_con() as con:
        rows = con.execute(
            "SELECT * FROM playlists ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_playlist(pid):
    with get_con() as con:
        vids = [r[0] for r in con.execute(
            "SELECT id FROM videos WHERE playlist_id=?", (pid,)).fetchall()]
        if vids:
            q = ",".join("?" * len(vids))
            con.execute(f"DELETE FROM segment_embeddings WHERE segment_id IN "
                        f"(SELECT id FROM segments WHERE video_id IN ({q}))", vids)
            con.execute(f"DELETE FROM segments_fts WHERE rowid IN "
                        f"(SELECT id FROM segments WHERE video_id IN ({q}))", vids)
            con.execute(f"DELETE FROM segments WHERE video_id IN ({q})", vids)
            con.execute("DELETE FROM videos WHERE playlist_id=?", (pid,))
        con.execute("DELETE FROM playlists WHERE id=?", (pid,))


# ---- videos ----
def add_videos(pid, videos):
    """videos: list of dicts(youtube_id,title,position,duration,url,thumbnail)."""
    with get_con() as con:
        for v in videos:
            con.execute(
                """INSERT OR IGNORE INTO videos
                   (playlist_id,youtube_id,title,position,duration,url,thumbnail)
                   VALUES(?,?,?,?,?,?,?)""",
                (pid, v["youtube_id"], v.get("title"), v.get("position"),
                 v.get("duration"), v.get("url"), v.get("thumbnail")),
            )


def get_videos(pid):
    with get_con() as con:
        rows = con.execute(
            "SELECT * FROM videos WHERE playlist_id=? ORDER BY position", (pid,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_video(vid):
    with get_con() as con:
        row = con.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
        return dict(row) if row else None


def mark_transcript(video_id, has, source=None):
    with get_con() as con:
        if source is not None:
            con.execute("UPDATE videos SET has_transcript=?, "
                        "transcript_source=? WHERE id=?",
                        (1 if has else 0, source if has else None,
                         video_id))
        else:
            con.execute("UPDATE videos SET has_transcript=? WHERE id=?",
                        (1 if has else 0, video_id))


def playlist_stats(pid):
    """Honest per-playlist accounting for the results header:
    discovered / searchable / missing / transcribed-by-source /
    skipped (private or deleted entries)."""
    vids = get_videos(pid)
    by_source = {}
    searchable = 0
    skipped = 0
    for v in vids:
        if v.get("has_transcript"):
            searchable += 1
            src = v.get("transcript_source") or "captions"
            by_source[src] = by_source.get(src, 0) + 1
        title = (v.get("title") or "").strip().lower()
        if title in ("private video", "deleted video",
                     "[private video]", "[deleted video]"):
            skipped += 1
    return {
        "total": len(vids), "searchable": searchable,
        "missing": len(vids) - searchable, "by_source": by_source,
        "transcribed": sum(by_source.get(s, 0) for s in
                           ("whisper", "subtitle", "upload")),
        "skipped": skipped,
    }


# ---- segments ----
def add_segments(video_id, segments):
    """segments: list of (start,end,text). Replaces existing for the video."""
    with get_con() as con:
        old = [r[0] for r in con.execute(
            "SELECT id FROM segments WHERE video_id=?", (video_id,))]
        if old:
            q = ",".join("?" * len(old))
            con.execute(f"DELETE FROM segment_embeddings WHERE segment_id IN ({q})", old)
            con.execute(f"DELETE FROM segments_fts WHERE rowid IN ({q})", old)
            con.execute("DELETE FROM segments WHERE video_id=?", (video_id,))
        for start, end, text in segments:
            cur = con.execute(
                "INSERT INTO segments(video_id,start,end,text) VALUES(?,?,?,?)",
                (video_id, start, end, text),
            )
            con.execute("INSERT INTO segments_fts(rowid,text) VALUES(?,?)",
                        (cur.lastrowid, text))


def get_segments(video_id):
    with get_con() as con:
        rows = con.execute(
            "SELECT id,start,end,text FROM segments WHERE video_id=? ORDER BY start",
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def segment_count(playlist_id):
    with get_con() as con:
        row = con.execute(
            """SELECT COUNT(*) c FROM segments s
               JOIN videos v ON v.id=s.video_id WHERE v.playlist_id=?""",
            (playlist_id,)).fetchone()
        return row["c"]


def save_embeddings(pairs):
    """pairs: list of (segment_id, dim, blob)."""
    with get_con() as con:
        con.executemany(
            "INSERT OR REPLACE INTO segment_embeddings(segment_id,dim,embedding) VALUES(?,?,?)",
            pairs,
        )


def load_embeddings(playlist_id):
    """Returns list of (segment_id, video_id, start, embedding_bytes)."""
    with get_con() as con:
        rows = con.execute(
            """SELECT e.segment_id, s.video_id, s.start, e.dim, e.embedding
               FROM segment_embeddings e
               JOIN segments s ON s.id=e.segment_id
               JOIN videos v ON v.id=s.video_id
               WHERE v.playlist_id=?""",
            (playlist_id,)).fetchall()
        return [(r["segment_id"], r["video_id"], r["start"], r["dim"], r["embedding"])
                for r in rows]


def fts_search(playlist_id, query, limit=200):
    """BM25 keyword search over transcript segments of a playlist."""
    # sanitize FTS5 query: quote each term. Cap is generous (24) so a
    # multilingual query's English concept expansions are not cut off.
    terms = [t for t in query.replace('"', "").split() if t]
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms[:24])
    with get_con() as con:
        rows = con.execute(
            """SELECT s.id, s.video_id, s.start, s.end, s.text,
                      bm25(segments_fts) AS rank
               FROM segments_fts f
               JOIN segments s ON s.id=f.rowid
               JOIN videos v ON v.id=s.video_id
               WHERE v.playlist_id=? AND segments_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (playlist_id, match, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ---- jobs ----
def create_job(kind, playlist_id=None):
    jid = uuid.uuid4().hex[:12]
    now = time.time()
    with get_con() as con:
        con.execute(
            "INSERT INTO jobs(id,kind,playlist_id,status,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?)",
            (jid, kind, playlist_id, "queued", now, now),
        )
    return jid


def update_job(jid, **fields):
    fields["updated_at"] = time.time()
    sets = ", ".join(f"{k}=?" for k in fields)
    with get_con() as con:
        con.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), jid))


def get_job(jid):
    with get_con() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return dict(row) if row else None


# ---- search history & feedback (Drama Detective AI) ----
def log_search(playlist_id, query, result_video_id=None,
               result_timestamp=None, confidence=None):
    """Record a search so the user can revisit it. Stored only in the
    app's own local database - never sent anywhere."""
    with get_con() as con:
        con.execute(
            "INSERT INTO searches(playlist_id,query,result_video_id,"
            "result_timestamp,confidence,created_at) VALUES(?,?,?,?,?,?)",
            (playlist_id, query, result_video_id, result_timestamp,
             confidence, time.time()),
        )


def recent_searches(playlist_id=None, limit=10):
    with get_con() as con:
        if playlist_id:
            rows = con.execute(
                "SELECT * FROM searches WHERE playlist_id=? "
                "ORDER BY created_at DESC LIMIT ?", (playlist_id, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM searches ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def log_feedback(playlist_id, query, result_video_id, verdict,
                 correct_video_id=None, note=None):
    """verdict: 'correct' or 'wrong'. When wrong, correct_video_id is the
    episode the user says actually contains the scene (manual correction)."""
    with get_con() as con:
        con.execute(
            "INSERT INTO feedback(playlist_id,query,result_video_id,verdict,"
            "correct_video_id,note,created_at) VALUES(?,?,?,?,?,?,?)",
            (playlist_id, query, result_video_id, verdict,
             correct_video_id, note, time.time()),
        )


def list_feedback(playlist_id=None, limit=50):
    with get_con() as con:
        if playlist_id:
            rows = con.execute(
                "SELECT * FROM feedback WHERE playlist_id=? "
                "ORDER BY created_at DESC LIMIT ?", (playlist_id, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
