"""Playlist-selection policy for the Streamlit UI (pure, testable).

The wrong-playlist bug this module exists to prevent: the app once let
the playlist picker silently keep a previously selected drama after
the user submitted a new one, so clip/scene searches ran against the
wrong playlist's transcripts while the user believed they were
searching the new drama. These helpers encode the safe policy in one
place so it can be regression-tested without a browser:

  * Selection is by stable playlist ID, never by display label
    (labels can collide when two playlists share a name).
  * A just-analyzed playlist becomes the selection.
  * Search results belong to the playlist that produced them; when
    the selection changes, stale results must be dropped.
  * If the user submitted a playlist that never became ready
    (analysis failed or found nothing), clip search against a
    *different* selected playlist is blocked with a clear error
    instead of silently returning the wrong drama's episode.
"""


def playlist_label(p):
    """Human label for one playlist row (dict with drama_name/title,
    video_count, url)."""
    name = p.get("drama_name") or p.get("title") or (p.get("url") or "")[:40]
    return f"{name} ({p.get('video_count', 0)} videos)"


def unique_labels(playlists):
    """{playlist_id: label} with guaranteed-unique labels.

    The old UI keyed its options dict by label, so two playlists with
    the same name and video count silently collapsed into one picker
    entry. Collisions now get a stable ' · #<id>' suffix."""
    labels, seen = {}, {}
    for p in playlists:
        lab = playlist_label(p)
        if lab in seen:
            lab = f"{lab} · #{p['id']}"
        seen[lab] = p["id"]
        labels[p["id"]] = lab
    return labels


def default_selection(ready, active_pid):
    """Which playlist to preselect when the picker has no value yet:
    the active (just-analyzed) playlist when it is ready, otherwise
    the largest ready playlist. None when nothing is ready."""
    if not ready:
        return None
    ids = [p["id"] for p in ready]
    if active_pid in ids:
        return active_pid
    biggest = max(ready, key=lambda p: p.get("video_count", 0) or 0)
    return biggest["id"]


def stale_results(results_pid, selected_pid):
    """True when on-screen search results were produced under a
    different playlist than the one now selected (they must be
    cleared, never re-labelled with the new drama's name)."""
    return results_pid is not None and results_pid != selected_pid


def clip_block_reason(submitted, selected_pid, ready_ids):
    """Error message when a clip search must be blocked, else None.

    `submitted` is {"pid": int, "name": str} for the playlist the user
    most recently submitted for analysis. The search is blocked only
    in the dangerous case: the submitted playlist never became ready
    AND the current selection is a different playlist — exactly the
    situation where the app would silently search the wrong drama.
    If the submitted playlist IS ready, switching away from it is the
    user's explicit choice and is allowed."""
    if not submitted:
        return None
    sub_pid = submitted.get("pid")
    if sub_pid is None or sub_pid == selected_pid:
        return None
    if sub_pid in ready_ids:
        return None
    name = submitted.get("name") or "submitted"
    return (
        f"⛔ Clip search rok di gayi hai. Aap ne '{name}' ki playlist "
        "submit ki thi, lekin wo abhi ready nahi hai (analysis "
        "mukammal nahi hua, fail ho gaya, ya koi video nahi mili). "
        "Is waqt selected playlist is se mukhtalif hai — ghalat drama "
        "ke episode se bachne ke liye clip us par match nahi kiya "
        "gaya. Pehle upar Step 1 se sahi playlist analyze karke ready "
        "karein, phir clip upload karein.")


def identity_notice(submitted, selected_pid, by_id):
    """A 'stop and correct' notice for Step 2/3, or None.

    Shown when the playlist the user most recently submitted is NOT
    the one currently selected because it never became ready: until
    the user corrects the link (or deliberately picks another ready
    playlist), every search screen must say plainly which playlist
    searches will actually run against, so a failed submission can
    never masquerade as a successful one. `by_id` maps playlist ids
    to their rows for every playlist (any status)."""
    if not submitted:
        return None
    sub_pid = submitted.get("pid")
    if sub_pid is None or sub_pid == selected_pid:
        return None
    sub = by_id.get(sub_pid)
    if sub is not None and sub.get("status") == "ready":
        return None  # submitted one is fine; the switch was explicit
    sel = by_id.get(selected_pid) or {}
    sel_name = sel.get("drama_name") or sel.get("title") or "selected"
    name = submitted.get("name") or (sub or {}).get("drama_name") \
        or "submitted"
    return (
        f"⚠️ Identity check: aap ki aakhri submitted playlist "
        f"('{name}') ready nahi hui — is liye abhi search sirf "
        f"selected playlist '{sel_name}' par chalega, us nayi "
        "playlist par NAHI. Agar aap ko usi nayi playlist par search "
        "chahiye tha, to Step 1 mein us ka link theek karke dobara "
        "analyze karein.")


def confirmed_identity(p):
    """One-line confirmed identity for a playlist row, built ONLY
    from what was actually fetched (YouTube title, canonical source
    id, real video count) — never from the user's typed name alone.
    Returns '' when the row has no fetched title yet."""
    title = (p or {}).get("title")
    if not title:
        return ""
    src = (p or {}).get("source_id") or ""
    count = (p or {}).get("video_count", 0)
    return f"🔗 Confirmed: '{title}' • {count} videos" + \
        (f" • source {src}" if src else "")
