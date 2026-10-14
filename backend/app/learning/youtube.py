"""YouTube Data API v3 search for one skill (Phase 10, docs/adr/0018). Real results only: every video comes back
from the API with its id, and its row records the call that produced it.

    search.list  part=snippet, type=video, q="<skill> tutorial", relevanceLanguage=en, safeSearch=strict
    videos.list  part=snippet,contentDetails for the returned ids (full description and duration)

Quota: search.list costs 100 units and videos.list 1, of a default 10,000 a day, so results are cached per query
(table youtube_searches) and each request is capped (config/learning.yaml). The key is never logged or stored.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API = "https://www.googleapis.com/youtube/v3"


class YouTubeUnavailable(RuntimeError):
    """No key, quota exhausted, or the API failed: the caller reports it and uses the catalogue only."""


def _get(endpoint: str, params: dict, key: str) -> dict:
    url = f"{API}/{endpoint}?" + urllib.parse.urlencode({**params, "key": key})
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:  # the message, never the URL (it holds the key)
        try:
            reason = json.loads(exc.read())["error"]["message"]
        except Exception:  # noqa: BLE001
            reason = exc.reason
        raise YouTubeUnavailable(f"YouTube API {endpoint} HTTP {exc.code}: {reason}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise YouTubeUnavailable(f"YouTube API {endpoint} unreachable: {type(exc).__name__}") from None


def iso_duration(value: str | None) -> str | None:
    """'PT2H24M25S' -> '2 h 24 min 25 s' (a format change only; None when the API gave no duration)."""
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value or "")
    if not value or not m or not any(m.groups()):
        return None
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    h += 24 * d
    return " ".join(f"{v} {u}" for v, u in ((h, "h"), (mi, "min"), (s, "s")) if v) or "0 s"


def search_videos(query: str, key: str, max_results: int) -> tuple[list[dict], dict]:
    """[{video row}], request record. Rows are learning_resources-shaped, ready to upsert."""
    if not key:
        raise YouTubeUnavailable("YOUTUBE_API_KEY is not set")
    params = {"part": "snippet", "type": "video", "q": query, "maxResults": max_results, "relevanceLanguage": "en",
              "safeSearch": "strict"}
    found = _get("search", params, key)
    ids = [it["id"]["videoId"] for it in found.get("items", []) if it.get("id", {}).get("videoId")]
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    request = {"endpoint": "search.list", **params, "fetched_at": fetched}
    if not ids:
        return [], request
    details = _get("videos", {"part": "snippet,contentDetails", "id": ",".join(ids)}, key)
    by_id = {it["id"]: it for it in details.get("items", [])}
    rows = []
    for vid in ids:  # the search's own order
        it = by_id.get(vid)
        if it is None:  # removed between the two calls: never filled in
            continue
        sn, duration = it["snippet"], iso_duration(it.get("contentDetails", {}).get("duration"))
        rows.append({
            "resource_id": f"youtube:{vid}", "title": sn["title"], "url": f"https://www.youtube.com/watch?v={vid}",
            "provider": f"YouTube: {sn.get('channelTitle', '')}".strip(), "description": sn.get("description") or None,
            "skills": [], "difficulty": None, "prerequisites": None,
            "estimated_effort": f"video length {duration}" if duration else None,
            "source_reference": (f"YouTube Data API v3 videos.list id={vid} (via search.list q={query!r}), "
                                 f"fetched {fetched}"),
            "kind": "video",
        })
    return rows, request
