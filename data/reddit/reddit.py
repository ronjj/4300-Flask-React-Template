import praw
import csv
import json
import re
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# WARNING: Do not hardcode credentials. Use a .env file with:
#   REDDIT_CLIENT_ID=...
#   REDDIT_SECRET=...

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

SUBREDDITS = ["soccer", "PremierLeague", "SerieA", "LaLiga"]
OUTPUT_CSV = SCRIPT_DIR / "all_leagues_reddit.csv"
PROGRESS_FILE = SCRIPT_DIR / "reddit_progress.json"
COMMENT_EXPAND_LIMIT = 0  # 0 = fast (top comments only), increase for deeper threads
SEARCH_LIMIT = 250  # max posts per search query
NUM_WORKERS = 4       # parallel threads (all share one Reddit instance + its rate limiter)
SAVE_INTERVAL = 5     # save progress every N completed players

MATCH_THREAD_RE = re.compile(
    r"\b(pre[\s\-]?match|match[\s\-]?thread|post[\s\-]?match[\s\-]?thread|post[\s\-]?match)\b",
    re.IGNORECASE,
)

# Each entry: (csv_path, column_mapping)
# column_mapping keys: player_id, player_name, first_name, last_name, known_name, team_short_name
LEAGUE_CSVS = [
    (
        PROJECT_ROOT / "prem" / "prem_all_players-1.csv",
        {
            "player_id": "player_id",
            "player_name": "player_name",
            "first_name": "first_name",
            "last_name": "last_name",
            "known_name": "known_name",
            "team_short_name": "team_short_name",
        },
    ),
    (
        PROJECT_ROOT / "serie-a" / "seriea_all_players.csv",
        {
            "player_id": "player_id",
            "player_name": "display_name",
            "first_name": "first_name",
            "last_name": "last_name",
            "known_name": "short_name",
            "team_short_name": "team_short_name",
        },
    ),
    (
        PROJECT_ROOT / "la-liga" / "laliga_all_players.csv",
        {
            "player_id": "player_id",
            "player_name": "player_name",
            "first_name": "",
            "last_name": "",
            "known_name": "nickname",
            "team_short_name": "team_shortname",
        },
    ),
]

OUTPUT_FIELDS = [
    "player_name",
    "player_id",
    "search_query",
    "subreddit",
    "type",
    "post_id",
    "post_title",
    "post_author",
    "post_score",
    "post_upvote_ratio",
    "post_created",
    "post_url",
    "post_text",
    "post_num_comments",
    "comment_id",
    "comment_author",
    "comment_body",
    "comment_score",
    "comment_created",
    "comment_parent_id",
]


def load_players() -> list[dict]:
    """Read unique players from all league CSVs. Deduplicates by player_id."""
    raw: dict[str, dict] = {}
    for csv_path, col_map in LEAGUE_CSVS:
        if not csv_path.exists():
            print(f"[warning] CSV not found, skipping: {csv_path}")
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row.get(col_map["player_id"], "").strip()
                if not pid or pid in raw:
                    continue
                name = row.get(col_map["player_name"], "").strip()
                known = row.get(col_map["known_name"], "").strip() if col_map["known_name"] else ""
                first = row.get(col_map["first_name"], "").strip() if col_map["first_name"] else ""
                last = row.get(col_map["last_name"], "").strip() if col_map["last_name"] else ""
                team = row.get(col_map["team_short_name"], "").strip()
                display = known or name
                if not display:
                    continue
                raw[pid] = {
                    "id": pid,
                    "name": display,
                    "first": first,
                    "last": last,
                    "team": team,
                }

    players = []
    for pid, info in sorted(raw.items(), key=lambda x: x[1]["name"]):
        queries = _build_search_queries(info)
        players.append({"id": pid, "name": info["name"], "search_queries": queries})
    return players


def _build_search_queries(info: dict) -> list[str]:
    """Build search queries for a player, handling single-word names."""
    display = info["name"]
    parts = display.split()

    if len(parts) >= 2:
        return [f'"{display}"']

    # Single-word name: try first+last, or name+team for disambiguation
    first, last = info["first"], info["last"]
    team = info["team"]
    queries = []
    if first and last:
        full = f"{first.split()[0]} {last.split()[0]}"
        if full.lower() != display.lower():
            queries.append(f'"{full}"')
    queries.append(f'"{display}" {team}' if team else f'"{display}" football')
    return queries


def is_match_thread(title: str) -> bool:
    return bool(MATCH_THREAD_RE.search(title))


def load_existing_post_ids() -> set[str]:
    """Read all post_ids already written to the output CSV."""
    if not OUTPUT_CSV.exists():
        return set()
    post_ids: set[str] = set()
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("post_id", "").strip()
            if pid:
                post_ids.add(pid)
    return post_ids


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_player_ids": [], "total_rows": 0}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def ensure_csv_header():
    if OUTPUT_CSV.exists():
        return
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=OUTPUT_FIELDS).writeheader()


def append_rows(rows: list[dict]):
    if not rows:
        return
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writerows(rows)


def search_subreddit(reddit: praw.Reddit, sub_name: str, query: str) -> list:
    """Search a subreddit and return list of submissions."""
    sub = reddit.subreddit(sub_name)
    posts = []
    try:
        for submission in sub.search(query, sort="relevance", limit=SEARCH_LIMIT):
            posts.append(submission)
    except Exception as e:
        print(f"    [error] search r/{sub_name} for {query!r}: {e}")
    return posts


def fetch_comments(submission) -> list[dict]:
    """Fetch comments for a submission (shallow expansion for speed)."""
    comments = []
    try:
        submission.comments.replace_more(limit=COMMENT_EXPAND_LIMIT)
        for comment in submission.comments.list():
            if not hasattr(comment, "body"):
                continue
            comments.append({
                "comment_id": comment.id,
                "comment_author": str(comment.author) if comment.author else "[deleted]",
                "comment_body": comment.body,
                "comment_score": comment.score,
                "comment_created": datetime.fromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                "comment_parent_id": comment.parent_id,
            })
    except Exception as e:
        print(f"    [error] comments for {submission.id}: {e}")
    return comments


def player_mentioned(text: str, name: str) -> bool:
    """Check if any part of the player's name appears in text (case-insensitive)."""
    text_lower = text.lower()
    # Check the full name
    if name.lower() in text_lower:
        return True
    # For multi-word names, check the last word (surname)
    parts = name.split()
    if len(parts) >= 2 and len(parts[-1]) > 3:
        return parts[-1].lower() in text_lower
    return False


def scrape_player(
    reddit: praw.Reddit, player: dict,
    skip_match_threads: bool = False,
    global_seen: set[str] | None = None,
) -> list[dict]:
    """Search for a single player across all subreddits. Returns rows for CSV."""
    name = player["name"]
    pid = player["id"]
    queries = player["search_queries"]
    seen_post_ids: set[str] = set()
    rows: list[dict] = []

    for query in queries:
        for sub_name in SUBREDDITS:
            posts = search_subreddit(reddit, sub_name, query)

            for submission in posts:
                if submission.id in seen_post_ids:
                    continue
                if global_seen is not None and submission.id in global_seen:
                    continue
                seen_post_ids.add(submission.id)

                title = submission.title or ""
                if skip_match_threads and is_match_thread(title):
                    continue
                selftext = submission.selftext or ""
                if not player_mentioned(f"{title} {selftext}", name):
                    continue

                post_created = datetime.fromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S")
                post_url = f"https://reddit.com{submission.permalink}"
                post_author = str(submission.author) if submission.author else "[deleted]"

                base = {
                    "player_name": name,
                    "player_id": pid,
                    "search_query": query,
                    "subreddit": sub_name,
                    "post_id": submission.id,
                    "post_title": title,
                    "post_author": post_author,
                    "post_score": submission.score,
                    "post_upvote_ratio": submission.upvote_ratio,
                    "post_created": post_created,
                    "post_url": post_url,
                    "post_num_comments": submission.num_comments,
                }

                rows.append({
                    **base,
                    "type": "post",
                    "post_text": selftext,
                    "comment_id": "",
                    "comment_author": "",
                    "comment_body": "",
                    "comment_score": "",
                    "comment_created": "",
                    "comment_parent_id": "",
                })

                for c in fetch_comments(submission):
                    if player_mentioned(c["comment_body"], name):
                        rows.append({
                            **base,
                            "type": "comment",
                            "post_text": "",
                            **c,
                        })

    return rows


def main():
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_SECRET"),
        user_agent="football_player_research/1.0",
        # Let PRAW sleep up to 5 min when rate-limited rather than raising
        ratelimit_seconds=300,
    )

    players = load_players()
    progress = load_progress()
    ensure_csv_header()

    def run_pass(pass_num: int, skip_match_threads: bool):
        key_completed = "completed_player_ids" if pass_num == 1 else "completed_player_ids_pass2"
        completed = set(progress.get(key_completed, []))
        total_rows = progress.get("total_rows", 0)
        global_seen = load_existing_post_ids() if pass_num == 2 else None

        remaining = [p for p in players if p["id"] not in completed]
        label = f"Pass {pass_num}" + (" (no match threads)" if skip_match_threads else "")
        print(f"\n{'=' * 60}")
        print(f"All Leagues Reddit Scraper — {label}")
        print(f"  Total players : {len(players)}")
        print(f"  Already done  : {len(completed)}")
        print(f"  Remaining     : {len(remaining)}")
        print(f"  Workers       : {NUM_WORKERS}")
        print(f"  Subreddits    : {', '.join(SUBREDDITS)}")
        print(f"  Output        : {OUTPUT_CSV}")
        print("=" * 60)

        if not remaining:
            print("  Nothing to do.")
            return

        done_count = 0
        unsaved_count = 0

        try:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {
                    executor.submit(scrape_player, reddit, p, skip_match_threads, global_seen): p
                    for p in remaining
                }

                for future in as_completed(futures):
                    player = futures[future]
                    done_count += 1

                    try:
                        rows = future.result()
                    except Exception as e:
                        print(f"[{done_count}/{len(remaining)}] [error] {player['name']}: {e}")
                        rows = []

                    if rows:
                        append_rows(rows)
                        total_rows += len(rows)
                        if global_seen is not None:
                            global_seen.update(r["post_id"] for r in rows if r.get("post_id"))
                        posts = sum(1 for r in rows if r["type"] == "post")
                        comments = sum(1 for r in rows if r["type"] == "comment")
                        print(f"[{done_count}/{len(remaining)}] {player['name']} -> {posts} posts, {comments} comments (total: {total_rows})")
                    else:
                        print(f"[{done_count}/{len(remaining)}] {player['name']} -> no results")

                    completed.add(player["id"])
                    unsaved_count += 1

                    if unsaved_count >= SAVE_INTERVAL:
                        progress[key_completed] = sorted(completed)
                        progress["total_rows"] = total_rows
                        save_progress(progress)
                        unsaved_count = 0

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            progress[key_completed] = sorted(completed)
            progress["total_rows"] = total_rows
            save_progress(progress)
            print(f"\nPass {pass_num} done! {total_rows} total rows saved to {OUTPUT_CSV}")

    run_pass(1, skip_match_threads=False)
    run_pass(2, skip_match_threads=True)


if __name__ == "__main__":
    main()
