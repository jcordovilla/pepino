"""
Utility script to generate narrative user summaries using a local Ollama model.

This script consumes the aggregated JSON produced by `pepino analyze users --all`,
selects the top N non-bot users (default: 100), performs lightweight NLP over
their messages stored in the SQLite database, and prompts a local LLM to craft
concise (<120 words) narratives describing each user's participation patterns.

Usage examples:
    poetry run python scripts/summarize_users.py \
        --input stats/user_analysis_all_20251110-232006.json \
        --output stats/user_summaries.json

    poetry run python scripts/summarize_users.py \
        --input stats/user_analysis_all_20251110-232006.json \
        --users alex jane

The script expects an Ollama-compatible endpoint (defaults to
http://localhost:11434). Configure via OLLAMA_HOST / OLLAMA_URL if needed.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import click
import requests

from pepino.config import Settings


DEFAULT_MODEL = "qwen3-coder"
DEFAULT_MAX_WORDS = 120
DEFAULT_LIMIT = 100
MESSAGE_SAMPLE_LIMIT = 500
LONG_MESSAGE_SAMPLE = 20
LONG_MESSAGE_MIN_CHARS = 280

SYSTEM_PROMPT = (
    "You are an expert community analyst. Write compact prose summaries (under {max_words} words) "
    "that describe a Discord member's focus areas, tone, and engagement. Use concrete evidence "
    "(message volumes, activity spans, topics, style, recommendations, key channels) when available. Avoid bullet "
    "points. Do not speculate or fabricate. Be critical and objective."
)

STOP_WORDS = {
    "the",
    "and",
    "that",
    "with",
    "this",
    "have",
    "from",
    "were",
    "about",
    "https",
    "http",
    "com",
    "www",
    "discord",
    "like",
    "just",
    "they",
    "there",
    "their",
    "also",
    "been",
    "what",
    "when",
    "your",
    "into",
    "will",
    "would",
    "could",
    "should",
    "them",
    "than",
    "then",
    "while",
    "where",
    "who",
    "whom",
    "dont",
    "cant",
    "didnt",
    "doesnt",
    "isnt",
    "theres",
    "its",
    "im",
    "ive",
    "ill",
    "aint",
    "youre",
    "youve",
    "youll",
    "hes",
    "shes",
    "were",
    "well",
    "theyre",
    "theyve",
    "theyll",
    "lets",
    "thanks",
    "thank",
    "appreciate",
}


def _load_user_entries(input_path: Path) -> List[Dict]:
    """Load aggregated user analysis JSON entries."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    users = payload.get("users", [])
    filtered: List[Dict] = []
    for entry in users:
        if entry.get("is_bot"):
            continue
        statistics = entry.get("statistics") or {}
        entry["_message_count"] = (
            statistics.get("message_count")
            or entry.get("metadata", {}).get("total_messages")
            or 0
        )
        entry["_channels_active"] = (
            statistics.get("channels_active")
            or entry.get("metadata", {}).get("unique_channels")
            or 0
        )
        entry["_display_name"] = (
            entry.get("display_name")
            or entry.get("username")
            or statistics.get("author_name")
        )
        entry["_user_id"] = entry.get("user_id")
        entry["_username"] = entry.get("username") or statistics.get("author_name")
        filtered.append(entry)

    return filtered


def _select_users(
    users: Sequence[Dict],
    requested: Optional[Iterable[str]],
    limit: int,
) -> List[Dict]:
    """Select which users to summarize."""
    if requested:
        requested_set = {r.lower() for r in requested}
        selected = [
            entry
            for entry in users
            if entry.get("_username", "").lower() in requested_set
            or (entry.get("_display_name") or "").lower() in requested_set
            or (entry.get("_user_id") or "").lower() in requested_set
        ]
        missing = requested_set.difference(
            {
                entry.get("_username", "").lower()
                for entry in selected
                if entry.get("_username")
            }
            | {
                (entry.get("_display_name") or "").lower()
                for entry in selected
                if entry.get("_display_name")
            }
            | {
                (entry.get("_user_id") or "").lower()
                for entry in selected
                if entry.get("_user_id")
            }
        )
        if missing:
            missing_display = ", ".join(sorted(missing))
            raise ValueError(f"Requested users not found in analysis JSON: {missing_display}")
        return selected

    sorted_users = sorted(users, key=lambda e: e.get("_message_count", 0), reverse=True)
    return sorted_users[:limit]


def _get_db_connection() -> sqlite3.Connection:
    """Obtain a SQLite connection using configured settings."""
    try:
        settings = Settings()
        db_path = settings.db_path
    except Exception:
        db_path = "data/discord_messages.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_user_messages(
    conn: sqlite3.Connection,
    author_id: Optional[str],
    username: Optional[str],
    limit: int = MESSAGE_SAMPLE_LIMIT,
) -> List[str]:
    """Fetch recent message contents for a user."""
    conditions = ["content IS NOT NULL", "TRIM(content) != ''"]
    params: List[str] = []

    if author_id:
        conditions.append("author_id = ?")
        params.append(author_id)
    if username:
        conditions.append("author_name = ?")
        params.append(username)

    if len(params) == 0:
        return []

    where_clause = " OR ".join(conditions[2:])
    query = f"""
        SELECT content
        FROM messages
        WHERE ({where_clause})
          AND content IS NOT NULL
          AND TRIM(content) != ''
        ORDER BY timestamp DESC
        LIMIT ?
    """
    rows = conn.execute(query, (*params[0: len(params)], limit)).fetchall()
    return [row["content"] for row in rows if row["content"]]


def _fetch_long_messages(
    conn: sqlite3.Connection,
    author_id: Optional[str],
    username: Optional[str],
    limit: int = LONG_MESSAGE_SAMPLE,
    min_chars: int = LONG_MESSAGE_MIN_CHARS,
) -> List[str]:
    """Fetch a sample of longer-form messages for context."""
    identifier_clauses = []
    params: List[str] = []

    if author_id:
        identifier_clauses.append("author_id = ?")
        params.append(author_id)
    if username:
        identifier_clauses.append("author_name = ?")
        params.append(username)

    if not identifier_clauses:
        return []

    where_identifier = " OR ".join(identifier_clauses)
    query = f"""
        SELECT content
        FROM messages
        WHERE ({where_identifier})
          AND content IS NOT NULL
          AND TRIM(content) != ''
          AND LENGTH(content) >= ?
        ORDER BY LENGTH(content) DESC, timestamp DESC
        LIMIT ?
    """
    rows = conn.execute(query, (*params, min_chars, limit)).fetchall()
    return [row["content"] for row in rows if row["content"]]


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for English text."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", text.lower())
    return [token for token in tokens if token not in STOP_WORDS]


def _analyze_messages(messages: Sequence[str]) -> Dict[str, object]:
    """Compute lightweight NLP stats from a set of messages."""
    if not messages:
        return {
            "total_messages_analyzed": 0,
            "average_length": 0,
            "top_words": [],
            "top_bigrams": [],
        }

    all_tokens: List[str] = []
    total_length = 0
    for message in messages:
        tokens = _tokenize(message)
        if tokens:
            all_tokens.extend(tokens)
        total_length += len(message)

    token_counts = Counter(all_tokens)
    top_words = token_counts.most_common(15)

    bigrams = Counter(zip(all_tokens, all_tokens[1:]))
    top_bigrams = [
        (" ".join(pair), count) for pair, count in bigrams.most_common(10) if count > 1
    ]

    average_length = total_length / len(messages)

    return {
        "total_messages_analyzed": len(messages),
        "average_length": round(average_length, 1),
        "top_words": top_words,
        "top_bigrams": top_bigrams,
    }


def _prepare_user_context(
    entry: Dict,
    nlp_summary: Dict[str, object],
    long_messages: Sequence[str],
) -> str:
    """Build a context string for the LLM summarization."""
    lines: List[str] = []
    display_name = entry.get("_display_name") or entry.get("_username")
    username = entry.get("_username")
    lines.append(f"User: {display_name}")
    if username and username != display_name:
        lines.append(f"Handle: {username}")
    user_id = entry.get("_user_id")
    if user_id:
        lines.append(f"User ID: {user_id}")

    statistics = entry.get("statistics") or {}
    lines.append("Activity Metrics:")
    metrics_to_show = [
        ("Messages", statistics.get("message_count") or entry.get("_message_count")),
        ("Active Channels", statistics.get("channels_active") or entry.get("_channels_active")),
        ("Active Days", statistics.get("active_days")),
        ("Avg Message Length", statistics.get("avg_message_length")),
        ("First Message", statistics.get("first_message_date") or entry.get("first_message_at")),
        ("Last Message", statistics.get("last_message_date") or entry.get("last_message_at")),
    ]
    for label, value in metrics_to_show:
        if value is not None:
            lines.append(f"  {label}: {value}")

    channel_activity = entry.get("channel_activity") or []
    if channel_activity:
        lines.append("Top Channels:")
        for item in channel_activity[:5]:
            channel_name = item.get("channel_name")
            message_count = item.get("message_count")
            if channel_name and message_count is not None:
                lines.append(f"  {channel_name}: {message_count} msgs")

    top_topics = entry.get("top_topics")
    if top_topics:
        lines.append("Model Topics:")
        for topic in top_topics[:5]:
            if isinstance(topic, dict):
                label = topic.get("topic") or topic.get("name")
                frequency = topic.get("frequency") or topic.get("message_count")
                if label:
                    if frequency is not None:
                        lines.append(f"  {label} ({frequency})")
                    else:
                        lines.append(f"  {label}")
            elif isinstance(topic, str):
                lines.append(f"  {topic}")

    lines.append("Message Highlights:")
    lines.append(f"  Sampled Messages: {nlp_summary['total_messages_analyzed']}")
    lines.append(f"  Average Length: {nlp_summary['average_length']} characters")

    if long_messages:
        lines.append("  Long-form Samples:")
        for idx, excerpt in enumerate(long_messages[:3], start=1):
            snippet = excerpt.strip()
            if len(snippet) > 320:
                snippet = snippet[:317].rstrip() + "..."
            lines.append(f"    [{idx}] {snippet}")

    return "\n".join(lines)


def _call_llm(
    model: str,
    context: str,
    base_url: str,
    max_words: int,
) -> str:
    """Invoke the local LLM to produce a summary."""
    prompt = (
        SYSTEM_PROMPT.format(max_words=max_words)
        + "\n\n"
        + f"Summarize the following Discord user analytics in under {max_words} words.\n"
        "Emphasize participation style, recurring themes, and observed impact.\n"
        "Analytics:\n"
        f"{context}"
    )

    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
        },
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    summary = data.get("response", "")
    return summary.strip()


def _normalize_keywords(counter_data: Sequence[Tuple[str, int]]) -> List[Dict[str, object]]:
    """Format keyword counters into serializable dictionaries."""
    results: List[Dict[str, object]] = []
    for item in counter_data:
        if isinstance(item, tuple) and len(item) == 2:
            results.append({"token": item[0], "count": item[1]})
    return results


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, readable=True),
    required=True,
    help="Path to the aggregated user analysis JSON file.",
)
@click.option(
    "--users",
    multiple=True,
    help="Specific usernames/display names/user_ids to summarize (case-insensitive).",
)
@click.option(
    "--limit",
    type=int,
    default=DEFAULT_LIMIT,
    show_default=True,
    help="Maximum number of top users to summarize (ignored when --users is provided).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, writable=True),
    default=None,
    help="Optional path to save the summaries as JSON.",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="LLM model name to use.",
)
@click.option(
    "--max-words",
    type=int,
    default=DEFAULT_MAX_WORDS,
    show_default=True,
    help="Maximum number of words allowed in each summary.",
)
def main(
    input_path: Path,
    users: Iterable[str],
    limit: int,
    output_path: Optional[Path],
    model: str,
    max_words: int,
) -> None:
    """Generate narrative user summaries using a local LLM."""
    base_url = (
        os.getenv("OLLAMA_HOST")
        or os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_URL")
        or "http://localhost:11434"
    )

    all_entries = _load_user_entries(input_path)
    if not all_entries:
        raise ValueError("No user entries found in the aggregated analysis JSON.")

    selected_entries = _select_users(all_entries, users, limit)
    conn = _get_db_connection()

    summaries: List[Dict[str, object]] = []
    progress_label = f"Summarizing {len(selected_entries)} users"

    with click.progressbar(
        selected_entries,
        label=progress_label,
        length=len(selected_entries),
    ) as progress_iterable:
        for entry in progress_iterable:
            author_id = entry.get("_user_id")
            username = entry.get("_username")
            messages = _fetch_user_messages(conn, author_id, username, limit=MESSAGE_SAMPLE_LIMIT)
            nlp_summary = _analyze_messages(messages)
            long_messages = _fetch_long_messages(conn, author_id, username)

            context = _prepare_user_context(entry, nlp_summary, long_messages)
            summary_text = _call_llm(model, context, base_url, max_words=max_words)

            payload = {
                "user_id": author_id,
                "username": username,
                "display_name": entry.get("_display_name"),
                "summary": summary_text,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "statistics": {
                    "message_count": entry.get("_message_count"),
                    "channels_active": entry.get("_channels_active"),
                },
                "messages_analyzed": nlp_summary.get("total_messages_analyzed", 0),
            }

            if long_messages:
                payload["long_message_samples"] = long_messages[:LONG_MESSAGE_SAMPLE]

            summaries.append(payload)

    conn.close()

    output_payload = {"summaries": summaries}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)
        click.echo(f"Summaries written to {output_path}")
    else:
        click.echo(json.dumps(output_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


