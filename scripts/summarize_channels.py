"""
Utility script to generate narrative channel summaries using a local Ollama model.

Given a JSON export produced by `pepino analyze channels --all`, this script:
- Extracts key analytics fields for the requested channels.
- Calls a local LLM (default: qwen3-coder) to produce a <100-word narrative summary
  describing channel purpose, content themes, and engagement.
- Writes the summaries to stdout and optionally to a JSON file.

Usage examples:
    poetry run python scripts/summarize_channels.py --input stats/channel_analysis_all_20251110-162929.json \\
        --channels general announcements

    poetry run python scripts/summarize_channels.py --input stats/channel_analysis_all_20251110-162929.json \\
        --sample 3 --output stats/channel_summaries.json
"""

from __future__ import annotations

import json
import sqlite3
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import click

import requests

from pepino.config import Settings


DEFAULT_MODEL = "qwen3-coder"
DEFAULT_MAX_WORDS = 100
SYSTEM_PROMPT = (
    "You are an expert community analyst. Craft concise (under 100 words) narrative "
    "summaries describing the purpose, recurring content, and engagement level of "
    "Discord channels. Use an informative but friendly tone. Avoid bullet points. "
    "Mention concrete signals (message counts, top contributors, recent activity, "
    "engagement stats) when available. Do not fabricate data."
)


def _resolve_parent_names(parent_ids: Set[str]) -> Dict[str, str]:
    """Resolve parent IDs to names using the channels table."""
    if not parent_ids:
        return {}

    try:
        settings = Settings()
        db_path = settings.db_path
    except Exception:
        db_path = "data/discord_messages.db"

    if not Path(db_path).exists():
        return {}

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in parent_ids)
        query = f"""
            SELECT channel_id, COALESCE(display_name, channel_name) AS name
            FROM channels
            WHERE channel_id IN ({placeholders})
        """
        rows = conn.execute(query, tuple(parent_ids)).fetchall()
        return {
            row["channel_id"]: row["name"]
            for row in rows
            if row["channel_id"] and row["name"]
        }
    except Exception:
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _load_channel_data(input_path: Path) -> Dict[str, Dict]:
    """Load the aggregated channel analysis JSON and index by channel name."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    channels = payload.get("channels", [])
    indexed: Dict[str, Dict] = {}
    missing_parent_ids: Set[str] = set()
    for entry in channels:
        metadata = entry.get("metadata") or {}
        entry.setdefault("metadata", metadata)

        channel_id = metadata.get("channel_id") or entry.get("channel_id")
        channel_name = (
            metadata.get("channel_name")
            or entry.get("channel_name")
            or entry.get("full_analysis", {}).get("channel_info", {}).get("channel_name")
        )
        parent_id = (
            metadata.get("parent_id")
            or entry.get("full_analysis", {}).get("channel_info", {}).get("parent_id")
        )
        parent_name = (
            metadata.get("parent_name")
            or entry.get("full_analysis", {}).get("channel_info", {}).get("parent_name")
        )

        if channel_name and not metadata.get("channel_name"):
            metadata["channel_name"] = channel_name
        if channel_id and not metadata.get("channel_id"):
            metadata["channel_id"] = channel_id
        if parent_id and not metadata.get("parent_id"):
            metadata["parent_id"] = parent_id
        if parent_name and not metadata.get("parent_name"):
            metadata["parent_name"] = parent_name

        key = channel_id or channel_name
        if not key:
            continue

        aliases: Set[str] = set()
        if channel_name:
            aliases.add(channel_name)
        if channel_id:
            aliases.add(channel_id)
        if parent_id and channel_name:
            aliases.add(f"{parent_id}:{channel_name}")
        if parent_name and channel_name:
            aliases.add(f"{parent_name}:{channel_name}")

        entry["_key"] = key
        entry["_parent_id"] = parent_id
        entry["_parent_name"] = parent_name
        entry["_aliases"] = sorted(alias for alias in aliases if alias)
        if parent_id and not parent_name:
            missing_parent_ids.add(parent_id)
        indexed[key] = entry

    if missing_parent_ids:
        parent_name_map = _resolve_parent_names(missing_parent_ids)
        if parent_name_map:
            for entry in indexed.values():
                pid = entry.get("_parent_id")
                if pid and not entry.get("_parent_name"):
                    resolved = parent_name_map.get(pid)
                    if resolved:
                        entry["_parent_name"] = resolved
                        entry.setdefault("metadata", {})["parent_name"] = resolved
    return indexed


def _prepare_channel_context(channel_name: str, channel_entry: Dict) -> str:
    """Build a context string summarizing the relevant analytics for LLM input."""
    analysis = channel_entry.get("full_analysis", {})
    metadata = channel_entry.get("metadata") or {}

    statistics = analysis.get("statistics")
    engagement_metrics = analysis.get("engagement_metrics")
    health_metrics = analysis.get("health_metrics")
    engagement_summary = analysis.get("engagement_summary") or channel_entry.get("summaries", {}).get("engagement")
    content_analysis = channel_entry.get("content_analysis") or {
        "top_topics": analysis.get("top_topics"),
    }

    lines: List[str] = [f"Channel: {channel_name}"]
    channel_id = metadata.get("channel_id") or channel_entry.get("channel_id")
    if channel_id:
        lines.append(f"Channel ID: {channel_id}")
    parent_id = metadata.get("parent_id") or channel_entry.get("_parent_id")
    parent_name = metadata.get("parent_name") or channel_entry.get("_parent_name")
    if parent_id:
        descriptor = parent_name or parent_id
        lines.append(f"Parent: {descriptor} (ID: {parent_id})")

    if statistics:
        lines.append("Statistics:")
        for key in (
            "total_messages",
            "unique_users",
            "unique_human_users",
            "avg_message_length",
            "human_messages",
            "bot_messages",
            "first_message",
            "last_message",
        ):
            if statistics.get(key) is not None:
                lines.append(f"  {key}: {statistics.get(key)}")

    if engagement_metrics:
        lines.append("Engagement Metrics:")
        for key, value in engagement_metrics.items():
            lines.append(f"  {key}: {value}")

    if health_metrics:
        lines.append("Health Metrics:")
        for key, value in health_metrics.items():
            lines.append(f"  {key}: {value}")

    if engagement_summary:
        lines.append(f"Engagement Summary: {engagement_summary}")

    if content_analysis:
        lines.append("Content Analysis:")
        top_topics = content_analysis.get("top_topics")
        if top_topics:
            if isinstance(top_topics, list):
                topic_strings = []
                for topic in top_topics:
                    if isinstance(topic, dict):
                        label = topic.get("topic") or topic.get("name") or topic.get("label")
                        freq = topic.get("frequency") or topic.get("message_count")
                        if label:
                            if freq is not None:
                                topic_strings.append(f"{label} ({freq})")
                            else:
                                topic_strings.append(label)
                    elif isinstance(topic, str):
                        topic_strings.append(topic)
                if topic_strings:
                    lines.append(f"  top_topics: {', '.join(topic_strings)}")
            else:
                lines.append(f"  top_topics: {top_topics}")

        clusters = content_analysis.get("content_clusters")
        if clusters:
            lines.append(f"  content_clusters: {clusters}")

        keywords = content_analysis.get("keywords")
        if keywords:
            lines.append(f"  keywords: {keywords}")

    return "\n".join(lines)


def _call_llm(
    model: str,
    channel_name: str,
    context: str,
    base_url: str,
    max_words: int = DEFAULT_MAX_WORDS,
) -> str:
    """Invoke the LLM and return the generated summary."""
    prompt = (
        f"Summarize the following Discord channel analytics in less than {max_words} words.\n"
        "Focus on the channel's purpose, recurring content, and engagement level.\n"
        "Channel analytics:\n"
        f"{context}"
    )

    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
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


def _select_channels(
    requested: Optional[Iterable[str]],
    sample: Optional[int],
    available: Dict[str, Dict],
) -> List[str]:
    """Determine which channels to summarize."""
    alias_map: Dict[str, Set[str]] = {}
    for key, entry in available.items():
        aliases = set(entry.get("_aliases", []))
        aliases.add(key)
        for alias in aliases:
            alias_map.setdefault(alias, set()).add(key)

    if requested:
        resolved_keys: List[str] = []
        missing: List[str] = []
        for name in requested:
            matches = alias_map.get(name)
            if not matches:
                missing.append(name)
                continue
            if len(matches) > 1:
                raise ValueError(
                    f"Channel '{name}' is ambiguous. "
                    f"Please specify the channel_id instead (options: {', '.join(sorted(matches))})."
                )
            resolved_keys.append(next(iter(matches)))
        if missing:
            raise ValueError(f"Requested channels not present in analysis JSON: {', '.join(missing)}")
        return list(dict.fromkeys(resolved_keys))

    names = list(available.keys())
    if sample is not None:
        if sample <= 0:
            raise ValueError("--sample must be a positive integer")
        if sample > len(names):
            sample = len(names)
        return random.sample(names, sample)

    return names


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, readable=True),
    required=True,
    help="Path to the aggregated channel analysis JSON file.",
)
@click.option(
    "--channels",
    multiple=True,
    help="Specific channel names to summarize. Provide multiple times for more than one channel.",
)
@click.option(
    "--sample",
    type=int,
    default=None,
    help="Randomly summarize N channels (ignored if --channels is provided).",
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
    channels: Iterable[str],
    sample: Optional[int],
    output_path: Optional[Path],
    model: str,
    max_words: int,
) -> None:
    """Generate narrative channel summaries using GPT5 nano."""
    base_url = (
        os.getenv("OLLAMA_HOST")
        or os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_URL")
        or "http://localhost:11434"
    )

    channel_data = _load_channel_data(input_path)
    selected_channels = _select_channels(channels, sample, channel_data)

    summaries = []
    for channel_key in selected_channels:
        entry = channel_data[channel_key]
        metadata = entry.get("metadata") or {}
        display_name = metadata.get("channel_name") or entry.get("channel_name") or channel_key
        parent_id = metadata.get("parent_id") or entry.get("_parent_id")
        parent_name = metadata.get("parent_name") or entry.get("_parent_name")
        if parent_name or parent_id:
            descriptor = parent_name or parent_id
            header_name = f"{descriptor} / {display_name}"
        else:
            header_name = display_name

        context = _prepare_channel_context(display_name, entry)
        summary = _call_llm(model, display_name, context, base_url, max_words=max_words)

        summaries.append(
            {
                "channel_key": channel_key,
                "channel_id": metadata.get("channel_id"),
                "channel_name": display_name,
                "parent_id": parent_id,
                "parent_name": parent_name,
                "summary": summary,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
            }
        )

        click.echo(f"\n# {header_name}\n{summary}\n")

    if output_path:
        target_path = output_path
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        target_path = Path("stats") / f"channel_summaries_{timestamp}.json"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        json.dump({"summaries": summaries}, f, indent=2)
    click.echo(f"Summaries written to {target_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

