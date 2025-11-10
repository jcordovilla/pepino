"""
Utility script to generate narrative channel summaries using a local Ollama model.

Given a JSON export produced by `pepino analyze channels --all`, this script:
- Extracts key analytics fields for the requested channels.
- Calls a local LLM (default: deepseek-r1:8b) to produce a <100-word narrative summary
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
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import click

import requests


DEFAULT_MODEL = "deepseek-r1:8b"
DEFAULT_MAX_WORDS = 100
SYSTEM_PROMPT = (
    "You are an expert community analyst. Craft concise (under 100 words) narrative "
    "summaries describing the purpose, recurring content, and engagement level of "
    "Discord channels. Use an informative but friendly tone. Avoid bullet points. "
    "Mention concrete signals (message counts, top contributors, recent activity, "
    "engagement stats) when available. Do not fabricate data."
)


def _load_channel_data(input_path: Path) -> Dict[str, Dict]:
    """Load the aggregated channel analysis JSON and index by channel name."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    channels = payload.get("channels", [])
    indexed = {}
    for entry in channels:
        name = entry.get("channel_name") or entry.get("full_analysis", {}).get("channel_info", {}).get("channel_name")
        if not name:
            continue
        indexed[name] = entry
    return indexed


def _prepare_channel_context(channel_name: str, channel_entry: Dict) -> str:
    """Build a context string summarizing the relevant analytics for LLM input."""
    analysis = channel_entry.get("full_analysis", {})

    statistics = analysis.get("statistics")
    engagement_metrics = analysis.get("engagement_metrics")
    health_metrics = analysis.get("health_metrics")
    engagement_summary = analysis.get("engagement_summary") or channel_entry.get("summaries", {}).get("engagement")
    content_analysis = channel_entry.get("content_analysis") or {
        "top_topics": analysis.get("top_topics"),
    }

    lines: List[str] = [f"Channel: {channel_name}"]

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
    if requested:
        missing = [name for name in requested if name not in available]
        if missing:
            raise ValueError(f"Requested channels not present in analysis JSON: {', '.join(missing)}")
        return list(dict.fromkeys(requested))

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
    for channel_name in selected_channels:
        entry = channel_data[channel_name]
        context = _prepare_channel_context(channel_name, entry)
        summary = _call_llm(model, channel_name, context, base_url, max_words=max_words)

        summaries.append(
            {
                "channel_name": channel_name,
                "summary": summary,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
            }
        )

        click.echo(f"\n# {channel_name}\n{summary}\n")

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

