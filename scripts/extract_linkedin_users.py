#!/usr/bin/env python3
"""
Extract Discord messages where users share their own LinkedIn user page.
Outputs a JSON file with display name, username, and LinkedIn URL.
"""
import sys
import json
import os
import re
import sqlite3
from typing import Dict, List, Tuple

from pepino.config import Settings

# Heuristic phrases indicating self-sharing
SELF_LINKEDIN_PHRASES = [
    r"\bmy\s+linkedin\b",
    r"\bhere'?s\s+my\s+linkedin\b",
    r"\bconnect\s+with\s+me\s+on\s+linkedin\b",
    r"\bthis\s+is\s+my\s+linkedin\b",
    r"\badd\s+me\s+on\s+linkedin\b",
    r"\bthat'?s\s+me\s+on\s+linkedin\b",
    r"\blinkedin\.com/in/",
]

# Exclude these LinkedIn URL patterns (not user pages)
EXCLUDE_PATTERNS = [
    r"linkedin\.com/company/",
    r"linkedin\.com/jobs/",
    r"linkedin\.com/pulse/",
    r"linkedin\.com/school/",
    r"linkedin\.com/groups/",
    r"linkedin\.com/feed/",
    r"linkedin\.com/events/",
]

# Update the regex to match all variants
LINKEDIN_USER_URL_RE = re.compile(
    r"(https?://)?(www\.)?linkedin\.com/in/([a-zA-Z0-9\-_%]+)", re.IGNORECASE
)

SELF_LINKEDIN_REGEXES = [re.compile(p, re.IGNORECASE) for p in SELF_LINKEDIN_PHRASES]
EXCLUDE_REGEXES = [re.compile(p, re.IGNORECASE) for p in EXCLUDE_PATTERNS]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def normalize_linkedin_url(url: str) -> Tuple[str, str]:
    # Extract the username part robustly
    match = re.search(r"linkedin\.com/in/([a-z0-9\-_%]+)", url.lower())
    if match:
        username = match.group(1).rstrip("/").split("?")[0]
        clean_username = username.split("/")[0]
        return f"https://linkedin.com/in/{clean_username}", clean_username
    normalized = url.strip().lower()
    return normalized, ""


def name_matches_linkedin_username(
    linkedin_username: str,
    *names: str,
) -> bool:
    if not linkedin_username:
        return False
    ln_norm = normalize(linkedin_username)
    if not ln_norm:
        return False

    for raw_name in names:
        if not raw_name:
            continue
        name_norm = normalize(raw_name)
        if ln_norm == name_norm:
            return True
        if ln_norm in name_norm or name_norm in ln_norm:
            return True
    return False


def contains_self_share_phrase(message: str) -> bool:
    return any(regex.search(message) for regex in SELF_LINKEDIN_REGEXES)


def contains_excluded_pattern(message: str) -> bool:
    return any(regex.search(message) for regex in EXCLUDE_REGEXES)


def is_self_linkedin_share(
    message: str,
    linkedin_username: str,
    discord_username: str,
    display_name: str,
) -> bool:
    msg_norm = message.lower()

    if name_matches_linkedin_username(linkedin_username, discord_username, display_name):
        return True

    if contains_self_share_phrase(msg_norm):
        return True

    # Additional heuristic: first-person pronouns near LinkedIn references
    if re.search(r"\b(my|me|mine)\b[^.]{0,80}\blinkedin", msg_norm):
        return True

    return False


def main():
    # Use default paths
    settings = Settings()
    default_db = settings.db_path if hasattr(settings, 'db_path') else 'data/discord_messages.db'
    default_output = 'linkedin_users.json'

    import argparse
    parser = argparse.ArgumentParser(description="Extract self-shared LinkedIn user pages from Discord messages.")
    parser.add_argument("--db", default=default_db, help="Path to the SQLite database (default: from config).")
    parser.add_argument("--output", default=default_output, help="Path to output JSON file (default: linkedin_users.json).")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Query all messages containing linkedin.com (broader catch, all channels)
    cur.execute(
        """
        SELECT author_name, author_display_name, content
        FROM messages
        WHERE content LIKE '%linkedin.com%'
          AND (author_is_bot IS NULL OR author_is_bot = 0)
    """
    )
    results = cur.fetchall()

    output: List[Dict] = []
    seen = set()
    for row in results:
        content = row["content"] or ""
        author_name = row["author_name"] or ""
        display_name = row["author_display_name"] or author_name

        if contains_excluded_pattern(content):
            continue

        # Only include LinkedIn profile URLs (linkedin.com/in/USERNAME)
        for match in LINKEDIN_USER_URL_RE.finditer(content):
            linkedin_url = match.group(0)
            normalized_url, linkedin_username = normalize_linkedin_url(linkedin_url)
            norm_display_name = normalize(display_name)
            norm_author_name = normalize(author_name)
            key = (norm_display_name, norm_author_name, normalized_url)

            if not is_self_linkedin_share(
                content,
                linkedin_username,
                author_name,
                display_name,
            ):
                continue

            if key not in seen:
                seen.add(key)
                output.append(
                    {
                        "display_name": display_name,
                        "username": author_name,
                        "linkedin_url": normalized_url,
                    }
                )

    # Sort output by display_name (case-insensitive)
    output.sort(key=lambda x: x['display_name'].lower())

    # Write output to JSON file
    result = {
        "total_users": len(output),
        "users": output,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"Extracted {result['total_users']} self-shared LinkedIn user pages. Output written to {args.output}"
    )


if __name__ == "__main__":
    main() 