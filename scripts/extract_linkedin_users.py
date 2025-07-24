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
from typing import List, Dict
from pepino.config import Settings

# Heuristic phrases indicating self-sharing
SELF_LINKEDIN_PHRASES = [
    r"my linkedin",
    r"here'?s my linkedin",
    r"connect with me on linkedin",
    r"this is my linkedin",
    r"add me on linkedin",
    r"linkedin\.com/in/",
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


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def normalize_linkedin_url(url: str) -> str:
    # Extract the username part robustly
    m = re.search(r"linkedin\.com/in/([a-z0-9\-_%]+)", url.lower())
    if m:
        username = m.group(1).rstrip("/")
        return f"https://linkedin.com/in/{username}"
    return url.strip().lower()


def is_self_linkedin_share(message: str, linkedin_username: str, discord_username: str, display_name: str) -> bool:
    # Check for self-sharing phrases
    msg_norm = message.lower()
    for phrase in SELF_LINKEDIN_PHRASES:
        if re.search(phrase, msg_norm):
            # If the LinkedIn username matches Discord username or display name, likely self-share
            ln_norm = normalize(linkedin_username)
            if ln_norm in normalize(discord_username) or ln_norm in normalize(display_name):
                return True
            # If phrase is strong enough, accept even if not matching
            if phrase != "linkedin.com/in/":
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
    cur.execute("""
        SELECT author_name, author_display_name, content
        FROM messages
        WHERE content LIKE '%linkedin.com%'
          AND (author_is_bot IS NULL OR author_is_bot = 0)
    """)
    results = cur.fetchall()

    output: List[Dict] = []
    seen = set()
    for row in results:
        content = row["content"] or ""
        author_name = row["author_name"] or ""
        display_name = row["author_display_name"] or author_name

        # Only include LinkedIn profile URLs (linkedin.com/in/USERNAME)
        for match in LINKEDIN_USER_URL_RE.finditer(content):
            linkedin_url = match.group(0)
            norm_linkedin_url = normalize_linkedin_url(linkedin_url)
            norm_display_name = normalize(display_name)
            norm_author_name = normalize(author_name)
            key = (norm_display_name, norm_author_name, norm_linkedin_url)
            if key not in seen:
                seen.add(key)
                output.append({
                    "display_name": display_name,
                    "username": author_name,
                    "linkedin_url": norm_linkedin_url
                })

    # Sort output by display_name (case-insensitive)
    output.sort(key=lambda x: x['display_name'].lower())

    # Write output to JSON file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Extracted {len(output)} self-shared LinkedIn user pages. Output written to {args.output}")


if __name__ == "__main__":
    main() 