# Pepino – Advanced Discord Analytics

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue.svg)](https://python-poetry.org/)

Pepino turns Discord server history into clear insights for community managers, product teams, and researchers. The project ships with both a friendly Discord bot and a scriptable CLI so you can explore engagement, topics, and growth from wherever you work.

---

## Quick Start

**Install**
```bash
git clone https://github.com/your-repo/pepino.git
cd pepino
make dev-setup          # installs dependencies, dev tools, and pre-commit hooks
```

**Configure**
```bash
cp .env.example .env
# Edit .env and set at least:
DISCORD_TOKEN=your_bot_token
DATABASE_URL=sqlite:///data/discord_messages.db
```

**Run**
```bash
poetry run pepino start
```
In Discord, type `/pepino_server_analytics overview` to confirm the bot is responding.

---

## What You Get
- Understand members: activity patterns, peak times, top contributors, trending topics.
- Monitor channels: health scores, responsiveness, participation, sentiment-ready data.
- Compare timeframes: growth insights and engagement deltas across 7-day, 30-day, or custom windows.
- Export everywhere: JSON, CSV, text summaries, and charts ready for reports or dashboards.

---

## Interfaces

### Discord Bot
- Autocomplete for channels and users.
- Rich embeds with charts and key metrics.
- Friendly command groups (`pepino_channel_analytics`, `pepino_user_analytics`, `pepino_server_analytics`, `pepino_lists`).

```bash
/pepino_server_analytics overview
/pepino_channel_analytics overview channel_name:general
/pepino_user_analytics username:alice days:30
/pepino_lists users
```

### Command Line
- Ideal for automation, CI pipelines, and data exports.
- Mirrors the Discord bot command set with additional sync and test tooling.

```bash
poetry run pepino analyze users --limit 20 --format json
poetry run pepino analyze channels --channel general --days 14
poetry run pepino sync run
poetry run pepino list channels --format csv --output channels.csv
```

#### CLI Command Index

**Global flags**
- `--db-path PATH` set database location (default `data/discord_messages.db`)
- `--verbose / -v` enable verbose logging

**Help**
- `pepino help` shows the full CLI reference.

**Analyze commands (`pepino analyze ...`)**
- `users [--user NAME] [--limit INT=10] [--output PATH] [--format text|json|csv]`
- `channels [--channel NAME] [--limit INT=10] [--output PATH] [--format text|json|csv]`
- `topics [--channel NAME] [--topics INT=20] [--days INT] [--output PATH] [--format text|json|csv]`
- `temporal [--channel NAME] [--days INT] [--granularity hour|day|week=day] [--output PATH] [--format text|json|csv]`
- `conversations [--channel NAME] [--output PATH] [--format text|json|csv]`
- `similar --query TEXT [--limit INT=10] [--threshold FLOAT=0.5] [--output PATH] [--format text|json|csv]`
- `embeddings [--batch-size INT=100] [--output PATH] [--format text|json|csv]`
- `sentiment [--channel NAME] [--limit INT=100] [--output PATH] [--format text|json|csv]`
- `duplicates [--channel NAME] [--threshold FLOAT=0.9] [--output PATH] [--format text|json|csv]`

**Sync commands (`pepino sync ...`, admin recommended)**
- `run [--force] [--full] [--clear] [--timeout INT=300]`
- `status`

**List commands (`pepino list ...`)**
- `channels [--limit INT=0] [--output PATH] [--format text|json|csv]`
- `users [--limit INT=0] [--output PATH] [--format text|json|csv]`
- `stats [--output PATH] [--format text|json|csv]`

**Performance commands (`pepino performance ...`, admin recommended)**
- `metrics [--output PATH] [--format text|json|csv]`
- `benchmark [--operations NAME...] [--iterations INT=3] [--output PATH] [--format text|json|csv]`
- `profile --operation NAME [--args JSON] [--output PATH]`

**Test commands (`pepino test ...`, admin recommended)**
- `data [--output PATH] [--format text|json|csv]`
- `analysis [--operation NAME] [--sample-size INT=10] [--output PATH] [--format text|json|csv]`
- `templates [--template NAME] [--output PATH] [--format text|json|csv]`
- `dependencies [--output PATH] [--format text|json|csv]`

**Other commands**
- `pepino start [--token TOKEN] [--prefix STRING=! ] [--debug]`
- `pepino export-data [--table messages|users|channels] [--output PATH] [--format json|csv=csv]`

---

## Configure Essentials
- `DISCORD_TOKEN` – required Discord bot token.
- `DATABASE_URL` – defaults to SQLite (`sqlite:///data/discord_messages.db`), supports PostgreSQL.
- `LOG_LEVEL` – `INFO` by default; set to `DEBUG` for development.
- `MAX_MESSAGES`, `ANALYSIS_TIMEOUT`, and `ENABLE_*` flags let you scale analyses to match your server size.

See `docs/operations.md` for detailed environment, performance, and privacy options.

---

## Develop & Test
```bash
make dev          # format + lint + fast tests
make quality      # full lint, type-check, formatting
make test         # full test suite
```

Architecture diagrams, module breakdowns, and contributor tips live in `docs/architecture.md` and `docs/testing.md`.

---

## Documentation & Support
- Operations: `docs/operations.md`
- Discord bot playbook: `docs/bot_operations.md`
- Architecture: `docs/architecture.md`
- Testing: `docs/testing.md`
- Issues & ideas: GitHub Issues and Discussions

---

## Contributing
- Follow conventional commits (`feat:`, `fix:`, `docs:`).
- Keep secrets out of git; use `.env` files and synthetic sample data.
- Run `make quality` before opening a pull request.

---

## License & Credits
- Licensed under the MIT License – see `LICENSE`.
- Created by Jose Cordovilla and the Pepino contributors community.

