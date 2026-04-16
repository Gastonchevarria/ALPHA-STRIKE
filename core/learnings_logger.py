"""Structured markdown logging for learnings, errors, and feature requests."""

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LEARNINGS_DIR = Path("learnings")


def ensure_learnings_dir():
    _LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["LEARNINGS.md", "ERRORS.md", "FEATURE_REQUESTS.md"]:
        path = _LEARNINGS_DIR / fname
        if not path.exists():
            path.write_text(f"# {fname.replace('.md', '').replace('_', ' ').title()}\n\n")


def _next_id(prefix: str, filepath: Path) -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    base = f"{prefix}-{today}"
    if filepath.exists():
        content = filepath.read_text()
        count = content.count(f"{base}-")
    else:
        count = 0
    return f"{base}-{count + 1:03d}"


def log_learning(
    category: str,
    summary: str,
    details: str,
    action: str,
    area: str = "backend",
    priority: str = "medium",
    tags: list[str] | None = None,
) -> str:
    ensure_learnings_dir()
    filepath = _LEARNINGS_DIR / "LEARNINGS.md"
    entry_id = _next_id("LRN", filepath)
    now = datetime.now(timezone.utc).isoformat()
    tag_str = ", ".join(tags) if tags else "general"

    entry = f"\n## {entry_id}: {summary}\n"
    entry += f"- **Category**: {category}\n"
    entry += f"- **Area**: {area} | **Priority**: {priority}\n"
    entry += f"- **Tags**: {tag_str}\n"
    entry += f"- **Timestamp**: {now}\n"
    entry += f"- **Details**: {details}\n"
    entry += f"- **Action**: {action}\n"

    with open(filepath, "a") as f:
        f.write(entry)
    return entry_id


def log_error(
    skill_or_command: str,
    summary: str,
    error_text: str,
    context: str,
    suggested_fix: str,
) -> str:
    ensure_learnings_dir()
    filepath = _LEARNINGS_DIR / "ERRORS.md"
    entry_id = _next_id("ERR", filepath)
    now = datetime.now(timezone.utc).isoformat()

    entry = f"\n## {entry_id}: {summary}\n"
    entry += f"- **Command**: {skill_or_command}\n"
    entry += f"- **Timestamp**: {now}\n"
    entry += f"- **Error**: {error_text}\n"
    entry += f"- **Context**: {context}\n"
    entry += f"- **Suggested Fix**: {suggested_fix}\n"

    with open(filepath, "a") as f:
        f.write(entry)
    return entry_id
