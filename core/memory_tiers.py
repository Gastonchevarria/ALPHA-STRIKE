"""3-tier memory system with TTL-based pruning."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LEARNINGS_DIR = Path("learnings")

_TIER_CONFIG = {
    "short": {"max_entries": 20, "ttl_hours": 24, "file": "memory_short.json"},
    "mid": {"max_entries": 50, "ttl_hours": 168, "file": "memory_mid.json"},
    "long": {"max_entries": 100, "ttl_hours": None, "file": "memory_long.json"},
}


def _tier_path(tier: str) -> Path:
    return _LEARNINGS_DIR / _TIER_CONFIG[tier]["file"]


def _load_tier(tier: str) -> list[dict]:
    path = _tier_path(tier)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError):
            return []
    return []


def _save_tier(tier: str, entries: list[dict]):
    _LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    _tier_path(tier).write_text(json.dumps(entries, indent=2))


def _prune(tier: str, entries: list[dict]) -> list[dict]:
    cfg = _TIER_CONFIG[tier]
    now = datetime.now(timezone.utc)

    if cfg["ttl_hours"] is not None:
        cutoff = now - timedelta(hours=cfg["ttl_hours"])
        entries = [
            e for e in entries
            if datetime.fromisoformat(e["ts"]) > cutoff
        ]

    if len(entries) > cfg["max_entries"]:
        entries = entries[-cfg["max_entries"]:]

    return entries


def add_memory(tier: str, content: str, tags: list[str] | None = None) -> dict:
    if tier not in _TIER_CONFIG:
        raise ValueError(f"Invalid tier: {tier}")

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "content": content,
        "tags": tags or [],
    }

    entries = _load_tier(tier)
    entries.append(entry)
    entries = _prune(tier, entries)
    _save_tier(tier, entries)
    return entry


def get_recent(tier: str, limit: int = 10) -> list[dict]:
    entries = _load_tier(tier)
    entries = _prune(tier, entries)
    return entries[-limit:]


def get_all_context(max_per_tier: int = 5) -> str:
    parts = []
    for tier in ["long", "mid", "short"]:
        entries = get_recent(tier, max_per_tier)
        if entries:
            parts.append(f"### {tier.upper()} MEMORY ({len(entries)} entries)")
            for e in entries:
                ts = e["ts"][:10]
                parts.append(f"- [{ts}] {e['content']}")
    return "\n".join(parts) if parts else "No memories stored yet."


def promote(from_tier: str, content: str, tags: list[str] | None = None):
    tier_order = ["short", "mid", "long"]
    idx = tier_order.index(from_tier)
    if idx < len(tier_order) - 1:
        next_tier = tier_order[idx + 1]
        add_memory(next_tier, content, tags)
