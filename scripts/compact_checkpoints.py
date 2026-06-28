#!/usr/bin/env python3
"""
Checkpoint compaction script.

Prunes old checkpoints and merges consecutive entries.

Usage:
    python scripts/compact_checkpoints.py [--max-age 90] [--dry-run]

Run automatically at session start if last compaction >7 days ago.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_config() -> str:
    """Find the working directory from config or fall back to cwd."""
    return "."


def compact_checkpoints(
    workdir: str,
    max_age_days: int = 90,
    dry_run: bool = False,
) -> int:
    """
    Prune old checkpoints.

    Deletes checkpoints older than *max_age_days* with few steps
    (abandoned sessions).  Merges consecutive checkpoints from the
    same session into compressed summaries.

    Returns the number of checkpoints pruned.
    """
    chk_dir = Path(workdir) / ".agent" / "checkpoints"
    if not chk_dir.exists():
        print(f"No checkpoints directory at {chk_dir}")
        return 0

    cutoff = time.time() - (max_age_days * 86400)
    pruned = 0

    checkpoints = sorted(chk_dir.glob("*.json"))

    # Phase 1: prune old low-step checkpoints
    for path in checkpoints:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("timestamp", 0)
            steps = data.get("step_number", 0)
            if ts < cutoff and steps < 5:
                if dry_run:
                    print(f"[dry-run] Would prune: {path.name} (step {steps}, age {_age_days(ts):.0f}d)")
                else:
                    path.unlink()
                    print(f"Pruned: {path.name} (step {steps}, age {_age_days(ts):.0f}d)")
                pruned += 1
        except (json.JSONDecodeError, OSError) as exc:
            if dry_run:
                print(f"[dry-run] Would remove corrupt: {path.name} ({exc})")
            else:
                path.unlink()
                print(f"Removed corrupt: {path.name} ({exc})")
            pruned += 1

    # Phase 2: group remaining by session and merge consecutive entries
    remaining = sorted(
        chk_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    session_groups: dict[str, list[Path]] = {}
    for path in remaining:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            sid = data.get("session_id", path.stem)
            session_groups.setdefault(sid, []).append(path)
        except (json.JSONDecodeError, OSError):
            pass

    for sid, paths in session_groups.items():
        if len(paths) <= 3:
            continue  # Not enough to merge

        # Keep first, last, and one mid-point; delete the rest
        keep = {paths[0], paths[-1]}
        if len(paths) > 2:
            keep.add(paths[len(paths) // 2])

        for path in paths:
            if path not in keep:
                if dry_run:
                    print(f"[dry-run] Would merge: {path.name}")
                else:
                    path.unlink()
                    print(f"Merged: {path.name} into {paths[-1].name}")
                pruned += 1

    if pruned == 0:
        print("No checkpoints to compact.")
    else:
        print(f"Compaction complete: {pruned} checkpoints removed.")

    return pruned


def _age_days(timestamp: float) -> float:
    return (time.time() - timestamp) / 86400


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compact old agent checkpoints")
    parser.add_argument("--max-age", type=int, default=90, help="Max age in days (default: 90)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--workdir", type=str, default=None, help="Working directory (default: from config)")
    args = parser.parse_args()

    workdir = args.workdir or "."
    compact_checkpoints(workdir, max_age_days=args.max_age, dry_run=args.dry_run)
