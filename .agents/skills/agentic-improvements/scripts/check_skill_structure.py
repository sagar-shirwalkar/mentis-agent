"""Verify agentic-improvements skill structure integrity.

Checks:
  - All required directories exist
  - SKILL.md leads match phases
  - Reference files linked from SKILL.md exist
  - No broken local file references

Usage:
    uv run scripts/check_skill_structure.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
ASSETS = SKILL_DIR / "assets"
REFERENCES = SKILL_DIR / "references"
SCRIPTS = SKILL_DIR / "scripts"
SKILL_MD = SKILL_DIR / "SKILL.md"

REQUIRED_DIRS = [ASSETS, REFERENCES, SCRIPTS]

PHASES = ["Vision", "Analyze", "Blueprint", "Adapt", "Crosswalk", "Reckon"]
LEADING_WORDS = ["analyze", "blueprint", "distill", "crosswalk"]


def check_directories() -> list[str]:
    errors: list[str] = []
    for d in REQUIRED_DIRS:
        if not d.is_dir():
            errors.append(f"Missing directory: {d}")
    return errors


def check_skill_md() -> list[str]:
    errors: list[str] = []
    if not SKILL_MD.is_file():
        return [f"Missing SKILL.md at {SKILL_MD}"]

    text = SKILL_MD.read_text(encoding="utf-8")

    # Check phases (formatted as `### Phase N: Name`)
    for phase in PHASES:
        count = text.count(f"### Phase ")
        phase_lines = [l for l in text.splitlines() if f"### Phase " in l]
        found = [l for l in phase_lines if phase.lower() in l.lower() and ":" in l]
        if len(found) != 1:
            errors.append(f"Phase '{phase}' section {'not found' if not found else f'found {len(found)} times'} in SKILL.md")

    # Check leading words
    for word in LEADING_WORDS:
        if word not in text.lower():
            errors.append(f"Leading word '{word}' not found in SKILL.md")

    # Check reference links
    ref_links = re.findall(r"references/([\w.-]+)", text)
    for ref in ref_links:
        ref_path = REFERENCES / ref
        if not ref_path.exists():
            errors.append(f"Reference file referenced but missing: {ref_path}")

    return errors


def main() -> int:
    errors = check_directories()
    errors += check_skill_md()

    if errors:
        print("Skill structure issues found:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    else:
        print("✓ Skill structure is valid")
        return 0


if __name__ == "__main__":
    sys.exit(main())
