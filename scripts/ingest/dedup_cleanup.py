#!/usr/bin/env python3
"""Archive redundant duplicate research notes (same paper re-ingested under BBT
a/b/c citekeys). Reversible: redundant notes are MOVED to .archive/duplicates/,
never deleted. The longest-body note in each same-title group is kept (most
extracted content); ties break toward the shortest citekey (the un-suffixed base).

    uv run python -m scripts.ingest.dedup_cleanup            # dry-run (default)
    uv run python -m scripts.ingest.dedup_cleanup --apply    # move redundant notes
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

DEFAULT_VAULT = Path("wiki/research")
DEFAULT_ARCHIVE = Path(".archive/duplicates/research")


def _norm_title(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _title_of(md_path: Path) -> str:
    """First alias in frontmatter, else the first '# ' heading, else the stem."""
    text = md_path.read_text(encoding="utf-8")
    alias = re.search(r"^aliases:\s*\[(.*?)\]", text, re.MULTILINE)
    if alias and alias.group(1).strip():
        return alias.group(1).split(",")[0].strip().strip("\"'")
    heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if heading:
        return heading.group(1).strip()
    return md_path.stem


def _body_len(md_path: Path) -> int:
    text = md_path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    return len((parts[2] if len(parts) >= 3 else text).strip())


def plan_cleanup(vault: Path) -> list[tuple[Path, list[Path]]]:
    """Return [(keeper, [redundant, ...]), ...] for each same-title group >1."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for md in sorted(vault.glob("*.md")):
        title = _norm_title(_title_of(md))
        if len(title) > 8:
            groups[title].append(md)
    plan = []
    for members in groups.values():
        if len(members) < 2:
            continue
        keeper = max(members, key=lambda p: (_body_len(p), -len(p.stem)))
        redundant = [m for m in members if m != keeper]
        plan.append((keeper, redundant))
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--apply", action="store_true", help="Actually move (default: dry-run)")
    args = parser.parse_args(argv)

    plan = plan_cleanup(args.vault)
    moved = 0
    for keeper, redundant in plan:
        for note in redundant:
            print(f"  keep {keeper.name}  <-  archive {note.name}")
            if args.apply:
                args.archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(note), str(args.archive / note.name))
            moved += 1
    verb = "moved" if args.apply else "would move"
    print(f"\n{len(plan)} duplicate groups; {verb} {moved} redundant notes to {args.archive}")
    if not args.apply:
        print("(dry-run — re-run with --apply to move; then rebuild the indexes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
