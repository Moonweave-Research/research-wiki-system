#!/usr/bin/env python3
"""Create a local dry-run export tree for the reusable research-wiki system.

This does not publish, push, move private data, or rewrite history. It copies a
tracked-file allowlist into a local output directory and writes a public-oriented
README so the candidate tree can be scanned before any real repository split.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

INCLUDE_PREFIXES = (
    "apps/",
    "scripts/",
    "tests/",
    "product_samples/",
)
INCLUDE_FILES = {
    "pyproject.toml",
    "uv.lock",
}
EXCLUDE_PREFIXES = (
    "wiki/",
    ".obsidian/",
    ".graph_index/",
    ".run/",
    ".omx/",
    ".wrangler/",
    ".git/",
    ".claude/",
    ".superpowers/",
    ".gitnexus/",
    "내 라이브러리/",
)
EXCLUDE_PARTS = {
    ".DS_Store",
    ".wrangler",
    "node_modules",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}
PRIVATE_MARKERS = (
    "choe" + "mun",
    "choi" + "moonyoung",
    "/" + "Users" + "/",
    "Google" + "Drive",
    "Cloud" + "Storage",
    "zotero://" + "select/library/items/",
    "base" + "AttachmentPath",
)

README = """# research-wiki-system export candidate

This is a local dry-run export of the reusable research-wiki system code.

It intentionally excludes the private vault corpus:

- `wiki/`
- Zotero/BetterBibTeX exports
- local PDF/library attachments
- personal Obsidian state
- generated indexes such as `.graph_index/`
- local runtime/deploy caches such as `.omx/`, `.run/`, and `.wrangler/`

Use `tests/fixtures/sample-vault/` as the public-safe sample vault while testing
system behavior outside the private observatory repository.
"""


def tracked_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, check=True, text=True, capture_output=True
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def is_system_export_path(path: Path) -> bool:
    posix = path.as_posix()
    if posix in INCLUDE_FILES:
        return True
    if any(posix.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return False
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    return any(posix.startswith(prefix) for prefix in INCLUDE_PREFIXES)


def has_private_marker(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return "unreadable"
    if b"\x00" in data[:4096]:
        return None
    text = data.decode("utf-8", errors="ignore")
    lowered = text.lower()
    for marker in PRIVATE_MARKERS:
        if marker == "/" + "Users" + "/":
            if marker in text:
                return marker
            continue
        if marker.lower() in lowered:
            return marker
    return None


def clean_output_dir(output_dir: Path, repo_root: Path) -> None:
    resolved = output_dir.resolve()
    allowed_root = (repo_root / ".run" / "system-export").resolve()
    if allowed_root != resolved and allowed_root not in resolved.parents:
        raise SystemExit(f"refusing --clean outside {allowed_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def export_system(repo_root: Path, output_dir: Path) -> list[Path]:
    copied: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for rel in tracked_files(repo_root):
        if not is_system_export_path(rel):
            continue
        src = repo_root / rel
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    (output_dir / "README.md").write_text(README, encoding="utf-8")
    copied.append(Path("README.md"))
    return copied


def scan_export(output_dir: Path) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(output_dir).as_posix()
        if any(part in EXCLUDE_PARTS for part in path.relative_to(output_dir).parts):
            findings.append((rel, "excluded-part-present"))
            continue
        marker = has_private_marker(path)
        if marker:
            findings.append((rel, marker))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--clean", action="store_true", help="Clean output first; only allowed under .run/system-export/."
    )
    parser.add_argument("--scan", action="store_true", help="Scan copied files for private markers.")
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    output_dir = args.output
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if args.clean:
        clean_output_dir(output_dir, repo_root)
    copied = export_system(repo_root, output_dir)
    print(f"exported {len(copied)} files -> {output_dir}")
    if args.scan:
        findings = scan_export(output_dir)
        if findings:
            print("private-marker findings:")
            for rel, marker in findings:
                print(f"  {rel}: {marker}")
            return 1
        print("scan passed: no private markers found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
