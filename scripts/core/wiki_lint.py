import yaml
import re
import sys
import glob
import os
from pathlib import Path

try:
    from scripts.core.wiki_core import validate_frontmatter
except ModuleNotFoundError:
    from wiki_core import validate_frontmatter  # type: ignore[no-redef]

LLM_REJECTION_RE = re.compile(
    r"I appreciate you testing|cannot summarize|outside (my|the) scope|"
    r"as an AI|I cannot provide|unable to provide",
    re.IGNORECASE,
)
FRONTMATTER_GARBAGE_RE = re.compile(r"[A-Za-z]@[A-Za-z]|&[A-Za-z]{3,}|Kqwords")
SKIPPED_LINT_PATHS = {
    os.path.join("wiki", "log.md"),
    os.path.join("wiki", "index.md"),
    os.path.join("wiki", "dashboard.md"),
    os.path.join("wiki", "00_Dashboard.md"),
    os.path.join("wiki", "research", "01_Literature_Index.md"),
    os.path.join("wiki", "research", "02_Research_Timeline.md"),
}
SKIPPED_LINT_DIR_PARTS = {
    os.path.join("wiki", "meta"),
    os.path.join("wiki", "templates"),
    os.path.join("wiki", "dispatches"),
}


def _should_skip_lint_path(filepath: str) -> bool:
    normalized = os.path.normpath(filepath)
    if normalized in SKIPPED_LINT_PATHS:
        return True
    return any(
        normalized == part or normalized.startswith(part + os.sep)
        for part in SKIPPED_LINT_DIR_PARTS
    )

def _should_skip_lint_file(
    filepath: str | os.PathLike[str], wiki_dir: str | os.PathLike[str]
) -> bool:
    """Return whether a markdown file should be skipped for a specific vault root.

    The public skip contract stays expressed as wiki-relative paths so callers
    that pass ``wiki/foo.md`` keep their current behavior, while the CLI can now
    lint an external vault path without hard-coding the repository-local
    ``wiki/`` directory.
    """
    path = Path(filepath)
    root = Path(wiki_dir)
    try:
        relative = path.relative_to(root)
    except ValueError:
        return _should_skip_lint_path(str(path))
    return _should_skip_lint_path(os.path.join("wiki", str(relative)))


def _has_garbage_in_values(values) -> bool:
    if values is None:
        return False
    if isinstance(values, str):
        return bool(FRONTMATTER_GARBAGE_RE.search(values))
    if isinstance(values, list):
        return any(_has_garbage_in_values(v) for v in values)
    return False


def lint_file(filepath: str) -> tuple[bool, str]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.match(r"^---\r?\n(.*?)\r?\n---(?:\r?\n|$)", content, re.DOTALL)
    if not match:
        return False, "No valid YAML frontmatter block found"

    try:
        data = yaml.safe_load(match.group(1))
        is_valid, msg = validate_frontmatter(data)
        if not is_valid:
            return False, msg
    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {str(e)}"

    if isinstance(data, dict):
        for field in ("aliases", "tags"):
            if _has_garbage_in_values(data.get(field)):
                return (
                    False,
                    f"frontmatter garbage in {field} (OCR artifact like @, &, or 'Kqwords')",
                )

    markdown_body = content[match.end() :]
    if re.search(r"\[.+?\]\((?!http[s]?://|zotero://).+?\)", markdown_body):
        return (
            False,
            "Standard Markdown links [text](url) are not allowed for internal files. Use [[wikilinks]]. HTTP and Zotero links are allowed.",
        )

    if isinstance(data, dict) and str(data.get("status") or "") != "irrelevant":
        if LLM_REJECTION_RE.search(markdown_body):
            return (
                False,
                "LLM refusal phrases detected in body — mark `status: irrelevant` if intentional",
            )

    if (
        isinstance(data, dict)
        and data.get("category") == "materials"
        and str(data.get("status") or "") != "irrelevant"
    ):
        if data.get("pdf_path") == "":
            return False, "empty pdf_path string — omit field instead"
        abstract_match = re.search(
            r"^##\s+Abstract\s*\n(.*?)(?=^##\s+|\Z)", markdown_body, re.MULTILINE | re.DOTALL
        )
        if abstract_match and not abstract_match.group(1).strip():
            return False, "missing abstract body under ## Abstract"
        if not re.search(r"^##\s+핵심\s*메커니즘", markdown_body, re.MULTILINE):
            return False, "missing ## 핵심 메커니즘 section for materials category"

    return True, ""


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def find_duplicate_titles(notes) -> dict[str, list[str]]:
    """Group notes sharing a normalized title WITHIN the same top-level folder.

    Scoping to a single folder is deliberate: a concept note and its same-named
    tag page (concepts/X vs meta/Tags/X) legitimately share a title and are NOT
    duplicates. Same-folder collisions — e.g. research/foo2021.md and the
    re-ingested research/foo2021a.md (BBT citekey disambiguation) — are the real
    duplicate-ingest signal. Titles under 3 chars after normalization are ignored.
    The returned key is "<folder>: <title>".
    """
    by_key: dict[str, list[str]] = {}
    for note in notes:
        normalized = _normalize_title(note.title or "")
        if len(normalized) < 3:
            continue
        folder = note.path.split("/", 1)[0] if "/" in note.path else ""
        by_key.setdefault(f"{folder}: {normalized}", []).append(note.path)
    return {key: paths for key, paths in by_key.items() if len(paths) > 1}


def _classify_failure(msg: str) -> str:
    if "frontmatter garbage" in msg:
        return "garbage-frontmatter"
    if "LLM refusal" in msg:
        return "contamination"
    if "Missing required keys" in msg:
        return "missing-required-keys"
    if "Invalid category" in msg:
        return "invalid-category"
    if "Invalid domain" in msg:
        return "invalid-domain"
    if "YAML parsing" in msg or "frontmatter block" in msg:
        return "frontmatter-parse-error"
    if "Markdown links" in msg:
        return "internal-markdown-link"
    if "missing abstract body" in msg:
        return "missing-abstract"
    if "missing ## 핵심 메커니즘" in msg:
        return "missing-mechanism"
    if "empty pdf_path" in msg:
        return "empty-pdf-path"
    return "other"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lint wiki markdown files")
    parser.add_argument("--verbose", action="store_true", help="Show per-file failure list")
    parser.add_argument(
        "--wiki-dir",
        default=os.getenv("WIKI_LINT_WIKI_DIR", "wiki"),
        help="Vault root to lint; defaults to WIKI_LINT_WIKI_DIR or repository-local wiki/.",
    )
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="Also report broken wikilinks (failure) and orphan notes (informational) "
        "via the link-graph index.",
    )
    args = parser.parse_args()

    wiki_dir = Path(args.wiki_dir)
    files = glob.glob(str(wiki_dir / "**" / "*.md"), recursive=True)
    failures_by_group: dict[str, list[tuple[str, str]]] = {}
    for f in files:
        if _should_skip_lint_file(f, wiki_dir):
            continue
        is_valid, msg = lint_file(f)
        if not is_valid:
            group = _classify_failure(msg)
            failures_by_group.setdefault(group, []).append((f, msg))

    try:
        from scripts.core.linkgraph import broken_links, orphan_notes, scan_vault
    except ModuleNotFoundError:
        from linkgraph import broken_links, orphan_notes, scan_vault  # type: ignore[no-redef]

    notes, links = scan_vault(wiki_dir)
    dup_notes = [
        note for note in notes if not _should_skip_lint_path(os.path.join("wiki", note.path))
    ]
    for normalized, paths in find_duplicate_titles(dup_notes).items():
        wiki_paths = sorted(str(wiki_dir / path) for path in paths)
        for wiki_path in wiki_paths:
            others = [other for other in wiki_paths if other != wiki_path]
            failures_by_group.setdefault("duplicate-title", []).append(
                (wiki_path, f"duplicate title '{normalized}' shared with {', '.join(others)}")
            )

    if args.check_links:
        for link in broken_links(links):
            if _should_skip_lint_path(os.path.join("wiki", link.src_path)):
                continue
            failures_by_group.setdefault("broken-wikilink", []).append(
                (link.src_path, f"unresolved wikilink [[{link.dst_target}]]")
            )
        orphans = [
            note
            for note in orphan_notes(notes, links)
            if not _should_skip_lint_path(os.path.join("wiki", note.path))
        ]
        if orphans:
            print(f"[lint] {len(orphans)} orphan notes (no inbound/outbound links):")
            for note in orphans:
                print(f"    orphan: {note.path}")

    if not failures_by_group:
        print("All wiki files passed linting.")
        sys.exit(0)

    total = sum(len(items) for items in failures_by_group.values())
    print(f"[lint] {total} failures across {len(failures_by_group)} groups:")
    for group in sorted(failures_by_group):
        items = failures_by_group[group]
        print(f"  {group}: {len(items)}")
        if args.verbose:
            for path, msg in items:
                print(f"    [{path}] {msg}")
    sys.exit(1)
