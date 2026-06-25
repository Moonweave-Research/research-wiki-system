import argparse
import json
import re
from pathlib import Path

import yaml

try:
    from legacy.graphrag.wiki_graph import parse_frontmatter, extract_abstract_text, resolve_wiki_root
except ModuleNotFoundError:
    from wiki_graph import parse_frontmatter, extract_abstract_text, resolve_wiki_root  # type: ignore[no-redef]

try:
    from scripts.ingest.wiki_pdf_audit import paper_notes
except ModuleNotFoundError:
    from wiki_pdf_audit import paper_notes  # type: ignore[no-redef]

ABSTRACT_HEADER_RE = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?(?:ARTICLE\s+INFO\s+)?ABSTRACT\b(?:\s*[:\-–—]\s*|\s+|$)(.*)$"
)

SECTION_BOUNDARY_PATTERNS = [
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?introduction\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:materials and methods|methods)\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:results|results and discussion)\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:conclusions?|summary)\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:keywords?|key\s*words?|index terms)\b"),
    re.compile(r"(?im)^\s*키워드\b"),
    re.compile(r"(?im)^\s*(?:received|revised|accepted|open\s+access|corresponding(?:\s+author)?|copyright|creative\s+commons)\b"),
    re.compile(r"(?im)^\s*[Ⅰ-Ⅻ]\."),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?references\b"),
]

MIN_ABSTRACT_CHARS = 40
MAX_ABSTRACT_CHARS = 3500
URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)
NOISY_ARTIFACT_PATTERNS = [
    re.compile(r"\bdownloaded\s+via\b", re.IGNORECASE),
    re.compile(r"\bdownloaded\s+from\b", re.IGNORECASE),
]


def clean_artifact_segments(text: str) -> str:
    """Remove noisy PDF provider attribution chunks from abstract candidates."""

    return re.sub(r"\s*Downloaded\s+(?:via|from)\b.*?(?:\(UTC\)\.|\)|\.|$)", " ", text, flags=re.I)


def read_cache_manifest(path: Path) -> dict[str, Path]:
    manifest_dir = path.parent
    mapping: dict[str, Path] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            paper_id = str(row.get("paper_id") or "")
            cache_path = str(row.get("cache_path") or "")
            if not (paper_id and row.get("cached") is True and cache_path):
                continue
            resolved_path = Path(cache_path)
            if not resolved_path.is_absolute():
                if not resolved_path.exists():
                    manifest_relative = manifest_dir / resolved_path
                    if manifest_relative.exists():
                        resolved_path = manifest_relative
            mapping[paper_id] = resolved_path
    return mapping


def extract_cached_abstract(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ""

    abstract_candidates = []
    for index, line in enumerate(lines):
        match = ABSTRACT_HEADER_RE.match(line)
        if not match:
            continue
        abstract_candidates.append((index, match.group(1).strip()))

    if not abstract_candidates:
        return ""

    boundary_indexes = []
    for index, line in enumerate(lines):
        for pattern in SECTION_BOUNDARY_PATTERNS:
            if pattern.match(line):
                boundary_indexes.append(index)
                break

    for start_idx, inline_text in abstract_candidates:
        end_idx = len(lines)
        for boundary in boundary_indexes:
            if boundary > start_idx:
                end_idx = min(end_idx, boundary)
                break

        body_lines = []
        if inline_text:
            body_lines.append(inline_text)
        for line in lines[start_idx + 1 : end_idx]:
            if URL_PATTERN.search(line):
                break
            cleaned_line = clean_artifact_segments(line)
            if cleaned_line.strip():
                body_lines.append(cleaned_line.strip())

        text_block = " ".join(part for part in body_lines if part)
        text_block = clean_artifact_segments(text_block)
        text_block = re.sub(r"\s+", " ", text_block).strip()
        if len(text_block) < MIN_ABSTRACT_CHARS or len(text_block) > MAX_ABSTRACT_CHARS:
            continue
        if not text_block[0].isalpha() or not text_block[0].isupper():
            continue
        lower = text_block.lower()
        if "graphic abstract" in lower or "copyright" in lower:
            continue
        if any(p.search(text_block) for p in NOISY_ARTIFACT_PATTERNS):
            continue
        if URL_PATTERN.search(text_block):
            continue
        return text_block

    return ""


def fill_missing_abstracts(
    wiki_dir: Path,
    *,
    manifest_path: Path,
    dry_run: bool = False,
    sample: int | None = None,
) -> dict[str, int]:
    wiki_root = resolve_wiki_root(wiki_dir)
    cache_map = read_cache_manifest(manifest_path)

    summary = {
        "scanned": 0,
        "updated": 0,
        "already_had_abstract": 0,
        "missing_cache": 0,
        "missing_cache_file": 0,
        "failed_extract": 0,
        "dry_run": int(dry_run),
    }

    for note in paper_notes(wiki_root):
        if sample is not None and summary["scanned"] >= sample:
            break

        path = wiki_root / note.rel_path
        content = path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)

        if extract_abstract_text(frontmatter, body):
            summary["already_had_abstract"] += 1
            summary["scanned"] += 1
            continue

        cache_path = cache_map.get(note.node_id)
        if not cache_path:
            summary["missing_cache"] += 1
            summary["scanned"] += 1
            continue

        if not cache_path.exists():
            summary["missing_cache_file"] += 1
            summary["scanned"] += 1
            continue

        extracted = extract_cached_abstract(cache_path.read_text(encoding="utf-8"))
        if not extracted:
            summary["failed_extract"] += 1
            summary["scanned"] += 1
            continue

        frontmatter["abstract"] = extracted

        if not dry_run:
            yaml_str = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
            path.write_text(f"---\n{yaml_str}---\n{body}", encoding="utf-8")

        summary["updated"] += 1
        summary["scanned"] += 1

    return summary


def print_report(report: dict[str, int]) -> None:
    print(f"scanned: {report['scanned']}")
    print(f"updated: {report['updated']}")
    print(f"already_had_abstract: {report['already_had_abstract']}")
    print(f"missing_cache: {report['missing_cache']}")
    print(f"missing_cache_file: {report['missing_cache_file']}")
    print(f"failed_extract: {report['failed_extract']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fill missing paper abstract frontmatter from cached PDF text"
    )
    parser.add_argument("--wiki-dir", default="wiki")
    parser.add_argument("--manifest", default=".graph_index/pdf_text_manifest.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample", type=int, help="Process only the first N paper notes")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report")
    args = parser.parse_args(argv)

    if args.sample is not None and args.sample <= 0:
        print("sample must be positive")
        raise SystemExit(2)

    try:
        report = fill_missing_abstracts(
            Path(args.wiki_dir),
            manifest_path=Path(args.manifest),
            dry_run=args.dry_run,
            sample=args.sample,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    print_report(report)


if __name__ == "__main__":
    main()
