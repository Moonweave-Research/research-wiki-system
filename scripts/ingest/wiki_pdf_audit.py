import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

try:
    from legacy.graphrag.wiki_graph import iter_markdown_files, parse_note, resolve_wiki_root
except ModuleNotFoundError:
    from wiki_graph import iter_markdown_files, parse_note, resolve_wiki_root  # type: ignore[no-redef]


TextExtractor = Callable[[Path], str]

SECTION_PATTERNS = [
    ("Abstract", re.compile(r"(?im)^\s*(?:\d+\.?\s*)?abstract\s*$")),
    ("Introduction", re.compile(r"(?im)^\s*(?:\d+\.?\s*)?introduction\s*$")),
    ("Methods", re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:methods?|materials and methods)\s*$")),
    (
        "Results",
        re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:results?|results and discussion)\s*$"),
    ),
    ("Conclusion", re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:conclusions?|summary)\s*$")),
]


def default_extract_text(pdf_path: Path, max_pages: int | None = None) -> str:
    # WIKI_PDF_MAX_PAGES overrides the page cap (default 3, current behavior);
    # a value <= 0 extracts the whole document.
    import os

    if max_pages is None:
        raw = os.environ.get("WIKI_PDF_MAX_PAGES", "3")
        try:
            max_pages = int(raw)
        except ValueError:
            max_pages = 3
    cmd = ["pdftotext"]
    if max_pages > 0:
        cmd += ["-f", "1", "-l", str(max_pages)]
    cmd += ["-layout", str(pdf_path), "-"]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def detect_section_headings(text: str) -> list[str]:
    headings = []
    for label, pattern in SECTION_PATTERNS:
        if pattern.search(text):
            headings.append(label)
    return headings


def paper_notes(wiki_dir: Path):
    wiki_root = resolve_wiki_root(wiki_dir)
    for path in iter_markdown_files(wiki_root):
        note = parse_note(path, wiki_root)
        if note.kind == "Paper":
            yield note


def audit_pdfs(
    wiki_dir: Path,
    *,
    extract_text: TextExtractor | None = None,
    sample: int | None = None,
) -> dict[str, Any]:
    extractor = extract_text or default_extract_text
    rows = []

    for note in paper_notes(wiki_dir):
        if sample is not None and len(rows) >= sample:
            break
        pdf_path = Path(note.pdf_path).expanduser() if note.pdf_path else None
        row = {
            "paper_id": note.node_id,
            "note_path": note.rel_path,
            "pdf_path": note.pdf_path,
            "has_pdf_path": bool(note.pdf_path),
            "pdf_exists": bool(pdf_path and pdf_path.exists()),
            "readable": False,
            "text_chars": 0,
            "section_headings": [],
            "error": "",
        }
        if pdf_path and pdf_path.exists():
            try:
                text = extractor(pdf_path)
                row["text_chars"] = len(text.strip())
                row["section_headings"] = detect_section_headings(text)
                row["readable"] = row["text_chars"] > 0
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                row["error"] = str(exc)
        rows.append(row)

    summary = {
        "papers": len(rows),
        "with_pdf_path": sum(1 for row in rows if row["has_pdf_path"]),
        "existing_pdfs": sum(1 for row in rows if row["pdf_exists"]),
        "readable_pdfs": sum(1 for row in rows if row["readable"]),
        "with_section_headings": sum(1 for row in rows if row["section_headings"]),
        "missing_pdfs": sum(1 for row in rows if row["has_pdf_path"] and not row["pdf_exists"]),
    }
    return {"summary": summary, "papers": rows}


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"papers: {summary['papers']}")
    print(f"with_pdf_path: {summary['with_pdf_path']}")
    print(f"existing_pdfs: {summary['existing_pdfs']}")
    print(f"readable_pdfs: {summary['readable_pdfs']}")
    print(f"with_section_headings: {summary['with_section_headings']}")
    print(f"missing_pdfs: {summary['missing_pdfs']}")

    problem_rows = [
        row
        for row in report["papers"]
        if row["has_pdf_path"] and (not row["pdf_exists"] or not row["readable"])
    ][:10]
    if problem_rows:
        print("\nproblem_samples:")
        for row in problem_rows:
            reason = "missing" if not row["pdf_exists"] else row["error"] or "not readable"
            print(f"  {row['paper_id']} {reason} {row['pdf_path']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit PDF readiness for semantic extraction.")
    parser.add_argument("--wiki-dir", default="wiki")
    parser.add_argument("--sample", type=int, help="Audit only the first N paper notes")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report")
    args = parser.parse_args(argv)

    if args.sample is not None and args.sample <= 0:
        print("sample must be positive")
        raise SystemExit(2)

    try:
        report = audit_pdfs(Path(args.wiki_dir), sample=args.sample)
    except (OSError, subprocess.SubprocessError) as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print_report(report)


if __name__ == "__main__":
    main()
