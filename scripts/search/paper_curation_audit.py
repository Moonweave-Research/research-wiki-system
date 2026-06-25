#!/usr/bin/env python3
"""Read-only paper curation gap report.

This complements the concept graph audit: it lists the concrete notes that block
claiming the paper corpus is fully curated.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from legacy.graphrag.wiki_graph import iter_markdown_files, parse_note, resolve_wiki_root
except ModuleNotFoundError:
    from wiki_graph import iter_markdown_files, parse_note, resolve_wiki_root  # type: ignore[no-redef]

from scripts.core.linkgraph import orphan_notes, scan_vault
from scripts.core.wiki_lint import _should_skip_lint_path
from scripts.ingest import concept_routing


DEFAULT_VAULT = Path("wiki")
DEFAULT_LINKGRAPH_DB = Path(".graph_index/linkgraph.sqlite")
ARCHIVE_REVIEW_TERMS = (
    "fuel cell",
    "magnetic bearing",
    "quantum",
    "reviewer",
    "writing a paper",
    "manual",
    "standard",
    "microwave discharge",
    "blood pump",
    "spindle",
)
ZOTERO_SELECT_PREFIX = "zotero://" + "select/library/items/"
ZOTERO_ITEM_RE = re.compile(re.escape(ZOTERO_SELECT_PREFIX) + r"([A-Z0-9]+)")


def _connect(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _paper_notes(vault_dir: Path) -> list[Any]:
    wiki_root = resolve_wiki_root(vault_dir)
    return [
        note
        for path in iter_markdown_files(wiki_root)
        if (note := parse_note(path, wiki_root)).kind == "Paper"
    ]


def _graph_research_rows(linkgraph_db: Path) -> dict[str, dict[str, str]]:
    conn = _connect(linkgraph_db)
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT path, title, status FROM notes WHERE path LIKE 'research/%'"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    return {
        str(row["path"]): {
            "path": str(row["path"]),
            "title": str(row["title"] or row["path"]),
            "status": str(row["status"] or "__missing__"),
        }
        for row in rows
    }


def _concept_linked_research(linkgraph_db: Path) -> set[str]:
    conn = _connect(linkgraph_db)
    if conn is None:
        return set()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT dst_target
            FROM links
            WHERE dst_resolved = 1
              AND src_path LIKE 'concepts/%'
              AND dst_target LIKE 'research/%'
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()
    return {str(row["dst_target"]) for row in rows}


def _pdf_state(pdf_path: str) -> str:
    if not pdf_path:
        return "no_pdf_path"
    return "pdf_exists" if Path(pdf_path).expanduser().exists() else "missing_pdf"


def _paper_row(note: Any) -> dict[str, Any]:
    state = _pdf_state(str(note.pdf_path or ""))
    return {
        "path": note.rel_path,
        "title": note.title,
        "status": note.status or "__missing__",
        "pdf_path": note.pdf_path,
        "pdf_state": state,
        "tags": list(note.tags),
        "sources": list(note.sources),
    }


def _zotero_key(sources: list[str]) -> str:
    for source in sources:
        match = ZOTERO_ITEM_RE.search(source)
        if match:
            return match.group(1)
    return ""


def _load_zotero_metadata(zotero_db: Path | None) -> dict[str, dict[str, Any]]:
    if zotero_db is None or not zotero_db.exists():
        return {}
    conn = _connect(zotero_db)
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            """
            SELECT i.itemID, i.key, it.typeName, f.fieldName, v.value
            FROM items i
            LEFT JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
            LEFT JOIN itemData d ON d.itemID = i.itemID
            LEFT JOIN fields f ON f.fieldID = d.fieldID
            LEFT JOIN itemDataValues v ON v.valueID = d.valueID
            """
        ).fetchall()
        attachment_rows = conn.execute(
            """
            SELECT parent.key AS parentKey, COUNT(ia.itemID) AS attachment_count
            FROM items parent
            LEFT JOIN itemAttachments ia ON ia.parentItemID = parent.itemID
            GROUP BY parent.key
            """
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()

    attachment_counts = {
        str(row["parentKey"]): int(row["attachment_count"] or 0)
        for row in attachment_rows
    }
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["key"])
        entry = metadata.setdefault(
            key,
            {
                "zotero_key": key,
                "zotero_item_type": str(row["typeName"] or ""),
                "zotero_attachment_count": attachment_counts.get(key, 0),
            },
        )
        field = str(row["fieldName"] or "")
        value = str(row["value"] or "")
        if field == "DOI":
            entry["doi"] = value
        elif field == "url":
            entry["url"] = value
        elif field == "ISBN":
            entry["isbn"] = value
        elif field == "publicationTitle":
            entry["publication_title"] = value
    return metadata


def _pdf_gap_row(row: dict[str, Any]) -> dict[str, Any]:
    zotero_key = str(row.get("zotero_key") or _zotero_key(list(row.get("sources", []))))
    doi = str(row.get("doi") or "").strip()
    url = str(row.get("url") or "").strip()
    isbn = str(row.get("isbn") or "").strip()
    attachment_count = int(row.get("zotero_attachment_count") or 0)
    reasons = ["no_zotero_attachment"] if zotero_key and attachment_count == 0 else []

    if doi:
        action = "retrieve_by_doi"
        locator = doi
        reasons.append("doi_available")
    elif url and "zotero.org/users/" not in url:
        action = "retrieve_by_url"
        locator = url
        reasons.append("url_available")
    elif isbn:
        action = "manual_book_or_library_access"
        locator = f"ISBN {isbn}"
        reasons.append("isbn_available")
    else:
        action = "manual_metadata_review"
        locator = zotero_key
        reasons.append("no_doi_url_or_isbn")

    return {
        **row,
        "zotero_key": zotero_key,
        "doi": doi,
        "url": url,
        "isbn": isbn,
        "zotero_attachment_count": attachment_count,
        "locator": locator,
        "acquisition_action": action,
        "acquisition_reasons": reasons,
    }


def _orphan_rows(
    vault_dir: Path,
    paper_by_path: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    notes, links = scan_vault(vault_dir)
    paper_rows = paper_by_path or {}
    rows = []
    for note in orphan_notes(notes, links):
        if _should_skip_lint_path(str(Path("wiki") / note.path)):
            continue
        paper = paper_rows.get(note.path, {})
        rows.append(
            {
                "path": note.path,
                "title": note.title,
                "kind": "Paper" if note.path.startswith("research/") else "Note",
                "status": note.status or "__missing__",
                "tags": paper.get("tags", []),
                "pdf_state": paper.get("pdf_state", ""),
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def _is_actionable_status(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "") != "irrelevant"


def _is_actionable_queue_row(row: dict[str, Any]) -> bool:
    if not _is_actionable_status(row):
        return False
    return str(row.get("action") or "") != "documented_scaffold"


def _is_skipped_wiki_path(path: str) -> bool:
    return _should_skip_lint_path(str(Path("wiki") / path))


def _matching_concepts(
    tags: list[str],
    concept_index: dict[str, concept_routing.ExistingConcept],
) -> list[str]:
    matches = []
    seen = set()
    for tag in tags:
        concept = concept_routing.find_concept(tag, concept_index)
        if concept is not None and concept.rel_path not in seen:
            seen.add(concept.rel_path)
            matches.append(concept.rel_path)
    return sorted(matches)


def _has_archive_review_hint(row: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(row.get("path", "")),
            str(row.get("title", "")),
            " ".join(str(tag) for tag in row.get("tags", [])),
        ]
    ).casefold()
    return any(term in text for term in ARCHIVE_REVIEW_TERMS)


def _classify_curation_row(
    row: dict[str, Any],
    *,
    concept_index: dict[str, concept_routing.ExistingConcept],
) -> dict[str, Any]:
    reasons = []
    matching = _matching_concepts(list(row.get("tags", [])), concept_index)
    if not _is_actionable_status(row):
        action = "already_irrelevant"
        reasons.append("status_irrelevant")
    elif _is_skipped_wiki_path(str(row["path"])):
        action = "documented_scaffold"
        reasons.append("skipped_wiki_path")
    elif _has_archive_review_hint(row):
        action = "archive_review_candidate"
        reasons.append("archive_keyword_hint")
    elif matching:
        action = "concept_link_candidate"
        reasons.append("tags_match_existing_concepts")
    elif row.get("pdf_state") and row.get("pdf_state") != "pdf_exists":
        action = "pdf_gap_first"
        reasons.append(str(row.get("pdf_state")))
    else:
        action = "taxonomy_gap_review"
        reasons.append("no_existing_concept_tag_match")
    return {
        **row,
        "action": action,
        "reasons": reasons,
        "matching_concepts": matching,
    }


def _curation_queue(
    zero_concept: list[dict[str, Any]],
    orphans: list[dict[str, Any]],
    *,
    concept_index: dict[str, concept_routing.ExistingConcept],
) -> list[dict[str, Any]]:
    by_path = {str(row["path"]): row for row in zero_concept}
    for row in orphans:
        path = str(row["path"])
        if row.get("kind") == "Paper":
            by_path[path] = {**by_path.get(path, {}), **row}
    return [
        _classify_curation_row(row, concept_index=concept_index)
        for row in sorted(by_path.values(), key=lambda item: str(item["path"]))
    ]


def build_curation_report(
    *,
    vault_dir: Path = DEFAULT_VAULT,
    linkgraph_db: Path = DEFAULT_LINKGRAPH_DB,
    zotero_db: Path | None = None,
) -> dict[str, Any]:
    papers = [_paper_row(note) for note in _paper_notes(vault_dir)]
    zotero_metadata = _load_zotero_metadata(zotero_db)
    papers = [
        {**row, **zotero_metadata.get(_zotero_key(list(row.get("sources", []))), {})}
        for row in papers
    ]
    paper_by_path = {str(row["path"]): row for row in papers}
    status_counts = Counter(str(row["status"]) for row in papers)
    graph_rows = _graph_research_rows(linkgraph_db)
    linked = _concept_linked_research(linkgraph_db)
    zero_concept = [
        {**row, **paper_by_path.get(path, {})}
        for path, row in sorted(graph_rows.items())
        if path not in linked
    ]
    pdf_gaps = [
        _pdf_gap_row(row)
        for row in sorted(papers, key=lambda item: item["path"])
        if row["pdf_state"] != "pdf_exists"
    ]
    orphans = _orphan_rows(vault_dir, paper_by_path)
    queue = _curation_queue(
        zero_concept,
        orphans,
        concept_index=concept_routing.load_concept_index(vault_dir),
    )
    action_counts = Counter(str(row["action"]) for row in queue if _is_actionable_queue_row(row))
    actionable_zero_concept = [
        row
        for row in zero_concept
        if _is_actionable_status(row) and not _is_skipped_wiki_path(str(row["path"]))
    ]
    actionable_pdf_gaps = [row for row in pdf_gaps if _is_actionable_status(row)]
    pdf_acquisition_counts = Counter(
        str(row["acquisition_action"]) for row in actionable_pdf_gaps
    )
    actionable_orphan_papers = [
        row
        for row in orphans
        if row["kind"] == "Paper" and _is_actionable_status(row)
    ]
    return {
        "summary": {
            "paper_notes": len(papers),
            "graph_research_nodes": len(graph_rows),
            "zero_concept": len(zero_concept),
            "pdf_gap_papers": len(pdf_gaps),
            "actionable_zero_concept": len(actionable_zero_concept),
            "actionable_pdf_gap_papers": len(actionable_pdf_gaps),
            "actionable_orphan_papers": len(actionable_orphan_papers),
            "orphan_notes": len(orphans),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "pdf_acquisition_counts": dict(sorted(pdf_acquisition_counts.items())),
        "curation_queue": queue,
        "zero_concept": zero_concept,
        "pdf_gaps": pdf_gaps,
        "orphan_notes": orphans,
    }


def _table(headers: list[str], rows: list[list[str]], *, limit: int = 40) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    if len(rows) > limit:
        lines.append(f"\n_showing {limit} of {len(rows)} rows_")
    return lines


def format_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Paper Curation Gap Audit",
        "",
        "## Summary",
        "",
        f"- paper_notes: {summary['paper_notes']}",
        f"- graph_research_nodes: {summary['graph_research_nodes']}",
        f"- zero_concept: {summary['zero_concept']}",
        f"- pdf_gap_papers: {summary['pdf_gap_papers']}",
        f"- actionable_zero_concept: {summary['actionable_zero_concept']}",
        f"- actionable_pdf_gap_papers: {summary['actionable_pdf_gap_papers']}",
        f"- actionable_orphan_papers: {summary['actionable_orphan_papers']}",
        f"- orphan_notes: {summary['orphan_notes']}",
        f"- status_counts: {json.dumps(report['status_counts'], ensure_ascii=False, sort_keys=True)}",
        f"- action_counts: {json.dumps(report.get('action_counts', {}), ensure_ascii=False, sort_keys=True)}",
        f"- pdf_acquisition_counts: {json.dumps(report.get('pdf_acquisition_counts', {}), ensure_ascii=False, sort_keys=True)}",
        "",
        "## Curation Queue",
        "",
    ]
    lines.extend(
        _table(
            ["path", "title", "action", "reasons", "matching_concepts"],
            [
                [
                    str(row["path"]),
                    str(row["title"]),
                    str(row["action"]),
                    ", ".join(str(reason) for reason in row.get("reasons", [])),
                    ", ".join(str(path) for path in row.get("matching_concepts", [])),
                ]
                for row in report.get("curation_queue", [])
            ],
        )
    )
    lines.extend(
        [
        "",
        "## Zero-Concept Research Nodes",
        "",
        ]
    )
    lines.extend(
        _table(
            ["path", "title", "status"],
            [
                [str(row["path"]), str(row["title"]), str(row.get("status", ""))]
                for row in report["zero_concept"]
            ],
        )
    )
    lines.extend(["", "## PDF Gaps", ""])
    lines.extend(
        _table(
            [
                "path",
                "title",
                "pdf_state",
                "status",
                "action",
                "locator",
                "zotero_attachments",
            ],
            [
                [
                    str(row["path"]),
                    str(row["title"]),
                    str(row["pdf_state"]),
                    str(row.get("status", "")),
                    str(row.get("acquisition_action", "")),
                    str(row.get("locator", "")),
                    str(row.get("zotero_attachment_count", "")),
                ]
                for row in report["pdf_gaps"]
            ],
        )
    )
    lines.extend(["", "## Orphan Notes", ""])
    lines.extend(
        _table(
            ["path", "title", "kind", "status"],
            [
                [
                    str(row["path"]),
                    str(row["title"]),
                    str(row["kind"]),
                    str(row.get("status", "")),
                ]
                for row in report["orphan_notes"]
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit paper curation coverage gaps.")
    parser.add_argument("--wiki-dir", default=str(DEFAULT_VAULT))
    parser.add_argument("--linkgraph-db", default=str(DEFAULT_LINKGRAPH_DB))
    parser.add_argument("--zotero-db", help="Optional Zotero sqlite copy for DOI/ISBN enrichment")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown")
    parser.add_argument("--output", help="Write report to this path instead of stdout")
    args = parser.parse_args(argv)

    report = build_curation_report(
        vault_dir=Path(args.wiki_dir),
        linkgraph_db=Path(args.linkgraph_db),
        zotero_db=Path(args.zotero_db) if args.zotero_db else None,
    )
    content = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.json
        else format_markdown(report)
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        return
    print(content, end="")


if __name__ == "__main__":
    main()
