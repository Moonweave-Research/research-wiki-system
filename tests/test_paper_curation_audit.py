import sqlite3
from pathlib import Path

from scripts.search import paper_curation_audit

ZOTERO_SELECT_PREFIX = "zotero://" + "select/library/items/"


def _write_note(
    vault: Path,
    rel_path: str,
    *,
    alias: str,
    category: str = "materials",
    status: str = "unread",
    tags: list[str] | None = None,
    pdf_path: str | None = None,
    body: str | None = None,
) -> None:
    path = vault / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_line = "" if pdf_path is None else f"pdf_path: {pdf_path}\n"
    note_body = body or f"# {alias}\n\n## Abstract\nA paper.\n\n**Year:** 2024\n"
    path.write_text(
        "---\n"
        f"aliases: [{alias}]\n"
        "domain: research\n"
        f"category: {category}\n"
        f"status: {status}\n"
        f"tags: {tags or []}\n"
        "sources: []\n"
        f"{pdf_line}"
        "---\n"
        f"{note_body}",
        encoding="utf-8",
    )


def _linkgraph(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE notes (
            path TEXT PRIMARY KEY,
            title TEXT,
            domain TEXT,
            category TEXT,
            status TEXT,
            aliases_json TEXT,
            tags_json TEXT,
            mtime REAL
        );
        CREATE TABLE links (
            src_path TEXT,
            dst_target TEXT,
            dst_resolved INTEGER,
            kind TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("concepts/Actuator.md", "Actuator", "research", "concepts", "", "[]", "[]", 0),
            (
                "research/linked.md",
                "Linked",
                "research",
                "materials",
                "unread",
                "[]",
                "[]",
                0,
            ),
            (
                "research/zero.md",
                "Zero",
                "research",
                "materials",
                "unread",
                "[]",
                "[]",
                0,
            ),
        ],
    )
    conn.execute(
        "INSERT INTO links VALUES (?, ?, ?, ?)",
        ("concepts/Actuator.md", "research/linked.md", 1, "wikilink"),
    )
    conn.commit()
    conn.close()


def test_build_curation_report_lists_zero_concept_pdf_gaps_and_orphans(tmp_path: Path) -> None:
    vault = tmp_path / "wiki"
    existing_pdf = tmp_path / "paper.pdf"
    existing_pdf.write_bytes(b"%PDF-1.4 fake")
    missing_pdf = tmp_path / "missing.pdf"
    _write_note(vault, "concepts/Actuator.md", alias="Actuator", category="concepts")
    _write_note(vault, "research/linked.md", alias="Linked", pdf_path=str(existing_pdf))
    _write_note(vault, "research/zero.md", alias="Zero", pdf_path=str(missing_pdf))
    _write_note(vault, "research/no_pdf.md", alias="No PDF", pdf_path=None)
    _write_note(
        vault,
        "research/index.md",
        alias="Index",
        category="index",
        body="# Index\n",
    )
    linkgraph_db = tmp_path / "linkgraph.sqlite"
    _linkgraph(linkgraph_db)

    report = paper_curation_audit.build_curation_report(
        vault_dir=vault,
        linkgraph_db=linkgraph_db,
    )

    assert report["summary"]["paper_notes"] == 3
    assert report["summary"]["graph_research_nodes"] == 2
    assert report["summary"]["zero_concept"] == 1
    assert report["summary"]["pdf_gap_papers"] == 2
    assert report["summary"]["actionable_zero_concept"] == 1
    assert report["summary"]["actionable_pdf_gap_papers"] == 2
    assert report["summary"]["actionable_orphan_papers"] >= 1
    assert report["summary"]["orphan_notes"] >= 1
    assert report["status_counts"] == {"unread": 3}
    assert report["zero_concept"][0]["path"] == "research/zero.md"
    pdf_gap_paths = {row["path"]: row["pdf_state"] for row in report["pdf_gaps"]}
    assert pdf_gap_paths == {
        "research/no_pdf.md": "no_pdf_path",
        "research/zero.md": "missing_pdf",
    }
    assert "research/index.md" not in {row["path"] for row in report["pdf_gaps"]}


def test_pdf_gap_rows_include_acquisition_metadata() -> None:
    row = {
        "path": "research/example.md",
        "title": "Example",
        "status": "unread",
        "pdf_path": "",
        "pdf_state": "no_pdf_path",
        "tags": [],
        "sources": [f"{ZOTERO_SELECT_PREFIX}ABC123"],
        "doi": "10.1000/example",
        "url": "https://doi.org/10.1000/example",
        "isbn": "",
        "zotero_attachment_count": 0,
    }

    enriched = paper_curation_audit._pdf_gap_row(row)

    assert enriched["zotero_key"] == "ABC123"
    assert enriched["locator"] == "10.1000/example"
    assert enriched["acquisition_action"] == "retrieve_by_doi"
    assert enriched["acquisition_reasons"] == ["no_zotero_attachment", "doi_available"]


def test_pdf_gap_rows_classify_books_as_manual_acquisition() -> None:
    row = {
        "path": "research/book.md",
        "title": "Book",
        "status": "unread",
        "pdf_path": "",
        "pdf_state": "no_pdf_path",
        "tags": [],
        "sources": [f"{ZOTERO_SELECT_PREFIX}BOOK123"],
        "doi": "",
        "url": "",
        "isbn": "0-08-102377-4",
        "zotero_attachment_count": 0,
    }

    enriched = paper_curation_audit._pdf_gap_row(row)

    assert enriched["zotero_key"] == "BOOK123"
    assert enriched["locator"] == "ISBN 0-08-102377-4"
    assert enriched["acquisition_action"] == "manual_book_or_library_access"
    assert "isbn_available" in enriched["acquisition_reasons"]


def test_curation_queue_classifies_next_actions(tmp_path: Path) -> None:
    vault = tmp_path / "wiki"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    _write_note(vault, "concepts/Actuator.md", alias="Actuator", category="concepts")
    _write_note(
        vault,
        "research/zero.md",
        alias="Zero",
        tags=["Actuator"],
        pdf_path=str(pdf),
    )
    _write_note(
        vault,
        "research/writing.md",
        alias="Whitesides Group Writing a paper",
        tags=[],
        pdf_path=str(pdf),
    )
    _write_note(vault, "research/no_pdf.md", alias="No PDF", tags=[], pdf_path=None)
    linkgraph_db = tmp_path / "linkgraph.sqlite"
    conn = sqlite3.connect(linkgraph_db)
    conn.executescript(
        """
        CREATE TABLE notes (
            path TEXT PRIMARY KEY,
            title TEXT,
            domain TEXT,
            category TEXT,
            status TEXT,
            aliases_json TEXT,
            tags_json TEXT,
            mtime REAL
        );
        CREATE TABLE links (
            src_path TEXT,
            dst_target TEXT,
            dst_resolved INTEGER,
            kind TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("concepts/Actuator.md", "Actuator", "research", "concepts", "", "[]", "[]", 0),
            ("research/zero.md", "Zero", "research", "materials", "unread", "[]", "[]", 0),
            (
                "research/writing.md",
                "Whitesides Group Writing a paper",
                "research",
                "materials",
                "unread",
                "[]",
                "[]",
                0,
            ),
            ("research/no_pdf.md", "No PDF", "research", "materials", "unread", "[]", "[]", 0),
        ],
    )
    conn.commit()
    conn.close()

    report = paper_curation_audit.build_curation_report(
        vault_dir=vault,
        linkgraph_db=linkgraph_db,
    )

    actions = {row["path"]: row["action"] for row in report["curation_queue"]}
    assert actions["research/zero.md"] == "concept_link_candidate"
    assert actions["research/writing.md"] == "archive_review_candidate"
    assert actions["research/no_pdf.md"] == "pdf_gap_first"
    concept_row = next(row for row in report["curation_queue"] if row["path"] == "research/zero.md")
    assert concept_row["matching_concepts"] == ["concepts/Actuator.md"]
    assert report["action_counts"]["concept_link_candidate"] == 1


def test_skipped_wiki_scaffolds_are_documented_not_actionable(tmp_path: Path) -> None:
    vault = tmp_path / "wiki"
    _write_note(
        vault,
        "research/01_Literature_Index.md",
        alias="Literature Index",
        category="index",
        body="# Literature Index\n",
    )
    linkgraph_db = tmp_path / "linkgraph.sqlite"
    conn = sqlite3.connect(linkgraph_db)
    conn.executescript(
        """
        CREATE TABLE notes (
            path TEXT PRIMARY KEY,
            title TEXT,
            domain TEXT,
            category TEXT,
            status TEXT,
            aliases_json TEXT,
            tags_json TEXT,
            mtime REAL
        );
        CREATE TABLE links (
            src_path TEXT,
            dst_target TEXT,
            dst_resolved INTEGER,
            kind TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "research/01_Literature_Index.md",
            "Literature Index",
            "research",
            "index",
            "",
            "[]",
            "[]",
            0,
        ),
    )
    conn.commit()
    conn.close()

    report = paper_curation_audit.build_curation_report(
        vault_dir=vault,
        linkgraph_db=linkgraph_db,
    )

    actions = {row["path"]: row["action"] for row in report["curation_queue"]}
    assert actions["research/01_Literature_Index.md"] == "documented_scaffold"
    assert report["action_counts"] == {}


def test_format_markdown_prioritizes_actionable_sections() -> None:
    report = {
        "summary": {
            "paper_notes": 3,
            "graph_research_nodes": 2,
            "zero_concept": 1,
            "pdf_gap_papers": 2,
            "actionable_zero_concept": 1,
            "actionable_pdf_gap_papers": 1,
            "actionable_orphan_papers": 1,
            "orphan_notes": 1,
        },
        "status_counts": {"unread": 3},
        "zero_concept": [{"path": "research/zero.md", "title": "Zero", "status": "unread"}],
        "pdf_gaps": [
            {
                "path": "research/no_pdf.md",
                "title": "No PDF",
                "pdf_state": "no_pdf_path",
                "status": "unread",
                "acquisition_action": "retrieve_by_doi",
                "locator": "10.1000/example",
                "zotero_attachment_count": 0,
            },
        ],
        "orphan_notes": [{"path": "research/lonely.md", "title": "Lonely", "kind": "Paper"}],
        "action_counts": {"concept_link_candidate": 1},
        "curation_queue": [
            {
                "path": "research/zero.md",
                "title": "Zero",
                "action": "concept_link_candidate",
                "reasons": ["tags_match_existing_concepts"],
                "matching_concepts": ["concepts/Actuator.md"],
            }
        ],
    }

    markdown = paper_curation_audit.format_markdown(report)

    assert "## Zero-Concept Research Nodes" in markdown
    assert "research/zero.md" in markdown
    assert "## PDF Gaps" in markdown
    assert "no_pdf_path" in markdown
    assert "retrieve_by_doi" in markdown
    assert "## Orphan Notes" in markdown
    assert "## Curation Queue" in markdown
    assert "concept_link_candidate" in markdown
