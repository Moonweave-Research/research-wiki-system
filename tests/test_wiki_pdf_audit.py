import json
import sys

import pytest

from scripts.ingest.wiki_pdf_audit import audit_pdfs, detect_section_headings, main


def write_note(path, *, alias, abstract="Some abstract.", pdf_path=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""---
aliases: [{alias}]
domain: research
category: materials
tags: []
sources: []
pdf_path: {pdf_path}
---
# {alias}

## Abstract
{abstract}
""",
        encoding="utf-8",
    )


def test_detect_section_headings_matches_common_pdf_headings():
    text = """
Abstract
This paper studies actuators.
1. Introduction
Methods
Results and Discussion
Conclusion
References
"""

    assert detect_section_headings(text) == [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Conclusion",
    ]


def test_audit_pdfs_reports_missing_existing_and_readable_pdfs(tmp_path):
    wiki = tmp_path / "wiki"
    existing_pdf = tmp_path / "existing.pdf"
    existing_pdf.write_bytes(b"%PDF-1.4 fake")
    missing_pdf = tmp_path / "missing.pdf"
    write_note(wiki / "research" / "readable.md", alias="Readable", pdf_path=str(existing_pdf))
    write_note(wiki / "research" / "missing.md", alias="Missing", pdf_path=str(missing_pdf))
    write_note(wiki / "research" / "no_pdf.md", alias="No PDF", pdf_path="")

    def extract_text(path):
        if path == existing_pdf:
            return "Abstract\nBody text\nMethods\nResult text\nConclusion\n"
        raise AssertionError(f"Unexpected extraction path: {path}")

    report = audit_pdfs(wiki, extract_text=extract_text)

    assert report["summary"] == {
        "papers": 3,
        "with_pdf_path": 2,
        "existing_pdfs": 1,
        "readable_pdfs": 1,
        "with_section_headings": 1,
        "missing_pdfs": 1,
    }
    rows = {row["paper_id"]: row for row in report["papers"]}
    assert rows["paper:readable"]["pdf_exists"] is True
    assert rows["paper:readable"]["readable"] is True
    assert rows["paper:readable"]["text_chars"] > 0
    assert rows["paper:readable"]["section_headings"] == ["Abstract", "Methods", "Conclusion"]
    assert rows["paper:missing"]["pdf_exists"] is False
    assert rows["paper:no_pdf"]["has_pdf_path"] is False


def test_audit_pdfs_records_text_extraction_errors(tmp_path):
    wiki = tmp_path / "wiki"
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    write_note(wiki / "research" / "broken.md", alias="Broken", pdf_path=str(pdf))

    def extract_text(path):
        raise RuntimeError("cannot extract")

    report = audit_pdfs(wiki, extract_text=extract_text)

    row = report["papers"][0]
    assert row["paper_id"] == "paper:broken"
    assert row["pdf_exists"] is True
    assert row["readable"] is False
    assert row["error"] == "cannot extract"
    assert report["summary"]["readable_pdfs"] == 0


def test_main_prints_json_report(monkeypatch, capsys, tmp_path):
    wiki = tmp_path / "wiki"
    write_note(wiki / "research" / "no_pdf.md", alias="No PDF", pdf_path="")
    monkeypatch.setattr(
        sys,
        "argv",
        ["wiki_pdf_audit.py", "--wiki-dir", str(wiki), "--json"],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["papers"] == 1
    assert payload["papers"][0]["paper_id"] == "paper:no_pdf"


def test_main_rejects_non_positive_sample(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["wiki_pdf_audit.py", "--wiki-dir", str(tmp_path / "wiki"), "--sample", "0"],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    assert "sample must be positive" in capsys.readouterr().out
