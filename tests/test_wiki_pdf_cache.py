import json
import sys
from pathlib import Path

import pytest

import scripts.ingest.wiki_pdf_cache as pdf_cache
from scripts.ingest.wiki_pdf_cache import build_pdf_text_cache, main


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


def test_build_pdf_text_cache_writes_text_files_and_manifest(tmp_path):
    wiki = tmp_path / "wiki"
    output_dir = tmp_path / ".graph_index" / "pdf_text"
    manifest = tmp_path / ".graph_index" / "pdf_text_manifest.jsonl"
    readable_pdf = tmp_path / "readable.pdf"
    readable_pdf.write_bytes(b"%PDF-1.4 fake")
    missing_pdf = tmp_path / "missing.pdf"
    write_note(wiki / "research" / "readable.md", alias="Readable", pdf_path=str(readable_pdf))
    write_note(wiki / "research" / "missing.md", alias="Missing", pdf_path=str(missing_pdf))
    write_note(wiki / "research" / "no_pdf.md", alias="No PDF", pdf_path="")

    def extract_text(path):
        if path == readable_pdf:
            return "Abstract\nCached text.\nMethods\n"
        raise AssertionError(f"Unexpected extraction path: {path}")

    report = build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=extract_text,
    )

    assert report["summary"] == {
        "papers": 3,
        "cached": 1,
        "skipped": 2,
        "missing_pdfs": 1,
        "extraction_errors": 0,
    }
    cache_file = output_dir / "readable.txt"
    assert cache_file.read_text(encoding="utf-8") == "Abstract\nCached text.\nMethods\n"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [row["paper_id"] for row in rows] == [
        "paper:missing",
        "paper:no_pdf",
        "paper:readable",
    ]
    readable = rows[2]
    assert readable["cached"] is True
    assert readable["cache_path"] == str(cache_file)
    assert readable["pdf_size"] == readable_pdf.stat().st_size
    assert readable["pdf_mtime_ns"] == readable_pdf.stat().st_mtime_ns
    assert readable["text_chars"] == len("Abstract\nCached text.\nMethods\n".strip())
    assert readable["section_headings"] == ["Abstract", "Methods"]


def test_build_pdf_text_cache_fast_path_reuses_manifest_when_vault_unchanged(
    monkeypatch,
    tmp_path,
):
    wiki = tmp_path / "wiki"
    output_dir = tmp_path / ".graph_index" / "pdf_text"
    manifest = tmp_path / ".graph_index" / "pdf_text_manifest.jsonl"
    linkgraph_meta = tmp_path / ".graph_index" / "linkgraph.sqlite.meta.json"
    readable_pdf = tmp_path / "readable.pdf"
    readable_pdf.write_bytes(b"%PDF-1.4 fake")
    write_note(wiki / "research" / "readable.md", alias="Readable", pdf_path=str(readable_pdf))

    build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=lambda path: "Abstract\nCached text.\n",
    )
    signature = [["research/readable.md", 1, 2]]
    linkgraph_meta.write_text(
        json.dumps({"signature": signature, "notes": 1, "links": 0}),
        encoding="utf-8",
    )
    pdf_cache.pdf_text_cache_meta_path(manifest).write_text(
        json.dumps({"vault_signature": signature}),
        encoding="utf-8",
    )

    def fail_paper_notes(path: Path):
        raise AssertionError(f"unchanged fast path should not parse notes under {path}")

    monkeypatch.setattr(pdf_cache, "paper_notes", fail_paper_notes)

    report = build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=lambda path: "Changed text.\n",
        incremental=True,
        linkgraph_meta_path=linkgraph_meta,
    )

    assert report["summary"]["fast_path"] is True
    assert report["summary"]["skipped_incremental"] == 1
    assert report["rows"][0]["paper_id"] == "paper:readable"


def test_build_pdf_text_cache_respects_sample_limit(tmp_path):
    wiki = tmp_path / "wiki"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    write_note(wiki / "research" / "first.md", alias="First", pdf_path=str(pdf))
    write_note(wiki / "research" / "second.md", alias="Second", pdf_path=str(pdf))

    report = build_pdf_text_cache(
        wiki,
        output_dir=tmp_path / "pdf_text",
        manifest_path=tmp_path / "manifest.jsonl",
        extract_text=lambda path: "Abstract\nText\n",
        sample=1,
    )

    assert report["summary"]["papers"] == 1
    assert report["summary"]["cached"] == 1


def test_build_pdf_text_cache_records_extraction_errors(tmp_path):
    wiki = tmp_path / "wiki"
    output_dir = tmp_path / "pdf_text"
    manifest = tmp_path / "manifest.jsonl"
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    write_note(wiki / "research" / "broken.md", alias="Broken", pdf_path=str(pdf))

    def extract_text(path):
        raise RuntimeError("cannot extract")

    report = build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=extract_text,
    )

    row = report["rows"][0]
    assert row["paper_id"] == "paper:broken"
    assert row["cached"] is False
    assert row["error"] == "cannot extract"
    assert report["summary"]["extraction_errors"] == 1
    assert not (output_dir / "broken.txt").exists()


def test_build_pdf_text_cache_removes_stale_cache_when_rerun_skips_paper(tmp_path):
    wiki = tmp_path / "wiki"
    output_dir = tmp_path / "pdf_text"
    manifest = tmp_path / "manifest.jsonl"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    write_note(wiki / "research" / "paper.md", alias="Paper", pdf_path=str(pdf))

    build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=lambda path: "Abstract\nCached text\n",
    )
    cache_file = output_dir / "paper.txt"
    assert cache_file.exists()

    report = build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=lambda path: "",
    )

    assert report["rows"][0]["cached"] is False
    assert report["rows"][0]["error"] == "empty text"
    assert not cache_file.exists()


def test_main_prints_json_report(monkeypatch, capsys, tmp_path):
    wiki = tmp_path / "wiki"
    write_note(wiki / "research" / "no_pdf.md", alias="No PDF", pdf_path="")
    output_dir = tmp_path / "pdf_text"
    manifest = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_pdf_cache.py",
            "--wiki-dir",
            str(wiki),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
            "--json",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["papers"] == 1
    assert payload["summary"]["cached"] == 0
    assert manifest.exists()


def test_main_prints_summary_only_json_report(monkeypatch, capsys, tmp_path):
    wiki = tmp_path / "wiki"
    write_note(wiki / "research" / "no_pdf.md", alias="No PDF", pdf_path="")
    manifest = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_pdf_cache.py",
            "--wiki-dir",
            str(wiki),
            "--manifest",
            str(manifest),
            "--json",
            "--summary-only",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "summary": {
            "cached": 0,
            "extraction_errors": 0,
            "missing_pdfs": 0,
            "papers": 1,
            "skipped": 1,
        }
    }


def test_main_rejects_non_positive_sample(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_pdf_cache.py",
            "--wiki-dir",
            str(tmp_path / "wiki"),
            "--sample",
            "0",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    assert "sample must be positive" in capsys.readouterr().out
