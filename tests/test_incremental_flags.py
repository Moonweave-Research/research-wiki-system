import json
import time
from pathlib import Path


from scripts.ingest.wiki_pdf_cache import build_pdf_text_cache
from scripts.ingest.wiki_pdf_fact_extractor import extract_pdf_facts


def write_note(path: Path, *, alias: str, pdf_path: str = "") -> None:
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
Some abstract.
""",
        encoding="utf-8",
    )


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def test_cache_incremental_skips_unchanged(tmp_path: Path) -> None:
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
        extract_text=lambda path: "Abstract\nLow-voltage operation.\n",
    )

    cache_file = output_dir / "paper.txt"
    assert cache_file.exists()
    mtime_before = cache_file.stat().st_mtime

    # ensure cache is newer than pdf before second run
    time.sleep(0.01)

    report2 = build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=lambda path: "Different text.\n",
        incremental=True,
    )

    assert report2["summary"]["skipped_incremental"] >= 1
    assert cache_file.stat().st_mtime == mtime_before


def test_cache_incremental_backfills_pdf_metadata_on_skipped_rows(tmp_path: Path) -> None:
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
        extract_text=lambda path: "Abstract\nLow-voltage operation.\n",
    )
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0].pop("pdf_size")
    rows[0].pop("pdf_mtime_ns")
    write_manifest(manifest, rows)

    report2 = build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=lambda path: "Different text.\n",
        incremental=True,
    )

    row = report2["rows"][0]
    assert report2["summary"]["skipped_incremental"] == 1
    assert row["pdf_size"] == pdf.stat().st_size
    assert row["pdf_mtime_ns"] == pdf.stat().st_mtime_ns


def test_cache_incremental_re_extracts_new_paper(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    output_dir = tmp_path / "pdf_text"
    manifest = tmp_path / "manifest.jsonl"
    pdf_a = tmp_path / "paper_a.pdf"
    pdf_a.write_bytes(b"%PDF-1.4 fake A")
    pdf_b = tmp_path / "paper_b.pdf"
    pdf_b.write_bytes(b"%PDF-1.4 fake B")

    write_note(wiki / "research" / "paper_a.md", alias="PaperA", pdf_path=str(pdf_a))

    build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=lambda path: "Abstract\nLow-voltage operation.\n",
    )

    time.sleep(0.01)

    write_note(wiki / "research" / "paper_b.md", alias="PaperB", pdf_path=str(pdf_b))

    extracted_paths = []

    def track_extract(path: Path) -> str:
        extracted_paths.append(path)
        return "Abstract\nNew paper text.\n"

    report2 = build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=track_extract,
        incremental=True,
    )

    assert report2["summary"]["skipped_incremental"] >= 1
    assert pdf_b in extracted_paths
    assert pdf_a not in extracted_paths


def test_cache_incremental_drops_deleted_paper(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    output_dir = tmp_path / "pdf_text"
    manifest = tmp_path / "manifest.jsonl"
    pdf_a = tmp_path / "paper_a.pdf"
    pdf_a.write_bytes(b"%PDF-1.4 fake A")
    pdf_b = tmp_path / "paper_b.pdf"
    pdf_b.write_bytes(b"%PDF-1.4 fake B")

    write_note(wiki / "research" / "paper_a.md", alias="PaperA", pdf_path=str(pdf_a))
    write_note(wiki / "research" / "paper_b.md", alias="PaperB", pdf_path=str(pdf_b))

    build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=lambda path: "Abstract\nLow-voltage operation.\n",
    )

    time.sleep(0.01)

    # Remove paper_b note
    (wiki / "research" / "paper_b.md").unlink()

    report2 = build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=lambda path: "Abstract\nLow-voltage operation.\n",
        incremental=True,
    )

    paper_ids = [row["paper_id"] for row in report2["rows"]]
    assert "paper:paper_b" not in paper_ids
    assert "paper:paper_a" in paper_ids


def test_cache_full_rebuild_default_unchanged(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    output_dir = tmp_path / "pdf_text"
    manifest = tmp_path / "manifest.jsonl"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    write_note(wiki / "research" / "paper.md", alias="Paper", pdf_path=str(pdf))

    report = build_pdf_text_cache(
        wiki,
        output_dir=output_dir,
        manifest_path=manifest,
        extract_text=lambda path: "Abstract\nLow-voltage operation.\n",
    )

    assert "skipped_incremental" not in report["summary"]
    assert report["summary"]["papers"] == 1
    assert report["summary"]["cached"] == 1


def test_extractor_incremental_preserves_existing(tmp_path: Path) -> None:
    cache_file = tmp_path / "paper_a.txt"
    cache_file.write_text("Low-voltage operation reached 3V.", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [{"paper_id": "paper:paper_a", "cached": True, "cache_path": str(cache_file)}],
    )

    output = tmp_path / "pdf_facts.jsonl"
    facts_first = extract_pdf_facts(manifest, limit_per_paper=5)
    from legacy.graphrag.wiki_semantic_schema import write_facts_jsonl

    write_facts_jsonl(output, facts_first)
    assert len(facts_first) >= 1

    facts_incremental = extract_pdf_facts(
        manifest,
        limit_per_paper=5,
        incremental=True,
        existing_facts_path=output,
    )

    assert len(facts_incremental) == len(facts_first)
    first_ids = {f.fact_id for f in facts_first}
    incremental_ids = {f.fact_id for f in facts_incremental}
    assert first_ids == incremental_ids


def test_extractor_incremental_extracts_new(tmp_path: Path) -> None:
    cache_a = tmp_path / "paper_a.txt"
    cache_a.write_text("Low-voltage operation reached 3V.", encoding="utf-8")
    cache_b = tmp_path / "paper_b.txt"
    cache_b.write_text("Thermal expansion mismatch drives bending.", encoding="utf-8")

    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            {"paper_id": "paper:paper_a", "cached": True, "cache_path": str(cache_a)},
            {"paper_id": "paper:paper_b", "cached": True, "cache_path": str(cache_b)},
        ],
    )

    output = tmp_path / "pdf_facts.jsonl"
    facts_a_only = extract_pdf_facts(manifest, limit_per_paper=5, sample=1)
    from legacy.graphrag.wiki_semantic_schema import write_facts_jsonl

    write_facts_jsonl(output, facts_a_only)
    assert all(f.paper_id == "paper:paper_a" for f in facts_a_only)

    facts_incremental = extract_pdf_facts(
        manifest,
        limit_per_paper=5,
        incremental=True,
        existing_facts_path=output,
    )

    paper_ids = {f.paper_id for f in facts_incremental}
    assert "paper:paper_a" in paper_ids
    assert "paper:paper_b" in paper_ids
    a_facts = [f for f in facts_incremental if f.paper_id == "paper:paper_a"]
    assert {f.fact_id for f in a_facts} == {f.fact_id for f in facts_a_only}


def test_extractor_incremental_skips_malformed_rows(tmp_path):
    """Crash-recovery guard: a malformed row in pdf_facts.jsonl (paper_id only,
    other required fields missing) must NOT crash fact_from_dict — that takes
    out the whole refresh-graph pipeline. The row should be skipped and the
    paper re-extracted from the cache.
    """
    cache_a = tmp_path / "cache_a.txt"
    cache_a.write_text("Some abstract text about polymers and actuators.\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            {
                "paper_id": "paper:paper_a",
                "cached": True,
                "cache_path": str(cache_a),
                "note_path": "research/paper_a.md",
                "pdf_path": "/tmp/missing.pdf",
                "has_pdf_path": True,
                "pdf_exists": False,
                "text_chars": 100,
                "section_headings": [],
                "error": "",
            }
        ],
    )
    # Malformed existing row: paper_id only, all other SemanticFact fields missing
    facts_path = tmp_path / "pdf_facts.jsonl"
    facts_path.write_text(
        json.dumps({"paper_id": "paper:paper_a"}) + "\n",
        encoding="utf-8",
    )

    # Must NOT raise KeyError
    facts = extract_pdf_facts(
        manifest,
        limit_per_paper=5,
        incremental=True,
        existing_facts_path=facts_path,
    )
    assert isinstance(facts, list)
