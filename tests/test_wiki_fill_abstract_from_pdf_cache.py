import json
from pathlib import Path

from scripts.ingest.wiki_fill_abstract_from_pdf_cache import fill_missing_abstracts, extract_cached_abstract
from legacy.graphrag.wiki_graph import parse_frontmatter


def write_paper(path: Path, *, alias: str, abstract: str = "", pdf_path: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """---
aliases: [""" + alias + """ ]
domain: research
category: materials
tags: []
sources: []
pdf_path: """ + pdf_path + """
""" + (f"abstract: {abstract}\n" if abstract else "") + """---
# """ + alias + """
 
- **Authors:** Author
- **Year:** 2024
""",
        encoding="utf-8",
    )


def write_cache(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_extract_cached_abstract_parses_inline_abstract_line() -> None:
    text = """
ABSTRACT: This article introduces a compact electromechanical material.
INTRODUCTION
More content.
"""

    assert (
        extract_cached_abstract(text)
        == "This article introduces a compact electromechanical material."
    )


def test_extract_cached_abstract_parses_non_colon_heading_and_stops_before_next_section() -> None:
    text = """
         ARTICLE INFO ABSTRACT   Understanding of the growth mechanism is discussed.
Some follow up explanation text.
Methods
The rest should be excluded.
"""

    assert extract_cached_abstract(text).startswith("Understanding of the growth mechanism")
    assert "The rest should be excluded" not in extract_cached_abstract(text)


def test_extract_cached_abstract_skips_download_artifact_lines() -> None:
    text = """
Abstract
The disordered transformation of the ordered aligned polar liquid crystal Downloaded via CHUNGNAM NATL UNIV on August 30, 2023 at 07:03:20 (UTC). molecules in LCEs under external fields.
Introduction
"""

    assert (
        extract_cached_abstract(text)
        == "The disordered transformation of the ordered aligned polar liquid crystal molecules in LCEs under external fields."
    )


def test_extract_cached_abstract_stops_at_keyword_or_received_boundaries() -> None:
    text = """
ABSTRACT This is a clean abstract sentence.
Another sentence follows.
키워드 : keyword1, keyword2
Received 1 January 2024
"""

    assert extract_cached_abstract(text) == "This is a clean abstract sentence. Another sentence follows."


def test_fill_missing_abstracts_updates_only_missing_without_abstract(tmp_path):
    wiki = tmp_path / "wiki"
    cache_root = tmp_path / ".graph_index"
    write_paper(
        wiki / "research" / "with_cache.md",
        alias="with_cache",
        pdf_path=str(tmp_path / "paper.pdf"),
    )
    write_paper(
        wiki / "research" / "with_abstract.md",
        alias="with_abstract",
        pdf_path=str(tmp_path / "paper2.pdf"),
        abstract="Already exists",
    )

    cache_file = cache_root / "with_cache.txt"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    write_cache(
        cache_file,
        """
ABSTRACT: This is a synthesized abstract from PDF cache.
Introduction
""",
    )

    manifest = cache_root / "pdf_text_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
            json.dumps(
                {
                    "paper_id": "paper:with_cache",
                    "cache_path": str(cache_file),
                    "cached": True,
                }
            ),
            json.dumps(
                {
                    "paper_id": "paper:with_abstract",
                    "cache_path": str(cache_file),
                    "cached": True,
                }
            ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = fill_missing_abstracts(
        wiki,
        manifest_path=manifest,
        dry_run=False,
    )

    assert report["updated"] == 1
    assert report["already_had_abstract"] == 1

    with_note = parse_frontmatter((wiki / "research" / "with_cache.md").read_text(encoding="utf-8"))[0]
    assert with_note["abstract"] == "This is a synthesized abstract from PDF cache."
