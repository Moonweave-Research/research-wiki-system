import json
import sqlite3
import sys
from pathlib import Path

import pytest

import scripts.ingest.wiki_pdf_fact_extractor as pdf_extractor
from scripts.ingest.wiki_pdf_fact_extractor import (
    cache_signature_for,
    extract_facts_from_cached_text,
    extract_pdf_facts,
    extract_pdf_facts_with_report,
    main,
    normalize_pdf_text,
    read_cache_manifest,
    source_signature_for,
    write_pdf_facts_manifest,
)
from legacy.graphrag.wiki_semantic_schema import read_facts_jsonl


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def test_read_cache_manifest_keeps_cached_rows_only(tmp_path):
    cached = tmp_path / "cached.txt"
    cached.write_text("Low-voltage actuation reached 3V.\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            {"paper_id": "paper:cached", "cached": True, "cache_path": str(cached)},
            {"paper_id": "paper:skipped", "cached": False, "cache_path": ""},
        ],
    )

    rows = read_cache_manifest(manifest)

    assert rows == [{"paper_id": "paper:cached", "cache_path": str(cached)}]


def test_extract_facts_from_cached_text_emits_targeted_pdf_claims(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "The actuator achieved low-voltage operation at 3V. "
        "This background sentence has no target. "
        "Thermal expansion mismatch drives bending.",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:cao2019", text_path, limit_per_paper=3)

    assert [fact.fact_type for fact in facts] == ["Claim", "Mechanism"]
    assert facts[0].paper_id == "paper:cao2019"
    assert facts[0].fact_id.startswith("fact:cao2019:pdf_claim:")
    assert facts[0].evidence.startswith("PDF cache:")
    assert "metric:voltage" in facts[0].targets
    assert "mechanism:electrothermal_actuation" in facts[1].targets


def test_extract_facts_from_cached_text_skips_overlong_layout_noise(tmp_path):
    text_path = tmp_path / "paper.txt"
    noisy_sentence = " ".join(["voltage"] * 90) + "."
    text_path.write_text(
        noisy_sentence + " Low-voltage operation reached 3V.",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:cao2019", text_path, limit_per_paper=3)

    assert [fact.text for fact in facts] == ["Low-voltage operation reached 3V."]


def test_extract_facts_from_cached_text_skips_pdf_boilerplate(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "Corresponding author. Tel.: 1 650 123 4567. Fax: 1 650 123 4568. "
        "The effect of applied voltage was reported. "
        "Low-voltage operation reached 3V.",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:cao2019", text_path, limit_per_paper=3)

    assert [fact.text for fact in facts] == ["Low-voltage operation reached 3V."]


def test_extract_facts_from_cached_text_keeps_prose_when_markdown_has_bullets(tmp_path):
    text_path = tmp_path / "paper.md"
    text_path.write_text(
        "- [14] Reference entry without useful targets.\n"
        "- [15] Another reference entry.\n\n"
        "Liquid crystal polymers can form soft actuators for haptic feedback. "
        "The actuator operated at 50 V.",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:lcp", text_path, limit_per_paper=3)

    assert any("Liquid crystal polymers" in fact.text for fact in facts)
    assert any("metric:voltage" in fact.targets for fact in facts)


def test_extract_facts_from_cached_text_skips_reference_dump_units(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "Pelrine R 2000 High-speed electrically actuated elastomers Science 287 836 "
        "[12] and Kovacs G 2009 stacked dielectric elastomer actuator voltage [13]. "
        "Low-voltage operation reached 3V.",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:refs", text_path, limit_per_paper=5)

    assert [fact.text for fact in facts] == ["Low-voltage operation reached 3V."]


def test_extract_facts_from_cached_text_skips_equation_fragments(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "The displacement current ∂C/∂z drives the measured voltage response. "
        "Low-voltage operation reached 3V.",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:eqn", text_path, limit_per_paper=5)

    assert [fact.text for fact in facts] == ["Low-voltage operation reached 3V."]


def test_extract_facts_from_cached_text_keeps_single_citation_prose(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "The tensile strength reached 42 kPa at 290% strain under applied voltage [30].",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:single", text_path, limit_per_paper=5)

    assert any("42 kPa" in fact.text for fact in facts)


def test_normalize_pdf_text_replaces_common_ligatures():
    assert normalize_pdf_text("ﬁvefold force eﬀect ﬂexible") == "fivefold force effect flexible"


def test_extract_facts_from_cached_text_applies_ligature_normalization(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "These clutches improve by ﬁvefold with a 40-fold reduction in operating voltage.",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:levine2023", text_path)

    assert len(facts) == 1
    assert "fivefold" in facts[0].text
    assert "metric:voltage" in facts[0].targets


def test_extract_facts_from_cached_text_merges_adjacent_pdf_context(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "This allows a soft tunable applied electric field at 50 V um-1. "
        "The bilayer DE lens achieved 21% focal length variation at 500 V. "
        "Background text without targets.",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:lens", text_path, limit_per_paper=3)

    assert any("soft tunable" in fact.text and "bilayer DE lens" in fact.text for fact in facts)


def test_extract_facts_from_cached_text_prioritizes_pdf_mechanisms(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "Operational voltage reached 3 V. "
        "Applied voltage remained stable at 3 V. "
        "Voltage was measured every second. "
        "The supply voltage stayed constant. "
        "Low-voltage operation was observed. "
        "Another voltage trace was reported. "
        "Upon resistive heating, the soft actuator inflates by phase change of microliquid droplets.",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:microliquid", text_path, limit_per_paper=1)

    assert len(facts) == 1
    assert facts[0].fact_type == "Mechanism"
    assert "microliquid droplets" in facts[0].text
    assert "mechanism:electrothermal_actuation" in facts[0].targets


def test_extract_facts_from_cached_text_prioritizes_pdf_concepts(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "The actuator generated 10 N force. "
        "Applied voltage was 3 V. "
        "Dielectric elastomer actuators have become important smart material transducers.",
        encoding="utf-8",
    )

    facts = extract_facts_from_cached_text("paper:dea", text_path, limit_per_paper=1)

    assert len(facts) == 1
    assert "important smart material transducers" in facts[0].text
    assert "concept:Dielectric_Elastomer_Actuator" in facts[0].targets


def test_extract_pdf_facts_respects_limit_sample_and_missing_cache_files(tmp_path):
    first = tmp_path / "first.txt"
    first.write_text(
        "Low-voltage operation reached 3V. The actuator generated 3 mN force.",
        encoding="utf-8",
    )
    second = tmp_path / "second.txt"
    second.write_text("Thermal expansion mismatch drives bending.", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            {"paper_id": "paper:first", "cached": True, "cache_path": str(first)},
            {"paper_id": "paper:second", "cached": True, "cache_path": str(second)},
            {
                "paper_id": "paper:missing",
                "cached": True,
                "cache_path": str(tmp_path / "missing.txt"),
            },
        ],
    )

    facts = extract_pdf_facts(manifest, limit_per_paper=1, sample=1)

    assert len(facts) == 1
    assert facts[0].paper_id == "paper:first"
    assert facts[0].targets == ("metric:voltage",)


def test_extract_pdf_facts_incremental_reuses_only_when_source_signature_matches(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text("Low-voltage operation reached 3V.", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "pdf_facts.jsonl"
    facts_manifest = tmp_path / "pdf_facts_manifest.jsonl"
    write_manifest(
        manifest,
        [{"paper_id": "paper:cao2019", "cached": True, "cache_path": str(text_path)}],
    )
    first_facts = extract_pdf_facts(manifest, limit_per_paper=5)
    from legacy.graphrag.wiki_semantic_schema import write_facts_jsonl

    write_facts_jsonl(output, first_facts)
    write_pdf_facts_manifest(
        facts_manifest,
        [{"paper_id": "paper:cao2019", "source_sha256": source_signature_for(text_path)}],
    )

    reused = extract_pdf_facts(
        manifest,
        limit_per_paper=5,
        incremental=True,
        existing_facts_path=output,
        existing_facts_manifest_path=facts_manifest,
    )
    assert [fact.fact_id for fact in reused] == [fact.fact_id for fact in first_facts]

    text_path.write_text("Thermal expansion mismatch drives bending.", encoding="utf-8")
    refreshed = extract_pdf_facts(
        manifest,
        limit_per_paper=5,
        incremental=True,
        existing_facts_path=output,
        existing_facts_manifest_path=facts_manifest,
    )
    assert refreshed[0].fact_type == "Mechanism"
    assert refreshed[0].fact_id != first_facts[0].fact_id


def test_extract_pdf_facts_with_report_counts_reuse_extract_and_missing(tmp_path):
    reused_path = tmp_path / "reused.txt"
    reused_path.write_text("Low-voltage operation reached 3V.", encoding="utf-8")
    fresh_path = tmp_path / "fresh.txt"
    fresh_path.write_text("Thermal expansion mismatch drives bending.", encoding="utf-8")
    missing_path = tmp_path / "missing.txt"
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "pdf_facts.jsonl"
    facts_manifest = tmp_path / "pdf_facts_manifest.jsonl"
    write_manifest(
        manifest,
        [
            {"paper_id": "paper:reused", "cached": True, "cache_path": str(reused_path)},
            {"paper_id": "paper:fresh", "cached": True, "cache_path": str(fresh_path)},
            {"paper_id": "paper:missing", "cached": True, "cache_path": str(missing_path)},
        ],
    )
    prior_facts = extract_pdf_facts(manifest, limit_per_paper=5, sample=1)
    from legacy.graphrag.wiki_semantic_schema import write_facts_jsonl

    write_facts_jsonl(output, prior_facts)
    write_pdf_facts_manifest(
        facts_manifest,
        [{"paper_id": "paper:reused", "source_sha256": source_signature_for(reused_path)}],
    )

    facts, report = extract_pdf_facts_with_report(
        manifest,
        limit_per_paper=5,
        incremental=True,
        existing_facts_path=output,
        existing_facts_manifest_path=facts_manifest,
    )

    assert {fact.paper_id for fact in facts} == {"paper:reused", "paper:fresh"}
    assert report["papers"] == {
        "total": 3,
        "reused": 1,
        "extracted": 1,
        "missing_cache": 1,
        "irrelevant": 0,
        "facts_reused": 1,
        "facts_extracted": 1,
    }
    assert report["facts"] == 2
    assert report["slowest_papers"]
    assert {row["status"] for row in report["slowest_papers"]} <= {
        "reused",
        "extracted",
        "missing_cache",
    }


def test_read_cache_manifest_accepts_pdf_markdown_manifest(tmp_path):
    markdown = tmp_path / "paper.md"
    markdown.write_text("## Abstract\nLow-voltage actuation reached 3 V.", encoding="utf-8")
    manifest = tmp_path / "pdf_markdown_manifest.jsonl"
    write_manifest(
        manifest,
        [{"paper_id": "paper:paper", "cached": True, "markdown_path": str(markdown)}],
    )

    assert read_cache_manifest(manifest) == [
        {"paper_id": "paper:paper", "cache_path": str(markdown)}
    ]


def test_pdf_fact_ids_are_stable_when_sentence_order_changes(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(
        "Low-voltage operation reached 3V. The actuator generated 3 mN force.",
        encoding="utf-8",
    )
    second.write_text(
        "The actuator generated 3 mN force. Low-voltage operation reached 3V.",
        encoding="utf-8",
    )

    first_by_text = {
        fact.text: fact.fact_id for fact in extract_facts_from_cached_text("paper:test", first)
    }
    second_by_text = {
        fact.text: fact.fact_id for fact in extract_facts_from_cached_text("paper:test", second)
    }

    assert first_by_text == second_by_text


def test_cli_writes_pdf_facts_jsonl(monkeypatch, capsys, tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text("Low-voltage operation reached 3V.", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "pdf_facts.jsonl"
    write_manifest(
        manifest,
        [{"paper_id": "paper:cao2019", "cached": True, "cache_path": str(text_path)}],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_pdf_fact_extractor.py",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--limit-per-paper",
            "2",
            "--sample",
            "1",
            "--json",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["facts"] == 1
    assert payload["facts_written"] == str(output)
    assert payload["papers"] == {
        "total": 1,
        "reused": 0,
        "extracted": 1,
        "missing_cache": 0,
        "irrelevant": 0,
        "facts_reused": 0,
        "facts_extracted": 1,
    }
    assert payload["slowest_papers"][0]["paper_id"] == "paper:cao2019"
    assert payload["slowest_papers"][0]["status"] == "extracted"
    facts = read_facts_jsonl(output)
    assert len(facts) == 1
    assert facts[0].paper_id == "paper:cao2019"
    assert json.loads(output.read_text(encoding="utf-8"))["fact_type"] == "Claim"
    facts_manifest = tmp_path / "pdf_facts_manifest.jsonl"
    facts_manifest_rows = [
        json.loads(line)
        for line in facts_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert facts_manifest_rows == [
        {
            "cache_path": str(text_path),
            "paper_id": "paper:cao2019",
            "source_sha256": cache_signature_for(text_path),
        }
    ]


def test_cli_skips_rewriting_unchanged_pdf_facts_outputs(monkeypatch, capsys, tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text("Low-voltage operation reached 3V.", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "pdf_facts.jsonl"
    write_manifest(
        manifest,
        [{"paper_id": "paper:cao2019", "cached": True, "cache_path": str(text_path)}],
    )
    argv = [
        "wiki_pdf_fact_extractor.py",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--limit-per-paper",
        "2",
        "--json",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    first_payload = json.loads(capsys.readouterr().out)

    monkeypatch.setattr(sys, "argv", argv)
    main()
    second_payload = json.loads(capsys.readouterr().out)

    assert first_payload["files"] == {
        "facts_written": True,
        "manifest_written": True,
    }
    assert second_payload["files"] == {
        "facts_written": False,
        "manifest_written": False,
    }
    assert read_facts_jsonl(output)[0].paper_id == "paper:cao2019"


def test_cli_incremental_fast_path_skips_unchanged_zero_fact_papers(monkeypatch, capsys, tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text("This background sentence has no indexed target.", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "pdf_facts.jsonl"
    write_manifest(
        manifest,
        [{"paper_id": "paper:zero", "cached": True, "cache_path": str(text_path)}],
    )
    argv = [
        "wiki_pdf_fact_extractor.py",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--incremental",
        "--json",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    first_payload = json.loads(capsys.readouterr().out)
    assert first_payload["facts"] == 0
    assert first_payload["files"]["manifest_written"] is True

    def fail_extract(*args, **kwargs):
        raise AssertionError("unchanged incremental run should not re-extract")

    monkeypatch.setattr(pdf_extractor, "extract_facts_from_cached_text", fail_extract)
    monkeypatch.setattr(sys, "argv", argv)
    main()
    second_payload = json.loads(capsys.readouterr().out)

    assert second_payload["unchanged"] is True
    assert second_payload["facts"] == 0
    assert second_payload["files"] == {
        "facts_written": False,
        "manifest_written": False,
    }


def test_cli_incremental_fast_path_does_not_hash_unchanged_cache_content(
    monkeypatch,
    capsys,
    tmp_path,
):
    text_path = tmp_path / "paper.txt"
    text_path.write_text("This background sentence has no indexed target.", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "pdf_facts.jsonl"
    write_manifest(
        manifest,
        [{"paper_id": "paper:zero", "cached": True, "cache_path": str(text_path)}],
    )
    argv = [
        "wiki_pdf_fact_extractor.py",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--incremental",
        "--json",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    capsys.readouterr()

    def fail_hash(path):
        raise AssertionError(f"unchanged fast path should not content-hash {path}")

    monkeypatch.setattr(pdf_extractor, "source_signature_for", fail_hash)
    monkeypatch.setattr(sys, "argv", argv)
    main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["unchanged"] is True


def test_incremental_reuse_with_cache_signature_does_not_hash_content(monkeypatch, tmp_path):
    reused_path = tmp_path / "reused.txt"
    reused_path.write_text("Low-voltage operation reached 3V.", encoding="utf-8")
    fresh_path = tmp_path / "fresh.txt"
    fresh_path.write_text("Thermal expansion mismatch drives bending.", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "pdf_facts.jsonl"
    facts_manifest = tmp_path / "pdf_facts_manifest.jsonl"
    write_manifest(
        manifest,
        [
            {"paper_id": "paper:reused", "cached": True, "cache_path": str(reused_path)},
            {"paper_id": "paper:fresh", "cached": True, "cache_path": str(fresh_path)},
        ],
    )
    prior_facts = extract_pdf_facts(manifest, limit_per_paper=5, sample=1)
    from legacy.graphrag.wiki_semantic_schema import write_facts_jsonl

    write_facts_jsonl(output, prior_facts)
    write_pdf_facts_manifest(
        facts_manifest,
        [{"paper_id": "paper:reused", "source_sha256": cache_signature_for(reused_path)}],
    )

    def fail_hash(path):
        raise AssertionError(f"cache-signature reuse should not content-hash {path}")

    monkeypatch.setattr(pdf_extractor, "source_signature_for", fail_hash)

    facts, report = extract_pdf_facts_with_report(
        manifest,
        limit_per_paper=5,
        incremental=True,
        existing_facts_path=output,
        existing_facts_manifest_path=facts_manifest,
    )

    assert {fact.paper_id for fact in facts} == {"paper:reused", "paper:fresh"}
    assert report["papers"]["reused"] == 1
    assert report["papers"]["extracted"] == 1


def test_load_irrelevant_paper_ids_uses_default_linkgraph_index(monkeypatch, tmp_path):
    db = tmp_path / ".graph_index" / "linkgraph.sqlite"
    db.parent.mkdir()
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE notes (path TEXT, status TEXT)")
    conn.executemany(
        "INSERT INTO notes(path, status) VALUES (?, ?)",
        [
            ("research/keep.md", "read"),
            ("research/skip.md", "irrelevant"),
            ("concepts/skip.md", "irrelevant"),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.chdir(tmp_path)

    assert pdf_extractor.load_irrelevant_paper_ids(Path("wiki")) == {"paper:skip"}


def test_main_rejects_non_positive_limit(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_pdf_fact_extractor.py",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--limit-per-paper",
            "0",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    assert "limit-per-paper must be positive" in capsys.readouterr().out


def test_main_rejects_malformed_manifest_without_traceback(monkeypatch, capsys, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_pdf_fact_extractor.py",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "facts.jsonl"),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    output = capsys.readouterr().out
    assert "invalid cached manifest row" in output
    assert "Traceback" not in output
