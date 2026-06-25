import json
import sys
from pathlib import Path

import pytest

from scripts.ingest.wiki_pdf_markdown_quality import (
    evaluate_providers,
    evaluate_manifest,
    main,
    normalize_paper_id,
    parse_providers,
    score_markdown,
    select_manifest_rows,
)


def write_jsonl(path: Path, rows: list[dict], *, malformed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if malformed:
            handle.write("{not-json}\n")


def manifest_row(
    paper_id: str,
    *,
    markdown_path: str = "",
    cached: bool = True,
    has_pdf_path: bool = True,
    pdf_exists: bool = True,
    error: str = "",
    provider: str = "plain",
) -> dict:
    return {
        "paper_id": paper_id,
        "note_path": f"research/{paper_id.removeprefix('paper:')}.md",
        "pdf_path": f"/tmp/{paper_id.removeprefix('paper:')}.pdf" if has_pdf_path else "",
        "markdown_path": markdown_path,
        "cached": cached,
        "has_pdf_path": has_pdf_path,
        "pdf_exists": pdf_exists,
        "provider": provider,
        "error": error,
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_score_markdown_rewards_structure_and_penalizes_noise() -> None:
    good = score_markdown(
        """# Paper

## Abstract

Dielectric elastomer actuators convert electrical energy into mechanical deformation.

## Results

| Voltage | Strain |
| 1 kV | 10% |

## References

[1] Example.
"""
    )
    noisy = score_markdown("A\nB\nC\n" + "bad-\nbreak\n" * 12 + "�" * 8)

    assert good["score"] > noisy["score"]
    assert good["has_abstract"] is True
    assert good["has_references"] is True
    assert good["heading_count"] == 3
    assert good["table_like_lines"] == 2
    assert noisy["replacement_char_count"] == 8
    assert noisy["hyphen_break_count"] == 12
    assert "replacement characters" in noisy["warnings"]


def test_evaluate_manifest_scores_rows_and_writes_reports(tmp_path: Path) -> None:
    good_md = tmp_path / "good.md"
    good_md.write_text("# Good\n\n## Abstract\n\nUseful content.\n\n## References\n\n[1]", encoding="utf-8")
    empty_md = tmp_path / "empty.md"
    empty_md.write_text("   \n", encoding="utf-8")
    missing_md = tmp_path / "missing.md"
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "quality.jsonl"
    report = tmp_path / "quality.md"
    write_jsonl(
        manifest,
        [
            manifest_row("paper:good", markdown_path=str(good_md)),
            manifest_row("paper:empty", markdown_path=str(empty_md)),
            manifest_row("paper:missing-md", markdown_path=str(missing_md)),
            manifest_row("paper:no-pdf", cached=False, has_pdf_path=False),
            manifest_row("paper:missing-pdf", cached=False, pdf_exists=False, error="missing pdf"),
            manifest_row("paper:provider-error", cached=False, error="provider failed"),
        ],
        malformed=True,
    )

    result = evaluate_manifest(
        manifest,
        output_jsonl=output,
        output_report=report,
        low_score=80,
    )

    assert result["summary"]["rows"] == 7
    assert result["summary"]["scored"] == 1
    assert result["summary"]["malformed_manifest_rows"] == 1
    rows = read_jsonl(output)
    assert {row["status"] for row in rows} == {
        "scored",
        "empty_markdown",
        "missing_markdown",
        "no_pdf_path",
        "missing_pdf",
        "provider_error",
        "malformed_manifest",
    }
    good = next(row for row in rows if row["paper_id"] == "paper:good")
    assert good["provider"] == "plain"
    assert good["score"] >= 80
    assert "paper:missing-md" in report.read_text(encoding="utf-8")


def test_manifest_row_selection_supports_stride_and_explicit_paper_ids() -> None:
    rows = [manifest_row(f"paper:p{i}") for i in range(10)]

    assert [row["paper_id"] for row in select_manifest_rows(rows, sample=4)] == [
        "paper:p0",
        "paper:p1",
        "paper:p2",
        "paper:p3",
    ]
    assert [row["paper_id"] for row in select_manifest_rows(rows, sample=4, sample_strategy="stride")] == [
        "paper:p0",
        "paper:p3",
        "paper:p6",
        "paper:p9",
    ]
    assert [row["paper_id"] for row in select_manifest_rows(rows, paper_ids={"p2", "paper:p8"})] == [
        "paper:p2",
        "paper:p8",
    ]


def test_normalize_paper_id_accepts_citekey_or_node_id() -> None:
    assert normalize_paper_id("paper:songActuator2024") == "paper:songActuator2024"
    assert normalize_paper_id("songActuator2024") == "paper:songActuator2024"


def test_quality_main_prints_json_and_validates_threshold(monkeypatch, capsys, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [manifest_row("paper:no-pdf", cached=False, has_pdf_path=False)])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_pdf_markdown_quality.py",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "quality.jsonl"),
            "--report",
            str(tmp_path / "quality.md"),
            "--json",
        ],
    )

    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["rows"] == 1

    monkeypatch.setattr(sys, "argv", ["wiki_pdf_markdown_quality.py", "--low-score", "101"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    assert "low-score must be between 0 and 100" in capsys.readouterr().out


def test_parse_providers_validates_names() -> None:
    assert parse_providers("plain,pymupdf4llm") == ["plain", "pymupdf4llm"]

    with pytest.raises(ValueError, match="unsupported provider"):
        parse_providers("plain,unknown")


def test_evaluate_providers_uses_isolated_outputs_and_selected_ids(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    bakeoff_dir = tmp_path / ".graph_index" / "pdf_markdown_bakeoff"
    output = tmp_path / ".graph_index" / "quality.jsonl"
    report = tmp_path / ".graph_index" / "quality.md"
    calls: list[dict] = []

    def fake_build_cache(**kwargs):
        calls.append(kwargs)
        provider = kwargs["provider"]
        output_dir = kwargs["output_dir"]
        manifest_path = kwargs["manifest_path"]
        markdown_path = output_dir / "paper.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            f"# {provider}\n\n## Abstract\n\nProvider output.\n\n## References\n\n[1]",
            encoding="utf-8",
        )
        write_jsonl(
            manifest_path,
            [
                manifest_row(
                    "paper:paper",
                    markdown_path=str(markdown_path),
                    provider=provider,
                )
            ],
        )
        return {"summary": {"cached": 1}, "rows": []}

    result = evaluate_providers(
        wiki,
        providers=["plain", "pymupdf4llm"],
        bakeoff_dir=bakeoff_dir,
        output_jsonl=output,
        output_report=report,
        paper_ids={"paper"},
        build_cache=fake_build_cache,
    )

    assert [call["provider"] for call in calls] == ["plain", "pymupdf4llm"]
    assert calls[0]["output_dir"] == bakeoff_dir / "plain"
    assert calls[1]["manifest_path"] == bakeoff_dir / "pymupdf4llm_manifest.jsonl"
    assert calls[0]["paper_ids"] == {"paper:paper"}
    assert result["summary"]["providers"] == ["plain", "pymupdf4llm"]
    assert result["summary"]["provider_summaries"]["plain"]["scored"] == 1
    assert len(read_jsonl(output)) == 2
