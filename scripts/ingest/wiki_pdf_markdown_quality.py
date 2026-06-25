import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from scripts.ingest.wiki_pdf_audit import paper_notes
    from scripts.ingest.wiki_pdf_markdown import SUPPORTED_PROVIDERS, build_pdf_markdown_cache
except ModuleNotFoundError:
    from wiki_pdf_audit import paper_notes  # type: ignore[no-redef]
    from wiki_pdf_markdown import SUPPORTED_PROVIDERS, build_pdf_markdown_cache  # type: ignore[no-redef]


@dataclass
class QualityRow:
    paper_id: str
    provider: str
    status: str
    score: int | None = None
    markdown_path: str = ""
    note_path: str = ""
    error: str = ""
    markdown_chars: int = 0
    word_count: int = 0
    line_count: int = 0
    heading_count: int = 0
    has_abstract: bool = False
    has_references: bool = False
    table_like_lines: int = 0
    replacement_char_count: int = 0
    single_char_line_ratio: float = 0.0
    long_line_ratio: float = 0.0
    duplicate_line_ratio: float = 0.0
    hyphen_break_count: int = 0
    warnings: list[str] = field(default_factory=list)


def normalize_paper_id(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("paper:"):
        return stripped
    return f"paper:{stripped}"


def score_markdown(markdown: str) -> dict[str, Any]:
    text = markdown.strip()
    lines = [line.rstrip() for line in text.splitlines()]
    nonempty_lines = [line.strip() for line in lines if line.strip()]
    lower = text.lower()
    words = re.findall(r"\b[\w-]+\b", text, flags=re.UNICODE)
    heading_count = sum(1 for line in nonempty_lines if re.match(r"^#{2,6}\s+\S", line))
    table_like_lines = sum(1 for line in nonempty_lines if "|" in line)
    replacement_char_count = text.count("�")
    single_char_lines = sum(1 for line in nonempty_lines if len(line) == 1)
    long_lines = sum(1 for line in nonempty_lines if len(line) > 180)
    duplicate_line_count = sum(count - 1 for count in Counter(nonempty_lines).values() if count > 1)
    line_count = len(nonempty_lines)
    hyphen_break_count = len(re.findall(r"-\n\s*[a-zA-Z]", text))

    single_char_line_ratio = single_char_lines / line_count if line_count else 0.0
    long_line_ratio = long_lines / line_count if line_count else 0.0
    duplicate_line_ratio = duplicate_line_count / line_count if line_count else 0.0
    has_abstract = bool(re.search(r"(^|\n)#{1,6}\s*abstract\b|\babstract\b", lower))
    has_references = bool(re.search(r"(^|\n)#{1,6}\s*references\b|\breferences\b", lower))

    score = 65
    score += min(15, heading_count * 5)
    if has_abstract:
        score += 10
    if has_references:
        score += 5
    if table_like_lines:
        score += min(5, table_like_lines)
    score += min(10, len(words) // 250)

    score -= min(20, replacement_char_count * 2)
    score -= min(20, hyphen_break_count)
    score -= round(single_char_line_ratio * 30)
    score -= round(long_line_ratio * 20)
    score -= round(duplicate_line_ratio * 20)
    if len(words) < 30:
        score -= 10

    warnings: list[str] = []
    if replacement_char_count:
        warnings.append("replacement characters")
    if single_char_line_ratio > 0.25:
        warnings.append("many single-character lines")
    if long_line_ratio > 0.30:
        warnings.append("many very long lines")
    if duplicate_line_ratio > 0.20:
        warnings.append("duplicate line noise")
    if hyphen_break_count > 5:
        warnings.append("hyphenated line-break noise")
    if len(words) < 30:
        warnings.append("very short markdown")

    return {
        "score": max(0, min(100, score)),
        "markdown_chars": len(text),
        "word_count": len(words),
        "line_count": line_count,
        "heading_count": heading_count,
        "has_abstract": has_abstract,
        "has_references": has_references,
        "table_like_lines": table_like_lines,
        "replacement_char_count": replacement_char_count,
        "single_char_line_ratio": round(single_char_line_ratio, 4),
        "long_line_ratio": round(long_line_ratio, 4),
        "duplicate_line_ratio": round(duplicate_line_ratio, 4),
        "hyphen_break_count": hyphen_break_count,
        "warnings": warnings,
    }


def read_manifest_rows(path: Path) -> tuple[list[dict[str, Any]], list[QualityRow]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    malformed: list[QualityRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                malformed.append(
                    QualityRow(
                        paper_id=f"__malformed__:{line_no}",
                        provider="manifest",
                        status="malformed_manifest",
                        error=str(exc),
                        warnings=["malformed manifest row"],
                    )
                )
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                malformed.append(
                    QualityRow(
                        paper_id=f"__malformed__:{line_no}",
                        provider="manifest",
                        status="malformed_manifest",
                        error="manifest row is not an object",
                        warnings=["malformed manifest row"],
                    )
                )
    return rows, malformed


def parse_providers(value: str) -> list[str]:
    providers = [item.strip() for item in value.split(",") if item.strip()]
    supported = set(SUPPORTED_PROVIDERS)
    unsupported = [provider for provider in providers if provider not in supported]
    if unsupported:
        raise ValueError(f"unsupported provider: {', '.join(unsupported)}")
    return providers


def select_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    sample: int | None = None,
    sample_strategy: str = "prefix",
    paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected = rows
    if paper_ids is not None:
        normalized = {normalize_paper_id(value) for value in paper_ids}
        selected = [row for row in rows if str(row.get("paper_id", "")) in normalized]
    if sample is None or sample >= len(selected):
        return list(selected)
    if sample <= 0:
        raise ValueError("sample must be positive")
    if sample_strategy == "prefix":
        return list(selected[:sample])
    if sample_strategy == "stride":
        if sample == 1:
            return [selected[0]]
        last = len(selected) - 1
        indexes = [round(index * last / (sample - 1)) for index in range(sample)]
        return [selected[index] for index in indexes]
    raise ValueError(f"unsupported sample strategy: {sample_strategy}")


def select_paper_ids_for_wiki(
    wiki_dir: Path,
    *,
    sample: int | None = None,
    sample_strategy: str = "prefix",
    paper_ids: set[str] | None = None,
) -> set[str] | None:
    if paper_ids is not None:
        return {normalize_paper_id(value) for value in paper_ids}
    if sample is None or sample_strategy == "prefix":
        return None
    rows = [{"paper_id": note.node_id} for note in paper_notes(wiki_dir)]
    return {str(row["paper_id"]) for row in select_manifest_rows(rows, sample=sample, sample_strategy=sample_strategy)}


def _status_row(row: dict[str, Any], status: str, error: str = "") -> QualityRow:
    return QualityRow(
        paper_id=str(row.get("paper_id", "")),
        provider=str(row.get("provider") or "manifest"),
        status=status,
        markdown_path=str(row.get("markdown_path", "")),
        note_path=str(row.get("note_path", "")),
        error=error or str(row.get("error", "")),
        warnings=[status.replace("_", " ")],
    )


def evaluate_manifest_row(row: dict[str, Any]) -> QualityRow:
    if not row.get("has_pdf_path", False):
        return _status_row(row, "no_pdf_path")
    if not row.get("pdf_exists", False):
        return _status_row(row, "missing_pdf")
    if not row.get("cached", False):
        return _status_row(row, "provider_error", str(row.get("error", "")))

    markdown_path = Path(str(row.get("markdown_path", "")))
    if not markdown_path.exists():
        return _status_row(row, "missing_markdown", f"missing markdown: {markdown_path}")
    markdown = markdown_path.read_text(encoding="utf-8")
    if not markdown.strip():
        return _status_row(row, "empty_markdown")

    metrics = score_markdown(markdown)
    return QualityRow(
        paper_id=str(row.get("paper_id", "")),
        provider=str(row.get("provider") or "manifest"),
        status="scored",
        markdown_path=str(markdown_path),
        note_path=str(row.get("note_path", "")),
        **metrics,
    )


def build_summary(rows: list[QualityRow], *, low_score: int) -> dict[str, Any]:
    scored_values = [row.score for row in rows if row.status == "scored" and row.score is not None]
    warning_counts = Counter(warning for row in rows for warning in row.warnings)
    return {
        "rows": len(rows),
        "scored": len(scored_values),
        "low_score_threshold": low_score,
        "low_score": sum(1 for value in scored_values if value < low_score),
        "average_score": round(statistics.mean(scored_values), 2) if scored_values else None,
        "median_score": round(statistics.median(scored_values), 2) if scored_values else None,
        "status_counts": dict(sorted(Counter(row.status for row in rows).items())),
        "malformed_manifest_rows": sum(1 for row in rows if row.status == "malformed_manifest"),
        "top_warnings": dict(warning_counts.most_common(10)),
    }


def evaluate_manifest_rows(
    manifest_path: Path,
    *,
    low_score: int,
    sample: int | None = None,
    sample_strategy: str = "prefix",
    paper_ids: set[str] | None = None,
) -> tuple[list[QualityRow], dict[str, Any]]:
    manifest_rows, malformed_rows = read_manifest_rows(manifest_path)
    selected_rows = select_manifest_rows(
        manifest_rows, sample=sample, sample_strategy=sample_strategy, paper_ids=paper_ids
    )
    rows = [evaluate_manifest_row(row) for row in selected_rows] + malformed_rows
    return rows, build_summary(rows, low_score=low_score)


def write_quality_outputs(
    *,
    output_jsonl: Path,
    output_report: Path,
    rows: list[QualityRow],
    summary: dict[str, Any],
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (item.provider, item.paper_id)):
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")

    output_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PDF Markdown Quality Report",
        "",
        "## Summary",
        "",
        f"- rows: {summary['rows']}",
        f"- scored: {summary['scored']}",
        f"- average score: {summary['average_score']}",
        f"- median score: {summary['median_score']}",
        f"- low score (<{summary['low_score_threshold']}): {summary['low_score']}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in summary["status_counts"].items():
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Lowest Scores And Errors", ""])
    ranked = sorted(
        rows,
        key=lambda item: (
            item.score if item.score is not None else -1,
            item.provider,
            item.paper_id,
        ),
    )
    for row in ranked[:25]:
        score = "" if row.score is None else str(row.score)
        detail = row.error or ", ".join(row.warnings)
        lines.append(f"- {row.paper_id} | {row.provider} | {row.status} | {score} | {detail}")
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_manifest(
    manifest_path: Path,
    *,
    output_jsonl: Path,
    output_report: Path,
    low_score: int = 55,
    sample: int | None = None,
    sample_strategy: str = "prefix",
    paper_ids: set[str] | None = None,
) -> dict[str, Any]:
    if low_score < 0 or low_score > 100:
        raise ValueError("low-score must be between 0 and 100")
    rows, summary = evaluate_manifest_rows(
        manifest_path,
        low_score=low_score,
        sample=sample,
        sample_strategy=sample_strategy,
        paper_ids=paper_ids,
    )
    write_quality_outputs(output_jsonl=output_jsonl, output_report=output_report, rows=rows, summary=summary)
    return {"summary": summary, "rows": [asdict(row) for row in rows]}


def evaluate_providers(
    wiki_dir: Path,
    *,
    providers: list[str],
    bakeoff_dir: Path,
    output_jsonl: Path,
    output_report: Path,
    low_score: int = 55,
    sample: int | None = None,
    sample_strategy: str = "prefix",
    paper_ids: set[str] | None = None,
    build_cache: Any = build_pdf_markdown_cache,
) -> dict[str, Any]:
    if low_score < 0 or low_score > 100:
        raise ValueError("low-score must be between 0 and 100")
    normalized_ids = select_paper_ids_for_wiki(
        wiki_dir, sample=sample, sample_strategy=sample_strategy, paper_ids=paper_ids
    )
    provider_rows: list[QualityRow] = []
    provider_summaries: dict[str, dict[str, Any]] = {}
    for provider in providers:
        parse_providers(provider)
        provider_output_dir = bakeoff_dir / provider
        provider_manifest = bakeoff_dir / f"{provider}_manifest.jsonl"
        build_cache(
            wiki_dir=wiki_dir,
            output_dir=provider_output_dir,
            manifest_path=provider_manifest,
            provider=provider,
            sample=sample if normalized_ids is None else None,
            paper_ids=normalized_ids,
        )
        rows, summary = evaluate_manifest_rows(
            provider_manifest,
            low_score=low_score,
            sample=sample if normalized_ids is None else None,
            sample_strategy=sample_strategy,
            paper_ids=normalized_ids,
        )
        provider_rows.extend(rows)
        provider_summaries[provider] = summary

    summary = build_summary(provider_rows, low_score=low_score)
    summary["providers"] = providers
    summary["provider_summaries"] = provider_summaries
    write_quality_outputs(
        output_jsonl=output_jsonl,
        output_report=output_report,
        rows=provider_rows,
        summary=summary,
    )
    return {"summary": summary, "rows": [asdict(row) for row in provider_rows]}


def read_paper_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        normalize_paper_id(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"rows: {summary['rows']}")
    print(f"scored: {summary['scored']}")
    print(f"average_score: {summary['average_score']}")
    print(f"median_score: {summary['median_score']}")
    print(f"low_score: {summary['low_score']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score generated PDF Markdown quality.")
    parser.add_argument("--manifest", default=".graph_index/pdf_markdown_manifest.jsonl")
    parser.add_argument("--output", default=".graph_index/pdf_markdown_quality.jsonl")
    parser.add_argument("--report", default=".graph_index/pdf_markdown_quality_report.md")
    parser.add_argument("--wiki-dir", default="wiki")
    parser.add_argument("--bakeoff-dir", default=".graph_index/pdf_markdown_bakeoff")
    parser.add_argument("--providers", help="Comma-separated provider list for extraction bakeoff")
    parser.add_argument("--low-score", type=int, default=55)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--sample-strategy", choices=["prefix", "stride"], default="prefix")
    parser.add_argument("--paper-ids", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.sample is not None and args.sample <= 0:
        print("sample must be positive")
        raise SystemExit(2)
    if args.low_score < 0 or args.low_score > 100:
        print("low-score must be between 0 and 100")
        raise SystemExit(2)

    try:
        paper_ids = read_paper_ids(args.paper_ids)
        if args.providers:
            report = evaluate_providers(
                Path(args.wiki_dir),
                providers=parse_providers(args.providers),
                bakeoff_dir=Path(args.bakeoff_dir),
                output_jsonl=Path(args.output),
                output_report=Path(args.report),
                low_score=args.low_score,
                sample=args.sample,
                sample_strategy=args.sample_strategy,
                paper_ids=paper_ids,
            )
        else:
            report = evaluate_manifest(
                Path(args.manifest),
                output_jsonl=Path(args.output),
                output_report=Path(args.report),
                low_score=args.low_score,
                sample=args.sample,
                sample_strategy=args.sample_strategy,
                paper_ids=paper_ids,
            )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print_report(report)


if __name__ == "__main__":
    main()
