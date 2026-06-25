import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from scripts.ingest.wiki_pdf_audit import (
        default_extract_text,
        detect_section_headings,
        paper_notes,
    )
except ModuleNotFoundError:
    from wiki_pdf_audit import (  # type: ignore[no-redef]
        default_extract_text,
        detect_section_headings,
        paper_notes,
    )


@dataclass(frozen=True)
class MarkdownExtraction:
    markdown: str
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    extractor: str = "plain"


MarkdownExtractor = Callable[[Path], MarkdownExtraction]

SUPPORTED_PROVIDERS = ("plain", "pymupdf4llm", "hybrid", "docling", "markitdown", "unstructured")


PDF_TEXT_TRANSLATION = str.maketrans(
    {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
    }
)


def markdown_filename(paper_id: str) -> str:
    return paper_id.removeprefix("paper:") + ".md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda item: item["paper_id"]):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def read_existing_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "paper_id" in row:
                result[str(row["paper_id"])] = row
    return result


def pdf_markdown_cache_meta_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(manifest_path.suffix + ".meta.json")


def read_linkgraph_vault_signature(path: Path | None) -> list[list[str | int]] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    signature = payload.get("signature")
    return signature if isinstance(signature, list) else None


def read_pdf_markdown_cache_meta(manifest_path: Path) -> dict[str, Any]:
    meta_path = pdf_markdown_cache_meta_path(manifest_path)
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_pdf_markdown_cache_meta(
    manifest_path: Path,
    *,
    vault_signature: list[list[str | int]] | None,
    pdf_cache_manifest_signature: str,
    provider: str,
) -> None:
    if vault_signature is None or not pdf_cache_manifest_signature:
        return
    pdf_markdown_cache_meta_path(manifest_path).write_text(
        json.dumps(
            {
                "pdf_cache_manifest_signature": pdf_cache_manifest_signature,
                "provider": provider,
                "vault_signature": vault_signature,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def text_to_markdown(text: str) -> str:
    normalized = text.translate(PDF_TEXT_TRANSLATION).replace("\f", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(" ".join(paragraph).strip())
            paragraph.clear()

    for line in lines:
        if not line:
            flush_paragraph()
            continue
        heading = _section_heading_label(line)
        if heading:
            flush_paragraph()
            blocks.append(f"## {heading}")
            continue
        paragraph.append(line)
    flush_paragraph()
    return "\n\n".join(blocks).strip()


def _section_heading_label(line: str) -> str | None:
    clean = re.sub(r"^#+\s*", "", line.strip())
    clean = re.sub(r"^\d+\.?\s*", "", clean).strip().lower()
    labels = {
        "abstract": "Abstract",
        "introduction": "Introduction",
        "method": "Methods",
        "methods": "Methods",
        "materials and methods": "Methods",
        "results": "Results",
        "results and discussion": "Results",
        "conclusion": "Conclusion",
        "conclusions": "Conclusion",
        "summary": "Conclusion",
    }
    return labels.get(clean)


def markdown_section_headings(markdown: str) -> list[str]:
    normalized_lines = []
    for line in markdown.splitlines():
        normalized_lines.append(re.sub(r"^\s*#+\s*", "", line))
    return detect_section_headings("\n".join(normalized_lines))


def generated_markdown(
    *,
    paper_id: str,
    title: str,
    pdf_path: str,
    extraction: MarkdownExtraction,
) -> str:
    body = extraction.markdown.strip()
    return (
        "<!-- generated by scripts.ingest.wiki_pdf_markdown; do not edit -->\n"
        f"<!-- paper_id: {paper_id}; extractor: {extraction.extractor}; pdf_path: {pdf_path} -->\n\n"
        f"# {title}\n\n"
        f"{body}\n"
    )


def extract_plain_markdown(pdf_path: Path) -> MarkdownExtraction:
    text = default_extract_text(pdf_path, max_pages=0)
    return MarkdownExtraction(markdown=text_to_markdown(text), extractor="plain")


def extract_pymupdf4llm_markdown(pdf_path: Path) -> MarkdownExtraction:
    try:
        module = importlib.import_module("pymupdf4llm")
    except ModuleNotFoundError as exc:
        raise RuntimeError("pymupdf4llm provider requested but pymupdf4llm is not installed") from exc
    markdown = module.to_markdown(str(pdf_path), use_ocr=False)
    return MarkdownExtraction(markdown=str(markdown), extractor="pymupdf4llm")


@dataclass(frozen=True)
class MarkdownShape:
    word_count: int
    heading_count: int
    has_abstract: bool
    duplicate_line_ratio: float
    replacement_char_count: int


def markdown_shape(markdown: str) -> MarkdownShape:
    text = markdown.strip()
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    duplicate_lines = sum(
        count - 1
        for count in {
            line: nonempty_lines.count(line)
            for line in set(nonempty_lines)
        }.values()
        if count > 1
    )
    lower = text.lower()
    return MarkdownShape(
        word_count=len(re.findall(r"\b[\w-]+\b", text, flags=re.UNICODE)),
        heading_count=sum(1 for line in nonempty_lines if re.match(r"^#{1,6}\s+\S", line)),
        has_abstract=bool(re.search(r"(^|\n)#{1,6}\s*abstract\b|\babstract\b", lower)),
        duplicate_line_ratio=duplicate_lines / len(nonempty_lines) if nonempty_lines else 0.0,
        replacement_char_count=text.count("\ufffd"),
    )


def hybrid_fallback_reason(candidate: str, baseline: str) -> str | None:
    candidate_shape = markdown_shape(candidate)
    baseline_shape = markdown_shape(baseline)
    if candidate_shape.word_count == 0:
        return "empty pymupdf4llm output"
    if baseline_shape.word_count >= 500 and candidate_shape.word_count < baseline_shape.word_count * 0.25:
        return (
            "pymupdf4llm under-extracted "
            f"({candidate_shape.word_count}/{baseline_shape.word_count} words)"
        )
    if candidate_shape.duplicate_line_ratio >= 0.55:
        return f"pymupdf4llm duplicate-line ratio {candidate_shape.duplicate_line_ratio:.2f}"
    if baseline_shape.replacement_char_count == 0 and candidate_shape.replacement_char_count >= 25:
        return f"pymupdf4llm introduced {candidate_shape.replacement_char_count} replacement characters"
    if (
        baseline_shape.replacement_char_count == 0
        and candidate_shape.replacement_char_count > 0
        and baseline_shape.word_count >= 500
        and candidate_shape.word_count < baseline_shape.word_count * 0.80
    ):
        return (
            "pymupdf4llm lost text with replacement characters "
            f"({candidate_shape.word_count}/{baseline_shape.word_count} words, "
            f"{candidate_shape.replacement_char_count} replacements)"
        )
    if (
        (baseline_shape.has_abstract or baseline_shape.heading_count > 0)
        and not candidate_shape.has_abstract
        and candidate_shape.heading_count == 0
    ):
        return "pymupdf4llm lost abstract and headings"
    return None


def extract_hybrid_markdown(pdf_path: Path) -> MarkdownExtraction:
    plain = extract_plain_markdown(pdf_path)
    try:
        pymupdf = extract_pymupdf4llm_markdown(pdf_path)
    except Exception as exc:
        return MarkdownExtraction(
            markdown=plain.markdown,
            page_count=plain.page_count,
            warnings=[*plain.warnings, f"hybrid fallback: pymupdf4llm failed: {exc}"],
            extractor="hybrid:plain",
        )
    reason = hybrid_fallback_reason(pymupdf.markdown, plain.markdown)
    if reason:
        return MarkdownExtraction(
            markdown=plain.markdown,
            page_count=plain.page_count,
            warnings=[*plain.warnings, f"hybrid fallback: {reason}"],
            extractor="hybrid:plain",
        )
    return MarkdownExtraction(
        markdown=pymupdf.markdown,
        page_count=pymupdf.page_count,
        warnings=[*pymupdf.warnings, "hybrid selected: pymupdf4llm"],
        extractor="hybrid:pymupdf4llm",
    )


def extract_docling_markdown(pdf_path: Path) -> MarkdownExtraction:
    try:
        module = importlib.import_module("docling.document_converter")
    except ModuleNotFoundError as exc:
        raise RuntimeError("docling provider requested but docling is not installed") from exc
    converter = module.DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    return MarkdownExtraction(markdown=str(markdown), extractor="docling")


def extract_markitdown_markdown(pdf_path: Path) -> MarkdownExtraction:
    try:
        module = importlib.import_module("markitdown")
    except ModuleNotFoundError as exc:
        raise RuntimeError("markitdown provider requested but markitdown is not installed") from exc
    result = module.MarkItDown().convert(str(pdf_path))
    return MarkdownExtraction(markdown=str(result.text_content), extractor="markitdown")


def _unstructured_element_text(element: object) -> str:
    text = str(getattr(element, "text", "") or "").strip()
    if not text:
        text = str(element).strip()
    if not text:
        return ""
    category = str(getattr(element, "category", "") or "")
    if category in {"Title", "Header"}:
        return f"## {text}"
    return text


def extract_unstructured_markdown(pdf_path: Path) -> MarkdownExtraction:
    try:
        module = importlib.import_module("unstructured.partition.pdf")
    except ModuleNotFoundError as exc:
        raise RuntimeError("unstructured provider requested but unstructured is not installed") from exc
    elements = module.partition_pdf(filename=str(pdf_path))
    markdown = "\n\n".join(
        text for element in elements if (text := _unstructured_element_text(element))
    )
    return MarkdownExtraction(markdown=markdown, extractor="unstructured")


def provider_extractor(provider: str) -> MarkdownExtractor:
    if provider == "plain":
        return extract_plain_markdown
    if provider == "pymupdf4llm":
        return extract_pymupdf4llm_markdown
    if provider == "hybrid":
        return extract_hybrid_markdown
    if provider == "docling":
        return extract_docling_markdown
    if provider == "markitdown":
        return extract_markitdown_markdown
    if provider == "unstructured":
        return extract_unstructured_markdown
    raise ValueError(f"unsupported provider: {provider}")


def _empty_row(note: Any, pdf_path: Path | None, markdown_path: Path) -> dict[str, Any]:
    pdf_exists = bool(pdf_path and pdf_path.exists())
    stat = pdf_path.stat() if pdf_exists and pdf_path is not None else None
    return {
        "paper_id": note.node_id,
        "note_path": note.rel_path,
        "pdf_path": note.pdf_path,
        "markdown_path": "",
        "has_pdf_path": bool(note.pdf_path),
        "pdf_exists": pdf_exists,
        "cached": False,
        "markdown_chars": 0,
        "page_count": None,
        "pdf_size": stat.st_size if stat is not None else 0,
        "pdf_mtime_ns": stat.st_mtime_ns if stat is not None else 0,
        "pdf_sha256": "",
        "extractor": "",
        "provider": "",
        "section_headings": [],
        "warnings": [],
        "error": "",
        "_markdown_path_obj": markdown_path,
    }


def read_pdf_cache_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("paper_id"), str):
                result[str(row["paper_id"])] = row
    return result


def _row_from_pdf_cache_metadata(
    note: Any,
    metadata: dict[str, Any] | None,
    markdown_path: Path,
) -> dict[str, Any] | None:
    if metadata is None or metadata.get("pdf_path") != note.pdf_path:
        return None
    return {
        "paper_id": note.node_id,
        "note_path": note.rel_path,
        "pdf_path": note.pdf_path,
        "markdown_path": "",
        "has_pdf_path": bool(note.pdf_path),
        "pdf_exists": bool(metadata.get("pdf_exists")),
        "cached": False,
        "markdown_chars": 0,
        "page_count": None,
        "pdf_size": int(metadata.get("pdf_size") or 0),
        "pdf_mtime_ns": int(metadata.get("pdf_mtime_ns") or 0),
        "pdf_sha256": "",
        "extractor": "",
        "provider": "",
        "section_headings": [],
        "warnings": [],
        "error": "",
        "_markdown_path_obj": markdown_path,
    }


def _strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    clean = dict(row)
    clean.pop("_markdown_path_obj", None)
    return clean


def _summary(rows: list[dict[str, Any]], *, skipped_incremental: int | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "papers": len(rows),
        "cached": sum(1 for row in rows if row["cached"]),
        "skipped": sum(1 for row in rows if not row["cached"]),
        "missing_pdfs": sum(1 for row in rows if row["has_pdf_path"] and not row["pdf_exists"]),
        "extraction_errors": sum(
            1 for row in rows if row["pdf_exists"] and not row["cached"] and bool(row["error"])
        ),
    }
    if skipped_incremental is not None:
        summary["skipped_incremental"] = skipped_incremental
    return summary


def _fast_path_rows(prior: dict[str, dict[str, Any]], provider: str) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    for row in prior.values():
        if row.get("cached") is True:
            markdown_path = Path(str(row.get("markdown_path") or ""))
            if row.get("provider", "plain") != provider or not markdown_path.exists():
                return None
        rows.append(row)
    return rows


def _prior_row_matches_pdf(row: dict[str, Any], prior_row: dict[str, Any], pdf_path: Path) -> bool:
    if prior_row.get("pdf_size") != row["pdf_size"]:
        return False
    if prior_row.get("pdf_mtime_ns") == row["pdf_mtime_ns"]:
        return True
    return prior_row.get("pdf_sha256") == sha256_file(pdf_path)


def build_pdf_markdown_cache(
    wiki_dir: Path,
    *,
    output_dir: Path,
    manifest_path: Path,
    provider: str = "plain",
    extract_markdown: MarkdownExtractor | None = None,
    sample: int | None = None,
    paper_ids: set[str] | None = None,
    incremental: bool = False,
    pdf_cache_manifest_path: Path | None = None,
    linkgraph_meta_path: Path | None = None,
) -> dict[str, Any]:
    extractor = extract_markdown or provider_extractor(provider)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = read_existing_manifest(manifest_path) if incremental else {}
    pdf_cache_metadata = read_pdf_cache_metadata(pdf_cache_manifest_path) if incremental else {}
    vault_signature = read_linkgraph_vault_signature(linkgraph_meta_path)
    pdf_cache_manifest_signature = (
        file_signature(pdf_cache_manifest_path) if pdf_cache_manifest_path is not None else ""
    )
    if (
        incremental
        and sample is None
        and paper_ids is None
        and prior
        and vault_signature is not None
        and pdf_cache_manifest_signature
    ):
        prior_meta = read_pdf_markdown_cache_meta(manifest_path)
        if (
            prior_meta.get("vault_signature") == vault_signature
            and prior_meta.get("pdf_cache_manifest_signature") == pdf_cache_manifest_signature
            and prior_meta.get("provider") == provider
        ):
            fast_rows = _fast_path_rows(prior, provider)
            if fast_rows is not None:
                sorted_rows = sorted(fast_rows, key=lambda item: item["paper_id"])
                summary = _summary(
                    sorted_rows,
                    skipped_incremental=sum(1 for row in sorted_rows if row["cached"]),
                )
                summary["fast_path"] = True
                return {"summary": summary, "rows": sorted_rows}
    rows: list[dict[str, Any]] = []
    skipped_incremental = 0

    for note in paper_notes(wiki_dir):
        if paper_ids is not None and note.node_id not in paper_ids:
            continue
        if sample is not None and len(rows) >= sample:
            break
        pdf_path = Path(note.pdf_path).expanduser() if note.pdf_path else None
        markdown_path = output_dir / markdown_filename(note.node_id)
        row = (
            _row_from_pdf_cache_metadata(
                note,
                pdf_cache_metadata.get(note.node_id),
                markdown_path,
            )
            or _empty_row(note, pdf_path, markdown_path)
        )

        if not incremental and markdown_path.exists():
            markdown_path.unlink()

        if not pdf_path:
            rows.append(_strip_internal(row))
            continue
        if not pdf_path.exists():
            row["error"] = "missing pdf"
            rows.append(_strip_internal(row))
            continue

        prior_row = prior.get(note.node_id)
        if (
            incremental
            and prior_row is not None
            and prior_row.get("cached") is True
            and prior_row.get("pdf_path") == note.pdf_path
            and _prior_row_matches_pdf(row, prior_row, pdf_path)
            and prior_row.get("provider", "plain") == provider
            and markdown_path.exists()
        ):
            rows.append(
                {
                    **prior_row,
                    "pdf_exists": row["pdf_exists"],
                    "pdf_size": row["pdf_size"],
                    "pdf_mtime_ns": row["pdf_mtime_ns"],
                }
            )
            skipped_incremental += 1
            continue

        try:
            row["pdf_sha256"] = sha256_file(pdf_path)
            extraction = extractor(pdf_path)
            markdown = generated_markdown(
                paper_id=note.node_id,
                title=str(getattr(note, "title", "") or Path(note.rel_path).stem),
                pdf_path=note.pdf_path,
                extraction=extraction,
            )
            if not extraction.markdown.strip():
                row["error"] = "empty markdown"
                if markdown_path.exists():
                    markdown_path.unlink()
                rows.append(_strip_internal(row))
                continue
            _atomic_write_text(markdown_path, markdown)
            row.update(
                {
                    "markdown_path": str(markdown_path),
                    "cached": True,
                    "markdown_chars": len(markdown.strip()),
                    "page_count": extraction.page_count,
                    "extractor": extraction.extractor,
                    "provider": provider,
                    "section_headings": markdown_section_headings(markdown),
                    "warnings": list(extraction.warnings),
                    "error": "",
                }
            )
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            row["error"] = str(exc)
            row["provider"] = provider
            if markdown_path.exists():
                markdown_path.unlink()
        rows.append(_strip_internal(row))

    write_manifest(manifest_path, rows)
    write_pdf_markdown_cache_meta(
        manifest_path,
        vault_signature=vault_signature,
        pdf_cache_manifest_signature=pdf_cache_manifest_signature,
        provider=provider,
    )
    summary = _summary(rows, skipped_incremental=skipped_incremental if incremental else None)
    return {"summary": summary, "rows": sorted(rows, key=lambda item: item["paper_id"])}


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"papers: {summary['papers']}")
    print(f"cached: {summary['cached']}")
    print(f"skipped: {summary['skipped']}")
    print(f"missing_pdfs: {summary['missing_pdfs']}")
    print(f"extraction_errors: {summary['extraction_errors']}")
    if "skipped_incremental" in summary:
        print(f"skipped_incremental: {summary['skipped_incremental']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build generated PDF Markdown cache.")
    parser.add_argument("--wiki-dir", default="wiki")
    parser.add_argument("--output-dir", default=".graph_index/pdf_markdown")
    parser.add_argument("--manifest", default=".graph_index/pdf_markdown_manifest.jsonl")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="plain")
    parser.add_argument("--sample", type=int, help="Cache only the first N paper notes")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument(
        "--pdf-cache-manifest",
        help="Reuse current PDF metadata from the text-cache manifest during incremental builds.",
    )
    parser.add_argument(
        "--linkgraph-meta",
        help="Reuse the linkgraph vault signature to skip unchanged incremental Markdown scans.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="With --json, omit per-paper rows from stdout.",
    )
    args = parser.parse_args(argv)

    if args.sample is not None and args.sample <= 0:
        print("sample must be positive")
        raise SystemExit(2)

    try:
        report = build_pdf_markdown_cache(
            Path(args.wiki_dir),
            output_dir=Path(args.output_dir),
            manifest_path=Path(args.manifest),
            provider=args.provider,
            sample=args.sample,
            incremental=args.incremental,
            pdf_cache_manifest_path=Path(args.pdf_cache_manifest)
            if args.pdf_cache_manifest
            else None,
            linkgraph_meta_path=Path(args.linkgraph_meta) if args.linkgraph_meta else None,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    if args.json:
        payload = {"summary": report["summary"]} if args.summary_only else report
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print_report(report)


if __name__ == "__main__":
    main()
