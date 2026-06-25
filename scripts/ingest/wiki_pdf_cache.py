import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from scripts.ingest.wiki_pdf_audit import (
        TextExtractor,
        default_extract_text,
        detect_section_headings,
        paper_notes,
    )
except ModuleNotFoundError:
    from wiki_pdf_audit import (  # type: ignore[no-redef]
        TextExtractor,
        default_extract_text,
        detect_section_headings,
        paper_notes,
    )


def cache_filename(paper_id: str) -> str:
    return paper_id.removeprefix("paper:") + ".txt"


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomic write: stream to <path>.tmp then os.replace.
    Prevents SIGKILL or OOM mid-write from leaving the manifest truncated,
    which would cause fact extraction to silently skip the missing tail.
    """
    import os

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


def read_existing_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if isinstance(row, dict) and "paper_id" in row:
                result[str(row["paper_id"])] = row
    return result


def pdf_text_cache_meta_path(manifest_path: Path) -> Path:
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


def read_pdf_text_cache_signature(path: Path) -> list[list[str | int]] | None:
    meta_path = pdf_text_cache_meta_path(path)
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    signature = payload.get("vault_signature")
    return signature if isinstance(signature, list) else None


def write_pdf_text_cache_meta(path: Path, vault_signature: list[list[str | int]] | None) -> None:
    if vault_signature is None:
        return
    pdf_text_cache_meta_path(path).write_text(
        json.dumps({"vault_signature": vault_signature}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _fast_path_rows(prior: dict[str, dict]) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    for row in prior.values():
        if row.get("cached") is True:
            cache_path = Path(str(row.get("cache_path") or ""))
            pdf_path = Path(str(row.get("pdf_path") or "")).expanduser()
            if not cache_path.exists() or not pdf_path.exists():
                return None
            pdf_stat = pdf_path.stat()
            if cache_path.stat().st_mtime <= pdf_stat.st_mtime:
                return None
            rows.append(
                {
                    **row,
                    "pdf_exists": True,
                    "pdf_size": pdf_stat.st_size,
                    "pdf_mtime_ns": pdf_stat.st_mtime_ns,
                }
            )
            continue
        if row.get("has_pdf_path") and not row.get("pdf_exists"):
            pdf_path = Path(str(row.get("pdf_path") or "")).expanduser()
            if pdf_path.exists():
                return None
        rows.append(row)
    return rows


def build_pdf_text_cache(
    wiki_dir: Path,
    *,
    output_dir: Path,
    manifest_path: Path,
    extract_text: TextExtractor | None = None,
    sample: int | None = None,
    incremental: bool = False,
    linkgraph_meta_path: Path | None = None,
) -> dict[str, Any]:
    extractor = extract_text or default_extract_text
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped_incremental = 0

    prior = read_existing_manifest(manifest_path) if incremental else {}
    vault_signature = read_linkgraph_vault_signature(linkgraph_meta_path)
    if (
        incremental
        and sample is None
        and prior
        and vault_signature is not None
        and read_pdf_text_cache_signature(manifest_path) == vault_signature
    ):
        fast_rows = _fast_path_rows(prior)
        if fast_rows is not None:
            sorted_rows = sorted(fast_rows, key=lambda item: item["paper_id"])
            summary = _summary(sorted_rows, skipped_incremental=sum(1 for row in sorted_rows if row["cached"]))
            summary["fast_path"] = True
            return {"summary": summary, "rows": sorted_rows}

    for note in paper_notes(wiki_dir):
        if sample is not None and len(rows) >= sample:
            break
        pdf_path = Path(note.pdf_path).expanduser() if note.pdf_path else None
        cache_path = output_dir / cache_filename(note.node_id)

        if incremental:
            prior_row = prior.get(note.node_id)
            if (
                prior_row is not None
                and prior_row.get("cached") is True
                and prior_row.get("pdf_path") == note.pdf_path
                and cache_path.exists()
                and pdf_path is not None
                and pdf_path.exists()
                and cache_path.stat().st_mtime > pdf_path.stat().st_mtime
            ):
                pdf_stat = pdf_path.stat()
                rows.append(
                    {
                        **prior_row,
                        "pdf_exists": True,
                        "pdf_size": pdf_stat.st_size,
                        "pdf_mtime_ns": pdf_stat.st_mtime_ns,
                    }
                )
                skipped_incremental += 1
                continue
        else:
            if cache_path.exists():
                cache_path.unlink()

        row = {
            "paper_id": note.node_id,
            "note_path": note.rel_path,
            "pdf_path": note.pdf_path,
            "cache_path": "",
            "has_pdf_path": bool(note.pdf_path),
            "pdf_exists": bool(pdf_path and pdf_path.exists()),
            "pdf_size": 0,
            "pdf_mtime_ns": 0,
            "cached": False,
            "text_chars": 0,
            "section_headings": [],
            "error": "",
        }
        if not pdf_path:
            rows.append(row)
            continue
        if not pdf_path.exists():
            row["error"] = "missing pdf"
            rows.append(row)
            continue
        pdf_stat = pdf_path.stat()
        row["pdf_size"] = pdf_stat.st_size
        row["pdf_mtime_ns"] = pdf_stat.st_mtime_ns
        try:
            text = extractor(pdf_path)
            text_to_write = text.strip()
            if text_to_write:
                cache_path.write_text(text_to_write + "\n", encoding="utf-8")
                row["cached"] = True
                row["cache_path"] = str(cache_path)
                row["text_chars"] = len(text_to_write)
                row["section_headings"] = detect_section_headings(text)
            else:
                row["error"] = "empty text"
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            row["error"] = str(exc)
        rows.append(row)

    write_manifest(manifest_path, rows)
    write_pdf_text_cache_meta(manifest_path, vault_signature)
    sorted_rows = sorted(rows, key=lambda item: item["paper_id"])
    summary = _summary(
        sorted_rows,
        skipped_incremental=skipped_incremental if incremental else None,
    )
    return {"summary": summary, "rows": sorted_rows}


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"papers: {summary['papers']}")
    print(f"cached: {summary['cached']}")
    print(f"skipped: {summary['skipped']}")
    print(f"missing_pdfs: {summary['missing_pdfs']}")
    print(f"extraction_errors: {summary['extraction_errors']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build generated PDF text cache.")
    parser.add_argument("--wiki-dir", default="wiki")
    parser.add_argument("--output-dir", default=".graph_index/pdf_text")
    parser.add_argument("--manifest", default=".graph_index/pdf_text_manifest.jsonl")
    parser.add_argument("--sample", type=int, help="Cache only the first N paper notes")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="With --json, omit per-paper rows from stdout.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Reuse existing entries in the manifest; only re-extract paper_ids that are new or whose pdf_path / pdf_size changed.",
    )
    parser.add_argument("--linkgraph-meta")
    args = parser.parse_args(argv)

    if args.sample is not None and args.sample <= 0:
        print("sample must be positive")
        raise SystemExit(2)

    try:
        report = build_pdf_text_cache(
            Path(args.wiki_dir),
            output_dir=Path(args.output_dir),
            manifest_path=Path(args.manifest),
            sample=args.sample,
            incremental=args.incremental,
            linkgraph_meta_path=Path(args.linkgraph_meta) if args.linkgraph_meta else None,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    if args.json:
        payload = {"summary": report["summary"]} if args.summary_only else report
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print_report(report)


if __name__ == "__main__":
    main()
