import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from scripts.ingest.arxiv_ingest import safe_arxiv_id
except ModuleNotFoundError:
    from arxiv_ingest import safe_arxiv_id  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent


def _paper_id_from_arxiv_id(raw_id: str) -> str:
    safe_id, _ = safe_arxiv_id(raw_id)
    return f"paper:{safe_id}"


def _run_stage1(
    arxiv_ids: list[str],
    vault: Path,
    cache_dir: Path,
    merge: bool,
    dry_run: bool,
    no_download: bool,
) -> tuple[int, list[dict]]:
    """Run arxiv_ingest as subprocess; return (returncode, parsed_results).

    A non-zero returncode does NOT necessarily mean all IDs failed (arxiv_ingest
    exits 1 only when ALL IDs failed or --strict was set). The caller must consult
    both signals: returncode AND the per-ID action field in parsed results.
    """
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "scripts.ingest.arxiv_ingest",
        "--ids",
        ",".join(arxiv_ids),
        "--vault",
        str(vault),
        "--cache-dir",
        str(cache_dir),
        "--json",
    ]
    if merge:
        cmd.append("--merge")
    if dry_run:
        cmd.append("--dry-run")
    if no_download:
        cmd.append("--no-download")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)

    if result.stderr:
        print(f"[stage1 stderr] {result.stderr}", file=sys.stderr)

    results: list[dict] = []
    json_errors = 0
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # arxiv_ingest emits one JSON line per ID plus a final human summary line.
        # Skip lines that are clearly not JSON; surface the count of unexpected
        # parse failures so silent regressions in the child's stdout format show up.
        if not stripped.startswith("{"):
            continue
        try:
            results.append(json.loads(stripped))
        except json.JSONDecodeError:
            json_errors += 1
    if json_errors:
        print(
            f"[stage1 warning] {json_errors} unparseable JSON line(s) from arxiv_ingest",
            file=sys.stderr,
        )

    return result.returncode, results


def _run_stage2(
    vault: Path,
    graph_index: Path,
) -> int:
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "scripts.ingest.wiki_pdf_cache",
        "--wiki-dir",
        str(vault / "wiki"),
        "--output-dir",
        str(graph_index / "pdf_text"),
        "--manifest",
        str(graph_index / "pdf_text_manifest.jsonl"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.stderr:
        print(f"[stage2 stderr] {result.stderr}", file=sys.stderr)
    return result.returncode


def _run_stage2_markdown(
    vault: Path,
    graph_index: Path,
) -> int:
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "scripts.ingest.wiki_pdf_markdown",
        "--wiki-dir",
        str(vault / "wiki"),
        "--output-dir",
        str(graph_index / "pdf_markdown"),
        "--manifest",
        str(graph_index / "pdf_markdown_manifest.jsonl"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.stderr:
        print(f"[stage2 markdown stderr] {result.stderr}", file=sys.stderr)
    return result.returncode


def _run_stage3(
    vault: Path,
    graph_index: Path,
    limit_per_paper: int,
) -> int:
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "scripts.ingest.wiki_pdf_fact_extractor",
        "--manifest",
        str(graph_index / "pdf_markdown_manifest.jsonl"),
        "--output",
        str(graph_index / "pdf_facts.jsonl"),
        "--wiki-dir",
        str(vault / "wiki"),
        "--limit-per-paper",
        str(limit_per_paper),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.stderr:
        print(f"[stage3 stderr] {result.stderr}", file=sys.stderr)
    return result.returncode


def _read_manifest_cached_count(manifest_path: Path, expected_paper_ids: set[str]) -> int:
    if not manifest_path.exists():
        return 0
    count = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if row.get("paper_id") in expected_paper_ids and row.get("cached") is True:
                count += 1
    return count


def _count_facts_for_paper_ids(facts_path: Path, paper_ids: set[str]) -> int:
    if not facts_path.exists():
        return 0
    count = 0
    with facts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if obj.get("paper_id") in paper_ids:
                count += 1
    return count


def _append_log(
    log_path: Path,
    ingested: int,
    cached: int,
    facts: int,
    paper_ids: list[str],
) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    ids_str = ", ".join(paper_ids)
    entry = (
        f"\n## [{now_str}] auto_ingest | "
        f"arxiv {ingested} ingested, {cached} cached, {facts} facts | {ids_str}\n"
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate arXiv ingest: arxiv_ingest -> wiki_pdf_cache -> wiki_pdf_fact_extractor."
    )
    parser.add_argument("--arxiv-ids", required=True, help="Comma-separated arXiv IDs")
    parser.add_argument("--vault", type=Path, default=Path("."), help="Vault root directory")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "research-wiki" / "arxiv",
        help="Directory to store downloaded PDFs",
    )
    parser.add_argument(
        "--graph-index",
        type=Path,
        default=Path(".graph_index"),
        help="Graph index directory",
    )
    parser.add_argument("--merge", action="store_true", help="Merge into existing notes")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    parser.add_argument(
        "--limit-per-paper", type=int, default=25, help="Max facts per paper in Stage 3"
    )
    parser.add_argument("--skip-facts", action="store_true", help="Skip Stage 3 (fact extraction)")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip PDF download in Stage 1 (frontmatter still set; stages 2-3 will skip empty pdf_paths)",
    )
    parser.add_argument("--json", action="store_true", dest="emit_json", help="Emit JSON output")

    args = parser.parse_args(argv)

    arxiv_ids = [raw_id.strip() for raw_id in args.arxiv_ids.split(",") if raw_id.strip()]
    if not arxiv_ids:
        print("Error: no arXiv IDs provided", file=sys.stderr)
        return 2

    vault = args.vault.resolve()
    cache_dir = args.cache_dir
    graph_index = args.graph_index if args.graph_index.is_absolute() else vault / args.graph_index
    cache_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()

    # Stage 1 — ingest
    stage1_rc, stage1_results = _run_stage1(
        arxiv_ids=arxiv_ids,
        vault=vault,
        cache_dir=cache_dir,
        merge=args.merge,
        dry_run=args.dry_run,
        no_download=args.no_download,
    )

    if args.emit_json:
        for res in stage1_results:
            print(json.dumps({"stage": "ingest", **res}))

    # If the subprocess crashed before emitting JSON for every requested ID, we
    # have less information than we asked for — treat that as Stage 1 failure
    # so partial-output is never silently treated as authoritative.
    if stage1_rc != 0 and len(stage1_results) < len(arxiv_ids):
        if args.emit_json:
            print(
                json.dumps(
                    {
                        "stage": "summary",
                        "status": "stage1_crashed",
                        "rc": stage1_rc,
                        "results_received": len(stage1_results),
                        "ids_requested": len(arxiv_ids),
                    }
                )
            )
        print(
            f"Error: arxiv_ingest exited {stage1_rc} with "
            f"{len(stage1_results)}/{len(arxiv_ids)} results; aborting chain",
            file=sys.stderr,
        )
        return 2

    successful_ids = [
        res["arxiv_id"] for res in stage1_results if res.get("action") in ("created", "merged")
    ]
    ingested_count = len(successful_ids)

    if ingested_count == 0 and len(arxiv_ids) > 0:
        # All IDs failed
        if args.emit_json:
            print(json.dumps({"stage": "summary", "status": "all_failed"}))
        print("Error: all arXiv IDs failed to ingest", file=sys.stderr)
        return 2

    if args.dry_run:
        elapsed = time.monotonic() - t_start
        if args.emit_json:
            print(
                json.dumps(
                    {
                        "stage": "summary",
                        "ingested": ingested_count,
                        "pdf_cached": 0,
                        "facts_extracted": 0,
                        "elapsed": round(elapsed, 1),
                        "dry_run": True,
                    }
                )
            )
        else:
            print(f"ingested: {ingested_count}")
            print("pdf_cached: 0")
            print("facts_extracted: 0")
            print(f"elapsed: {elapsed:.1f}s")
        return 0

    expected_paper_ids = {_paper_id_from_arxiv_id(aid) for aid in successful_ids}

    # Stage 2 — pdf cache
    rc2 = _run_stage2(vault=vault, graph_index=graph_index)
    if rc2 != 0:
        return 3

    rc2_markdown = _run_stage2_markdown(vault=vault, graph_index=graph_index)
    if rc2_markdown != 0:
        return 3

    cached_count = _read_manifest_cached_count(
        graph_index / "pdf_markdown_manifest.jsonl", expected_paper_ids
    )

    facts_count = 0
    if not args.skip_facts:
        # Stage 3 — fact extraction
        rc3 = _run_stage3(
            vault=vault,
            graph_index=graph_index,
            limit_per_paper=args.limit_per_paper,
        )
        if rc3 != 0:
            return 4

        facts_count = _count_facts_for_paper_ids(
            graph_index / "pdf_facts.jsonl", expected_paper_ids
        )

    # Stage 4 — log
    log_path = vault / "wiki" / "log.md"
    if log_path.exists():
        _append_log(
            log_path,
            ingested=ingested_count,
            cached=cached_count,
            facts=facts_count,
            paper_ids=sorted(expected_paper_ids),
        )

    elapsed = time.monotonic() - t_start

    # Stage 5 — output
    if args.emit_json:
        print(
            json.dumps(
                {
                    "stage": "summary",
                    "ingested": ingested_count,
                    "pdf_cached": cached_count,
                    "facts_extracted": facts_count,
                    "elapsed": round(elapsed, 1),
                }
            )
        )
    else:
        print(f"ingested: {ingested_count}")
        print(f"pdf_cached: {cached_count}")
        print(f"facts_extracted: {facts_count}")
        print(f"elapsed: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
