#!/usr/bin/env python3
"""Lean foundation build — the clean-path counterpart to the frozen
legacy/graphrag/refresh-graph.sh.

Chains the markdown-native foundation steps (concept wikilinks -> PDF text cache
-> PDF Markdown source cache -> PDF-derived facts -> fact embeddings -> related
notes and graph communities) without the frozen GraphRAG layer.

Run from the repo root:
    uv run python -m scripts.core.build            # run the pipeline
    uv run python -m scripts.core.build --dry-run  # print the steps only
    uv run --with pymupdf4llm python -m scripts.core.build --pdf-markdown-profile high-quality
    uv run --with pymupdf4llm python -m scripts.core.build --pdf-markdown-profile high-quality --pdf-incremental
    uv run --with pymupdf4llm python -m scripts.core.build --pdf-markdown-profile high-quality --pdf-markdown-incremental
    uv run --with pymupdf4llm python -m scripts.core.build --pdf-markdown-profile high-quality --pdf-incremental --json
"""

import argparse
import json
import subprocess
import sys
import time

# (label, module, args) — each step runs as `python -m <module> <args>`. The
# ingest/embedding steps route through the `wiki` CLI so they inherit its
# argument defaults. Concept mutation needs a seed linkgraph for paper tags and
# a final linkgraph rebuild so new/updated wikilinks are indexed before
# downstream graph artifacts.
STEPS: list[tuple[str, str, list[str]]] = [
    ("link-graph seed", "scripts.core.linkgraph", ["build"]),
    ("concept backfill", "scripts.ingest.concept_backfill", ["--apply"]),
    ("concept split", "scripts.ingest.concept_split", ["--apply"]),
    ("link-graph final", "scripts.core.linkgraph", ["build"]),
    ("pdf cache", "scripts.core.wiki", ["pdf", "cache"]),
    ("pdf markdown", "scripts.core.wiki", ["pdf", "markdown"]),
    (
        "pdf facts",
        "scripts.core.wiki",
        ["pdf", "facts", "--manifest", ".graph_index/pdf_markdown_manifest.jsonl"],
    ),
    ("fact embeddings", "scripts.core.wiki", ["facts", "embed"]),
    ("relatedness", "scripts.search.related_notes", ["build"]),
    ("graph communities", "scripts.search.graph_communities", ["build"]),
]

PDF_MARKDOWN_PROFILES: dict[str, list[str]] = {
    "default": [],
    "high-quality": ["--provider", "hybrid"],
}
JSON_REPORT_STEP_LABELS = {"pdf cache", "pdf markdown", "pdf facts", "fact embeddings"}
SUMMARY_ONLY_STEP_LABELS = {"pdf cache", "pdf markdown"}


def build_steps(
    pdf_markdown_profile: str,
    *,
    pdf_incremental: bool = False,
    pdf_markdown_incremental: bool = False,
) -> list[tuple[str, str, list[str]]]:
    provider_args = PDF_MARKDOWN_PROFILES[pdf_markdown_profile]
    steps: list[tuple[str, str, list[str]]] = []
    for label, module, sub_args in STEPS:
        args = [*sub_args]
        if label == "pdf cache" and pdf_incremental:
            args.append("--incremental")
            args.extend(["--linkgraph-meta", ".graph_index/linkgraph.sqlite.meta.json"])
        if label == "pdf markdown":
            args.extend(provider_args)
            if pdf_incremental or pdf_markdown_incremental:
                args.append("--incremental")
            if pdf_incremental:
                args.extend(["--pdf-cache-manifest", ".graph_index/pdf_text_manifest.jsonl"])
                args.extend(["--linkgraph-meta", ".graph_index/linkgraph.sqlite.meta.json"])
        if label == "pdf facts" and pdf_incremental:
            args.append("--incremental")
        steps.append((label, module, args))
    return steps


def step_command(module: str, sub_args: list[str]) -> list[str]:
    return [sys.executable, "-m", module, *sub_args]


def step_plan(
    pdf_markdown_profile: str,
    *,
    pdf_incremental: bool = False,
    pdf_markdown_incremental: bool = False,
    json_report: bool = False,
) -> list[dict[str, object]]:
    steps = []
    for label, module, sub_args in build_steps(
        pdf_markdown_profile,
        pdf_incremental=pdf_incremental,
        pdf_markdown_incremental=pdf_markdown_incremental,
    ):
        args = [*sub_args]
        if json_report and label in JSON_REPORT_STEP_LABELS:
            args.append("--json")
            if label in SUMMARY_ONLY_STEP_LABELS:
                args.append("--summary-only")
        steps.append({"label": label, "cmd": step_command(module, args)})
    return steps


def emit_json_report(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_child_report(stdout: str) -> object | None:
    stripped = stdout.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the step sequence without executing it."
    )
    parser.add_argument(
        "--pdf-markdown-profile",
        choices=PDF_MARKDOWN_PROFILES,
        default="default",
        help=(
            "PDF Markdown extraction profile. Use 'high-quality' with "
            "`uv run --with pymupdf4llm` to route through the hybrid provider."
        ),
    )
    parser.add_argument(
        "--pdf-markdown-incremental",
        action="store_true",
        help="Reuse cached PDF Markdown rows when the PDF path, hash, and provider are unchanged.",
    )
    parser.add_argument(
        "--pdf-incremental",
        action="store_true",
        help="Reuse cached PDF text, PDF Markdown, and PDF facts where supported.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable build report with per-step commands and elapsed seconds.",
    )
    args = parser.parse_args(argv)

    planned_steps = step_plan(
        args.pdf_markdown_profile,
        pdf_incremental=args.pdf_incremental,
        pdf_markdown_incremental=args.pdf_markdown_incremental,
        json_report=args.json,
    )
    if args.dry_run and args.json:
        emit_json_report({"dry_run": True, "steps": planned_steps})
        return 0

    total_start = time.perf_counter()
    step_reports: list[dict[str, object]] = []
    for step in planned_steps:
        label = str(step["label"])
        cmd = list(step["cmd"])
        if args.dry_run:
            print(f"[build] {label}: {' '.join(cmd)}", flush=True)
            continue
        if not args.json:
            print(f"[build] {label} ...", flush=True)
        step_start = time.perf_counter()
        if args.json:
            result = subprocess.run(cmd, text=True, capture_output=True)
        else:
            result = subprocess.run(cmd)
        elapsed_s = round(time.perf_counter() - step_start, 3)
        if args.json:
            step_report = {
                "label": label,
                "cmd": cmd,
                "returncode": result.returncode,
                "elapsed_s": elapsed_s,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            if label in JSON_REPORT_STEP_LABELS and (report := parse_child_report(result.stdout)):
                step_report["report"] = report
            step_reports.append(step_report)
        if result.returncode != 0:
            if args.json:
                emit_json_report(
                    {
                        "dry_run": False,
                        "ok": False,
                        "elapsed_s": round(time.perf_counter() - total_start, 3),
                        "steps": step_reports,
                    }
                )
                return result.returncode
            print(
                f"[build] step '{label}' failed (rc={result.returncode})",
                file=sys.stderr,
                flush=True,
            )
            return result.returncode
    if args.json:
        emit_json_report(
            {
                "dry_run": False,
                "ok": True,
                "elapsed_s": round(time.perf_counter() - total_start, 3),
                "steps": step_reports,
            }
        )
        return 0
    if not args.dry_run:
        print("[build] foundation build complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
