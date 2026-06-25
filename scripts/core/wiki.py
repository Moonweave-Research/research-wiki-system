#!/usr/bin/env python3
import argparse
import sys
import subprocess
from pathlib import Path

try:
    from scripts.search.wiki_fact_embeddings import (
        DEFAULT_FASTEMBED_MODEL as _DEFAULT_FASTEMBED_MODEL,
    )
except ModuleNotFoundError:
    import importlib as _importlib

    _DEFAULT_FASTEMBED_MODEL = _importlib.import_module(
        "wiki_fact_embeddings"
    ).DEFAULT_FASTEMBED_MODEL  # type: ignore[assignment]

DEFAULT_FASTEMBED_MODEL = _DEFAULT_FASTEMBED_MODEL
DEFAULT_CANDIDATE_LIMIT = 80
DEFAULT_SCORE_MARGIN = 0.02
DEFAULT_KEYWORD_WEIGHT = 0.3
PDF_MARKDOWN_PROVIDER_CHOICES = [
    "plain",
    "pymupdf4llm",
    "hybrid",
    "docling",
    "markitdown",
    "unstructured",
]
GRAPH_REFRESH_OPTION_FLAGS = {
    "--wiki-dir",
    "--output-dir",
    "--pdf-text-dir",
    "--pdf-manifest",
    "--pdf-markdown-dir",
    "--pdf-markdown-manifest",
    "--pdf-markdown-provider",
    "--pdf-facts",
    "--limit-per-paper",
    "--sample",
    "--provider",
    "--model",
}


def graph_refresh_has_custom_args(argv: list[str]) -> bool:
    if "refresh" not in argv:
        return False
    return any(
        arg in GRAPH_REFRESH_OPTION_FLAGS
        or any(arg.startswith(f"{flag}=") for flag in GRAPH_REFRESH_OPTION_FLAGS)
        for arg in argv
    )


def run_command(cmd_args):
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(subprocess.os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{subprocess.os.pathsep}{existing}" if existing else str(repo_root)
    )
    try:
        subprocess.run(cmd_args, check=True, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Wiki CLI - Integrated Knowledge Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  answer    Create a local evidence answer from semantic facts
  ingest    Inject external knowledge (e.g., Zotero JSON) into the wiki
  lint      Check markdown files for frontmatter and linking errors
  query     Search the knowledge base
  graph     Build or audit the generated graph index
  facts     Extract or evaluate semantic facts
  pdf       Audit PDF readiness, cache extracted text, or extract PDF facts
  summarize Generate an AI summary for a specific paper
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- ANSWER ---
    parser_answer = subparsers.add_parser(
        "answer",
        help="Create a local evidence answer from semantic facts",
    )
    parser_answer.add_argument("query")
    parser_answer.add_argument("--graph-index", default=".graph_index")
    parser_answer.add_argument("--paper")
    parser_answer.add_argument("--target")
    parser_answer.add_argument("--fact-type")
    parser_answer.add_argument("--limit", type=int, default=5)
    parser_answer.add_argument(
        "--match-mode",
        choices=["auto", "keyword", "semantic", "embedding", "hybrid"],
        default="auto",
    )
    parser_answer.add_argument("--provider", choices=["fastembed"], default="fastembed")
    parser_answer.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    parser_answer.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser_answer.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)
    parser_answer.add_argument("--score-margin", type=float, default=DEFAULT_SCORE_MARGIN)
    parser_answer.add_argument("--json", action="store_true")

    # --- INGEST ---
    parser_ingest = subparsers.add_parser(
        "ingest", help="Inject external knowledge (Zotero) into the wiki"
    )
    parser_ingest.add_argument(
        "input_file", type=Path, help="Path to the Zotero JSON or BibTeX export file"
    )
    parser_ingest.add_argument(
        "--domain", required=True, choices=["research", "general"], help="Target domain"
    )
    parser_ingest.add_argument("--category", required=True, help="Target category (e.g., papers)")
    parser_ingest.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    # --- LINT ---
    subparsers.add_parser("lint", help="Check markdown files for formatting and linking errors")

    # --- QUERY ---
    parser_query = subparsers.add_parser("query", help="Search the knowledge base")
    parser_query.add_argument("term", nargs="?", help="Legacy positional search term")
    parser_query.add_argument("--domain", help="Filter by domain (research|general)")
    parser_query.add_argument("--tag", help="Filter by tag (case-insensitive match)")
    parser_query.add_argument(
        "--text",
        help="Full-text search in title + abstract (AND logic for multiple words)",
    )
    parser_query.add_argument(
        "--concept",
        help="Filter by concept note name (e.g. Dielectric_Elastomer_Actuator)",
    )
    parser_query.add_argument("--near", help="Show graph neighbors for a node id")
    parser_query.add_argument("--facts", action="store_true", help="Query semantic facts")
    parser_query.add_argument("--paper", help="Filter semantic facts by paper node id")
    parser_query.add_argument("--target", help="Filter semantic facts by target node id")
    parser_query.add_argument("--fact-type", help="Filter semantic facts by fact type")
    parser_query.add_argument("--graph-index", default=".graph_index", help="Graph index directory")
    parser_query.add_argument("--limit", type=int, default=25, help="Maximum result count")
    parser_query.add_argument(
        "--match-mode",
        choices=["auto", "keyword", "semantic", "embedding", "hybrid"],
        default="auto",
    )
    parser_query.add_argument("--provider", choices=["fastembed"], default="fastembed")
    parser_query.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    parser_query.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser_query.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)
    parser_query.add_argument("--score-margin", type=float, default=DEFAULT_SCORE_MARGIN)
    parser_query.add_argument(
        "--context",
        action="store_true",
        help="Print a compact GraphRAG context bundle for fact results",
    )
    parser_query.add_argument("--json", action="store_true")

    # --- GRAPH ---
    parser_graph = subparsers.add_parser(
        "graph", help="Build, audit, or refresh the generated graph index"
    )
    parser_graph.add_argument(
        "graph_command",
        choices=[
            "audit",
            "build",
            "refresh",
            "duplicates",
            "duplicate-triage",
            "duplicate-report",
        ],
    )
    parser_graph.add_argument("--wiki-dir", default="wiki")
    parser_graph.add_argument("--output-dir", default=".graph_index")
    parser_graph.add_argument("--facts-file")
    parser_graph.add_argument("--pdf-text-dir", default=".graph_index/pdf_text")
    parser_graph.add_argument("--pdf-manifest", default=".graph_index/pdf_text_manifest.jsonl")
    parser_graph.add_argument("--pdf-markdown-dir", default=".graph_index/pdf_markdown")
    parser_graph.add_argument(
        "--pdf-markdown-manifest", default=".graph_index/pdf_markdown_manifest.jsonl"
    )
    parser_graph.add_argument(
        "--pdf-markdown-provider", choices=PDF_MARKDOWN_PROVIDER_CHOICES, default="plain"
    )
    parser_graph.add_argument("--pdf-facts", default=".graph_index/pdf_facts.jsonl")
    parser_graph.add_argument("--limit-per-paper", type=int, default=20)
    parser_graph.add_argument("--sample", type=int)
    parser_graph.add_argument("--provider", choices=["fastembed"], default="fastembed")
    parser_graph.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    parser_graph.add_argument("--output")
    parser_graph.add_argument("--limit", type=int, default=None)
    parser_graph.add_argument("--json", action="store_true")

    # --- FACTS ---
    parser_facts = subparsers.add_parser(
        "facts", help="Extract or evaluate generated semantic facts"
    )
    parser_facts.add_argument("facts_command", choices=["extract", "eval", "embed"])
    parser_facts.add_argument("--wiki-dir", default="wiki")
    parser_facts.add_argument("--output", default=".graph_index/facts.jsonl")
    parser_facts.add_argument("--paper", action="append", default=[])
    parser_facts.add_argument("--limit-per-paper", type=int, default=6)
    parser_facts.add_argument("--graph-index", default=".graph_index")
    parser_facts.add_argument("--benchmark")
    parser_facts.add_argument("--limit", type=int, default=5)
    parser_facts.add_argument(
        "--match-mode", choices=["keyword", "semantic", "embedding", "hybrid"], default="keyword"
    )
    parser_facts.add_argument("--provider", choices=["fastembed"], default="fastembed")
    parser_facts.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    parser_facts.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser_facts.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)
    parser_facts.add_argument("--score-margin", type=float, default=DEFAULT_SCORE_MARGIN)
    parser_facts.add_argument("--min-recall", type=float)
    parser_facts.add_argument("--min-precision", type=float)
    parser_facts.add_argument("--max-unexpected", type=int)
    parser_facts.add_argument("--max-forbidden", type=int)
    parser_facts.add_argument("--json", action="store_true")

    # --- PDF ---
    parser_pdf = subparsers.add_parser("pdf", help="Audit, cache, or extract facts from PDF text")
    pdf_subparsers = parser_pdf.add_subparsers(
        dest="pdf_command", required=True, help="PDF commands"
    )

    parser_pdf_audit = pdf_subparsers.add_parser("audit", help="Audit PDF readiness")
    parser_pdf_audit.add_argument("--wiki-dir", default="wiki")
    parser_pdf_audit.add_argument("--sample", type=int)
    parser_pdf_audit.add_argument("--json", action="store_true")

    parser_pdf_cache = pdf_subparsers.add_parser("cache", help="Cache extracted PDF text")
    parser_pdf_cache.add_argument("--wiki-dir", default="wiki")
    parser_pdf_cache.add_argument("--output-dir", default=".graph_index/pdf_text")
    parser_pdf_cache.add_argument("--manifest", default=".graph_index/pdf_text_manifest.jsonl")
    parser_pdf_cache.add_argument("--sample", type=int)
    parser_pdf_cache.add_argument("--incremental", action="store_true")
    parser_pdf_cache.add_argument("--linkgraph-meta")
    parser_pdf_cache.add_argument("--json", action="store_true")
    parser_pdf_cache.add_argument("--summary-only", action="store_true")

    parser_pdf_markdown = pdf_subparsers.add_parser("markdown", help="Cache generated PDF Markdown")
    parser_pdf_markdown.add_argument("--wiki-dir", default="wiki")
    parser_pdf_markdown.add_argument("--output-dir", default=".graph_index/pdf_markdown")
    parser_pdf_markdown.add_argument(
        "--manifest", default=".graph_index/pdf_markdown_manifest.jsonl"
    )
    parser_pdf_markdown.add_argument(
        "--provider",
        choices=PDF_MARKDOWN_PROVIDER_CHOICES,
        default="plain",
    )
    parser_pdf_markdown.add_argument("--sample", type=int)
    parser_pdf_markdown.add_argument("--incremental", action="store_true")
    parser_pdf_markdown.add_argument("--pdf-cache-manifest")
    parser_pdf_markdown.add_argument("--linkgraph-meta")
    parser_pdf_markdown.add_argument("--json", action="store_true")
    parser_pdf_markdown.add_argument("--summary-only", action="store_true")

    parser_pdf_facts = pdf_subparsers.add_parser("facts", help="Extract facts from cached PDF text")
    parser_pdf_facts.add_argument("--manifest", default=".graph_index/pdf_text_manifest.jsonl")
    parser_pdf_facts.add_argument("--output", default=".graph_index/pdf_facts.jsonl")
    parser_pdf_facts.add_argument("--limit-per-paper", type=int, default=20)
    parser_pdf_facts.add_argument("--sample", type=int)
    parser_pdf_facts.add_argument("--incremental", action="store_true")
    parser_pdf_facts.add_argument("--json", action="store_true")

    parser_pdf_quality = pdf_subparsers.add_parser(
        "quality", help="Score generated PDF Markdown quality"
    )
    parser_pdf_quality.add_argument("--wiki-dir", default="wiki")
    parser_pdf_quality.add_argument(
        "--manifest", default=".graph_index/pdf_markdown_manifest.jsonl"
    )
    parser_pdf_quality.add_argument("--output", default=".graph_index/pdf_markdown_quality.jsonl")
    parser_pdf_quality.add_argument(
        "--report", default=".graph_index/pdf_markdown_quality_report.md"
    )
    parser_pdf_quality.add_argument("--bakeoff-dir", default=".graph_index/pdf_markdown_bakeoff")
    parser_pdf_quality.add_argument("--providers")
    parser_pdf_quality.add_argument("--sample", type=int)
    parser_pdf_quality.add_argument(
        "--sample-strategy", choices=["prefix", "stride"], default="prefix"
    )
    parser_pdf_quality.add_argument("--paper-ids")
    parser_pdf_quality.add_argument("--low-score", type=int, default=55)
    parser_pdf_quality.add_argument("--json", action="store_true")

    # --- SUMMARIZE ---
    parser_summarize = subparsers.add_parser("summarize", help="Generate an AI summary for a paper")
    parser_summarize.add_argument("citekey", help="The citation key or filename of the paper")

    # --- CONCEPT ---
    subparsers.add_parser("concept", help="Inspect concept notes and their linked paper counts")

    # --- LINK-BACK ---
    parser_linkback = subparsers.add_parser(
        "link-back",
        help="Add concept back-links to literature notes (builds bidirectional graph)",
    )
    parser_linkback.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing files",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    if args.command == "answer":
        cmd = [
            sys.executable,
            "-m",
            "legacy.graphrag.wiki_answer",
            args.query,
            "--graph-index",
            args.graph_index,
            "--limit",
            str(args.limit),
            "--match-mode",
            args.match_mode,
        ]
        if args.match_mode in {"auto", "embedding", "hybrid"}:
            cmd.extend(
                [
                    "--provider",
                    args.provider,
                    "--model",
                    args.model,
                    "--candidate-limit",
                    str(args.candidate_limit),
                    "--keyword-weight",
                    str(args.keyword_weight),
                    "--score-margin",
                    str(args.score_margin),
                ]
            )
        if args.paper:
            cmd.extend(["--paper", args.paper])
        if args.target:
            cmd.extend(["--target", args.target])
        if args.fact_type:
            cmd.extend(["--fact-type", args.fact_type])
        if args.json:
            cmd.append("--json")
        run_command(cmd)

    elif args.command == "ingest":
        import re
        import yaml

        wiki_root = repo_root
        research_dir = wiki_root / "wiki" / "research"
        concepts_dir = wiki_root / "wiki" / "concepts"

        # Snapshot existing files before ingest
        before = {f.stem for f in research_dir.glob("*.md")}

        cmd = [
            sys.executable,
            "-m",
            "scripts.ingest.zotero_sync",
            str(args.input_file),
            "--domain",
            args.domain,
            "--category",
            args.category,
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"Running Ingest: {' '.join(cmd)}")
        run_command(cmd)

        # Detect newly created files
        after = {f.stem for f in research_dir.glob("*.md")}
        new_stems = sorted(after - before)

        if not new_stems:
            print("\nNo new papers added.")
            return

        print(f"\n{len(new_stems)} new paper(s) added.")

        # Build concept → distinctive keyword set (exclude words shared by >2 concepts)
        def _tokenize(s: str) -> set[str]:
            return {w.lower() for w in re.split(r"[\s\-_/]+", str(s)) if len(w) > 3}

        raw_concept_kws: dict[str, set[str]] = {}
        for cf in concepts_dir.glob("*.md"):
            raw = cf.read_text(encoding="utf-8")
            fm_match = re.match(r"^---\r?\n(.*?)\r?\n---", raw, re.DOTALL)
            fm: dict = {}
            if fm_match:
                try:
                    fm = yaml.safe_load(fm_match.group(1)) or {}
                except yaml.YAMLError:
                    pass
            kws: set[str] = set()
            for t in fm.get("tags") or []:
                kws |= _tokenize(t)
            for a in fm.get("aliases") or []:
                kws |= _tokenize(a)
            raw_concept_kws[cf.stem] = kws

        # Count how many concepts each keyword appears in
        from collections import Counter

        kw_freq: Counter = Counter(kw for kws in raw_concept_kws.values() for kw in kws)
        # Keep only keywords that appear in ≤2 concept notes (discriminating)
        concept_keywords = {
            concept: {kw for kw in kws if kw_freq[kw] <= 2}
            for concept, kws in raw_concept_kws.items()
        }

        # For each new paper, find matching concepts
        suggestions: dict[str, list[str]] = {}
        for stem in new_stems:
            paper_path = research_dir / f"{stem}.md"
            raw = paper_path.read_text(encoding="utf-8")
            fm_match = re.match(r"^---\r?\n(.*?)\r?\n---", raw, re.DOTALL)
            paper_tags: set[str] = set()
            if fm_match:
                try:
                    fm = yaml.safe_load(fm_match.group(1)) or {}
                except yaml.YAMLError:
                    fm = {}
                for t in fm.get("tags") or []:
                    paper_tags |= _tokenize(t)

            matched = [concept for concept, kws in concept_keywords.items() if kws & paper_tags]
            if matched:
                suggestions[stem] = sorted(matched)

        if suggestions:
            print("\nConcept notes to consider updating:")
            print("-" * 60)
            for stem, concepts in suggestions.items():
                print(f"  {stem}")
                for c in concepts:
                    print(f"    → [[{c}]]")
        else:
            print("\nNo concept matches found — may be a new topic.")

        # Auto link-back for new papers
        print("\nRunning link-back for new papers…")
        paper_to_concepts: dict[str, set[str]] = {}
        for cf in sorted(concepts_dir.glob("*.md")):
            content = cf.read_text(encoding="utf-8")
            links = re.findall(r"\[\[([^\]|#]+?)(?:\|[^\]]*)?\]\]", content)
            for link in links:
                if link in after:
                    paper_to_concepts.setdefault(link, set()).add(cf.stem)

        updated = 0
        for stem in new_stems:
            if stem not in paper_to_concepts:
                continue
            paper_path = research_dir / f"{stem}.md"
            content = paper_path.read_text(encoding="utf-8")
            new_line = "**Concepts:** " + " ".join(
                f"[[{c}]]" for c in sorted(paper_to_concepts[stem])
            )
            existing = re.search(r"\*\*Concepts:\*\*.*", content)
            if existing and existing.group() == new_line:
                continue
            if existing:
                content = re.sub(r"\*\*Concepts:\*\*.*", new_line, content)
            else:
                content = content.rstrip("\n") + f"\n\n{new_line}\n"
            paper_path.write_text(content, encoding="utf-8")
            updated += 1
        print(f"Back-linked {updated} of {len(new_stems)} new papers.")

    elif args.command == "lint":
        cmd = [sys.executable, "-m", "scripts.core.wiki_lint"]
        print(f"🔍 Running Lint: {' '.join(cmd)}")
        # Need to ensure PYTHONPATH is set to project root
        env = dict(subprocess.os.environ)
        env["PYTHONPATH"] = str(repo_root)
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)

    elif args.command == "query":
        if args.near or args.facts:
            cmd = [
                sys.executable,
                "-m",
                "legacy.graphrag.wiki_query",
                "--graph-index",
                args.graph_index,
                "--limit",
                str(args.limit),
            ]
            if args.near:
                cmd.extend(["--near", args.near])
                if args.json:
                    cmd.append("--json")
            if args.facts:
                unsupported_filters = []
                if args.domain:
                    unsupported_filters.append("--domain")
                if args.tag:
                    unsupported_filters.append("--tag")
                if args.concept:
                    unsupported_filters.append("--concept")
                if unsupported_filters:
                    print(
                        "Semantic fact queries do not support filters: "
                        + ", ".join(unsupported_filters)
                    )
                    sys.exit(2)
                cmd.append("--facts")
                cmd.extend(["--match-mode", args.match_mode])
                if args.match_mode in {"auto", "embedding", "hybrid"}:
                    cmd.extend(
                        [
                            "--provider",
                            args.provider,
                            "--model",
                            args.model,
                            "--candidate-limit",
                            str(args.candidate_limit),
                            "--keyword-weight",
                            str(args.keyword_weight),
                            "--score-margin",
                            str(args.score_margin),
                        ]
                    )
                if args.paper:
                    cmd.extend(["--paper", args.paper])
                if args.target:
                    cmd.extend(["--target", args.target])
                if args.fact_type:
                    cmd.extend(["--fact-type", args.fact_type])
                if args.text:
                    cmd.extend(["--text", args.text])
                elif args.term:
                    cmd.append(args.term)
                if args.context:
                    cmd.append("--context")
                if args.json:
                    cmd.append("--json")
            run_command(cmd)
            return

        from legacy.graphrag.wiki_query import query_wiki

        text_query = args.text or args.term
        results = query_wiki(
            domain=args.domain,
            tag=args.tag,
            text=text_query,
            concept=args.concept,
        )

        if not results:
            print("No results found.")
            return

        print(f"{'Year':<6} {'Title':<60} {'Concepts'}")
        print("-" * 100)
        for r in results:
            concepts_str = ", ".join(r["concepts"]) if r["concepts"] else "-"
            title = r["title"][:58] + ".." if len(r["title"]) > 60 else r["title"]
            print(f"{r['year']:<6} {title:<60} {concepts_str}")
            if r["excerpt"]:
                print(f"       {r['excerpt']}")
        print(f"\n{len(results)} result(s)")

    elif args.command == "graph":
        if args.graph_command == "refresh":
            if not graph_refresh_has_custom_args(sys.argv):
                run_command(["bash", str(repo_root / "legacy" / "graphrag" / "refresh-graph.sh")])
                return
            cache_cmd = [
                sys.executable,
                "-m",
                "scripts.ingest.wiki_pdf_cache",
                "--wiki-dir",
                args.wiki_dir,
                "--output-dir",
                args.pdf_text_dir,
                "--manifest",
                args.pdf_manifest,
            ]
            facts_cmd = [
                sys.executable,
                "-m",
                "scripts.ingest.wiki_pdf_fact_extractor",
                "--manifest",
                args.pdf_markdown_manifest,
                "--output",
                args.pdf_facts,
                "--limit-per-paper",
                str(args.limit_per_paper),
            ]
            markdown_cmd = [
                sys.executable,
                "-m",
                "scripts.ingest.wiki_pdf_markdown",
                "--wiki-dir",
                args.wiki_dir,
                "--output-dir",
                args.pdf_markdown_dir,
                "--manifest",
                args.pdf_markdown_manifest,
                "--provider",
                args.pdf_markdown_provider,
            ]
            if args.sample is not None:
                cache_cmd.extend(["--sample", str(args.sample)])
                markdown_cmd.extend(["--sample", str(args.sample)])
                facts_cmd.extend(["--sample", str(args.sample)])
            run_command(cache_cmd)
            run_command(markdown_cmd)
            run_command(facts_cmd)
            run_command(
                [
                    sys.executable,
                    "-m",
                    "legacy.graphrag.wiki_graph",
                    "build",
                    "--wiki-dir",
                    args.wiki_dir,
                    "--output-dir",
                    args.output_dir,
                    "--facts-file",
                    args.pdf_facts,
                ]
            )
            run_command(
                [
                    sys.executable,
                    "-m",
                    "scripts.search.wiki_fact_embeddings",
                    "build",
                    "--graph-index",
                    args.output_dir,
                    "--provider",
                    args.provider,
                    "--model",
                    args.model,
                ]
            )
            run_command(
                [
                    sys.executable,
                    "-m",
                    "legacy.graphrag.wiki_communities",
                    "--graph-index",
                    args.output_dir,
                    "--seed",
                    "42",
                ]
            )
            return
        cmd = [
            sys.executable,
            "-m",
            "legacy.graphrag.wiki_graph",
            args.graph_command,
            "--wiki-dir",
            args.wiki_dir,
            "--output-dir",
            args.output_dir,
        ]
        if args.facts_file:
            cmd.extend(["--facts-file", args.facts_file])
        if args.output:
            cmd.extend(["--output", args.output])
        if args.graph_command == "duplicate-report" and args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.json:
            cmd.append("--json")
        run_command(cmd)

    elif args.command == "facts":
        if args.facts_command == "extract":
            cmd = [
                sys.executable,
                "-m",
                "scripts.ingest.wiki_fact_extractor",
                "--wiki-dir",
                args.wiki_dir,
                "--output",
                args.output,
                "--limit-per-paper",
                str(args.limit_per_paper),
            ]
            for paper in args.paper:
                cmd.extend(["--paper", paper])
        elif args.facts_command == "eval":
            if not args.benchmark:
                print("facts eval requires --benchmark")
                sys.exit(2)
            cmd = [
                sys.executable,
                "-m",
                "eval.wiki_fact_eval",
                "--graph-index",
                args.graph_index,
                "--benchmark",
                args.benchmark,
                "--limit",
                str(args.limit),
                "--match-mode",
                args.match_mode,
            ]
            if args.match_mode in {"embedding", "hybrid"}:
                cmd.extend(
                    [
                        "--provider",
                        args.provider,
                        "--model",
                        args.model,
                        "--candidate-limit",
                        str(args.candidate_limit),
                        "--keyword-weight",
                        str(args.keyword_weight),
                        "--score-margin",
                        str(args.score_margin),
                    ]
                )
            for flag_name, value in (
                ("--min-recall", args.min_recall),
                ("--min-precision", args.min_precision),
                ("--max-unexpected", args.max_unexpected),
                ("--max-forbidden", args.max_forbidden),
            ):
                if value is not None:
                    cmd.extend([flag_name, str(value)])
            if args.json:
                cmd.append("--json")
        else:
            cmd = [
                sys.executable,
                "-m",
                "scripts.search.wiki_fact_embeddings",
                "build",
                "--graph-index",
                args.graph_index,
                "--provider",
                args.provider,
                "--model",
                args.model,
            ]
            if args.json:
                cmd.append("--json")
        run_command(cmd)

    elif args.command == "pdf":
        pdf_module = {
            "audit": "scripts.ingest.wiki_pdf_audit",
            "cache": "scripts.ingest.wiki_pdf_cache",
            "markdown": "scripts.ingest.wiki_pdf_markdown",
            "facts": "scripts.ingest.wiki_pdf_fact_extractor",
            "quality": "scripts.ingest.wiki_pdf_markdown_quality",
        }[args.pdf_command]
        cmd = [sys.executable, "-m", pdf_module]
        if args.pdf_command in {"audit", "cache", "markdown", "quality"}:
            cmd.extend(["--wiki-dir", args.wiki_dir])
        if args.pdf_command in {"cache", "markdown"}:
            cmd.extend(["--output-dir", args.output_dir, "--manifest", args.manifest])
        if args.pdf_command == "cache" and args.linkgraph_meta:
            cmd.extend(["--linkgraph-meta", args.linkgraph_meta])
        if args.pdf_command == "markdown":
            cmd.extend(["--provider", args.provider])
            if args.pdf_cache_manifest:
                cmd.extend(["--pdf-cache-manifest", args.pdf_cache_manifest])
            if args.linkgraph_meta:
                cmd.extend(["--linkgraph-meta", args.linkgraph_meta])
        if args.pdf_command == "quality":
            cmd.extend(
                [
                    "--manifest",
                    args.manifest,
                    "--output",
                    args.output,
                    "--report",
                    args.report,
                    "--bakeoff-dir",
                    args.bakeoff_dir,
                ]
            )
            if args.providers:
                cmd.extend(["--providers", args.providers])
            cmd.extend(
                ["--sample-strategy", args.sample_strategy, "--low-score", str(args.low_score)]
            )
            if args.paper_ids:
                cmd.extend(["--paper-ids", args.paper_ids])
        if args.pdf_command == "facts":
            cmd.extend(
                [
                    "--manifest",
                    args.manifest,
                    "--output",
                    args.output,
                    "--limit-per-paper",
                    str(args.limit_per_paper),
                ]
            )
        if args.sample is not None:
            cmd.extend(["--sample", str(args.sample)])
        if args.pdf_command in {"cache", "markdown", "facts"} and args.incremental:
            cmd.append("--incremental")
        if args.json:
            cmd.append("--json")
            if args.pdf_command in {"cache", "markdown"} and args.summary_only:
                cmd.append("--summary-only")
        run_command(cmd)

    elif args.command == "summarize":
        # Find the file and print its path for the Gemini CLI agent to use
        citekey = args.citekey
        if citekey.endswith(".md"):
            citekey = citekey[:-3]

        # Try exact match first
        found = list(Path("wiki").glob(f"**/{citekey}.md"))
        if not found:
            print(f"Error: Paper '{citekey}' not found.")
            sys.exit(1)

        print(f"FOUND_FILE: {found[0]}")

    elif args.command == "concept":
        import re

        wiki_root = repo_root
        concepts_dir = wiki_root / "wiki" / "concepts"
        research_stems = {f.stem for f in (wiki_root / "wiki" / "research").glob("*.md")}

        concept_files = sorted(concepts_dir.glob("*.md"))
        if not concept_files:
            print("No concept notes found in wiki/concepts/")
            sys.exit(0)

        print(f"{'Concept':<42} {'Papers':>7}  {'All Links':>10}")
        print("-" * 63)
        total_papers = 0
        for cf in concept_files:
            content = cf.read_text(encoding="utf-8")
            all_links = re.findall(r"\[\[([^\]|#]+?)(?:\|[^\]]*)?\]\]", content)
            paper_links = [link for link in all_links if link in research_stems]
            total_papers += len(paper_links)
            print(f"{cf.stem:<42} {len(paper_links):>7}  {len(all_links):>10}")
        print("-" * 63)
        print(f"{'TOTAL':<42} {total_papers:>7}")

    elif args.command == "link-back":
        import re

        wiki_root = repo_root
        concepts_dir = wiki_root / "wiki" / "concepts"
        research_dir = wiki_root / "wiki" / "research"
        research_stems = {f.stem for f in research_dir.glob("*.md")}

        # Build paper → {concept, ...} mapping from all concept notes (set deduplicates)
        paper_to_concepts: dict[str, set[str]] = {}
        for cf in sorted(concepts_dir.glob("*.md")):
            content = cf.read_text(encoding="utf-8")
            links = re.findall(r"\[\[([^\]|#]+?)(?:\|[^\]]*)?\]\]", content)
            for link in links:
                if link in research_stems:
                    paper_to_concepts.setdefault(link, set()).add(cf.stem)

        updated = skipped = 0
        for paper_stem, concepts in sorted(paper_to_concepts.items()):
            paper_path = research_dir / f"{paper_stem}.md"
            content = paper_path.read_text(encoding="utf-8")
            new_line = "**Concepts:** " + " ".join(f"[[{c}]]" for c in sorted(concepts))

            existing = re.search(r"\*\*Concepts:\*\*.*", content)
            if existing and existing.group() == new_line:
                skipped += 1
                continue

            if args.dry_run:
                action = "UPDATE" if existing else "ADD"
                print(f"  {action}: {paper_stem}")
                print(f"         → {new_line}")
                updated += 1
                continue

            if existing:
                content = re.sub(r"\*\*Concepts:\*\*.*", new_line, content)
            else:
                content = content.rstrip("\n") + f"\n\n{new_line}\n"

            paper_path.write_text(content, encoding="utf-8")
            updated += 1

        mode = "Would update" if args.dry_run else "Updated"
        print(f"{mode}: {updated} papers | Already current: {skipped}")


if __name__ == "__main__":
    main()
