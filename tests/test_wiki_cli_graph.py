import sys
from pathlib import Path

import pytest

from scripts.core import wiki
from scripts.search.wiki_embedding_defaults import DEFAULT_FASTEMBED_MODEL


def test_wiki_cli_dispatches_query_near_to_graph_query(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "query",
            "--near",
            "paper:center",
            "--graph-index",
            ".graph_index",
            "--limit",
            "5",
            "--context",
            "--json",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:3] == [sys.executable, "-m", "legacy.graphrag.wiki_query"]
    assert cmd[3:] == [
        "--graph-index",
        ".graph_index",
        "--limit",
        "5",
        "--near",
        "paper:center",
        "--json",
    ]


def test_wiki_cli_dispatches_fact_query_to_graph_query(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "query",
            "voltage",
            "--facts",
            "--paper",
            "paper:center",
            "--target",
            "metric:voltage",
            "--fact-type",
            "Claim",
            "--graph-index",
            ".graph_index",
            "--limit",
            "5",
            "--context",
            "--json",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:3] == [sys.executable, "-m", "legacy.graphrag.wiki_query"]
    assert cmd[3:] == [
        "--graph-index",
        ".graph_index",
        "--limit",
        "5",
        "--facts",
        "--match-mode",
        "auto",
        "--provider",
        "fastembed",
        "--model",
        DEFAULT_FASTEMBED_MODEL,
        "--candidate-limit",
        "80",
        "--keyword-weight",
        "0.3",
        "--score-margin",
        "0.02",
        "--paper",
        "paper:center",
        "--target",
        "metric:voltage",
        "--fact-type",
        "Claim",
        "voltage",
        "--context",
        "--json",
    ]


def test_wiki_cli_dispatches_answer_command(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "answer",
            "low voltage actuator",
            "--target",
            "metric:voltage",
            "--graph-index",
            ".graph_index",
            "--limit",
            "3",
            "--match-mode",
            "keyword",
            "--json",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:3] == [sys.executable, "-m", "legacy.graphrag.wiki_answer"]
    assert cmd[3:] == [
        "low voltage actuator",
        "--graph-index",
        ".graph_index",
        "--limit",
        "3",
        "--match-mode",
        "keyword",
        "--target",
        "metric:voltage",
        "--json",
    ]


def test_wiki_cli_dispatches_graph_duplicates(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "graph",
            "duplicates",
            "--output-dir",
            ".graph_index",
            "--json",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:3] == [sys.executable, "-m", "legacy.graphrag.wiki_graph"]
    assert cmd[3:] == [
        "duplicates",
        "--wiki-dir",
        "wiki",
        "--output-dir",
        ".graph_index",
        "--json",
    ]


def test_wiki_cli_dispatches_graph_duplicate_triage(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "graph",
            "duplicate-triage",
            "--output-dir",
            ".graph_index",
            "--json",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:3] == [sys.executable, "-m", "legacy.graphrag.wiki_graph"]
    assert cmd[3:] == [
        "duplicate-triage",
        "--wiki-dir",
        "wiki",
        "--output-dir",
        ".graph_index",
        "--json",
    ]


def test_wiki_cli_dispatches_graph_duplicate_report(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "graph",
            "duplicate-report",
            "--output-dir",
            ".graph_index",
            "--output",
            "reports/duplicates.md",
            "--limit",
            "5",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:3] == [sys.executable, "-m", "legacy.graphrag.wiki_graph"]
    assert cmd[3:] == [
        "duplicate-report",
        "--wiki-dir",
        "wiki",
        "--output-dir",
        ".graph_index",
        "--output",
        "reports/duplicates.md",
        "--limit",
        "5",
    ]


def test_wiki_cli_dispatches_hybrid_fact_query_options(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "query",
            "heated droplet actuator",
            "--facts",
            "--match-mode",
            "hybrid",
            "--target",
            "mechanism:electrothermal_actuation",
            "--graph-index",
            ".graph_index",
            "--limit",
            "5",
            "--candidate-limit",
            "50",
            "--keyword-weight",
            "0.1",
            "--score-margin",
            "0.04",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:3] == [sys.executable, "-m", "legacy.graphrag.wiki_query"]
    assert cmd[3:] == [
        "--graph-index",
        ".graph_index",
        "--limit",
        "5",
        "--facts",
        "--match-mode",
        "hybrid",
        "--provider",
        "fastembed",
        "--model",
        DEFAULT_FASTEMBED_MODEL,
        "--candidate-limit",
        "50",
        "--keyword-weight",
        "0.1",
        "--score-margin",
        "0.04",
        "--target",
        "mechanism:electrothermal_actuation",
        "heated droplet actuator",
    ]


def test_wiki_cli_rejects_unsupported_fact_filters(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["wiki.py", "query", "--facts", "--domain", "research"],
    )

    with pytest.raises(SystemExit) as excinfo:
        wiki.main()

    assert excinfo.value.code == 2
    assert "Semantic fact queries do not support filters: --domain" in capsys.readouterr().out


def test_wiki_cli_dispatches_graph_build(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "graph",
            "build",
            "--wiki-dir",
            "wiki",
            "--output-dir",
            ".graph_index",
            "--facts-file",
            "facts.jsonl",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:4] == [
        sys.executable,
        "-m",
        "legacy.graphrag.wiki_graph",
        "build",
    ]
    assert cmd[4:] == [
        "--wiki-dir",
        "wiki",
        "--output-dir",
        ".graph_index",
        "--facts-file",
        "facts.jsonl",
    ]


def test_wiki_cli_dispatches_graph_refresh_pipeline(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "graph",
            "refresh",
        ],
    )

    wiki.main()

    refresh_script = (
        Path(wiki.__file__).resolve().parents[2] / "legacy" / "graphrag" / "refresh-graph.sh"
    )
    assert commands == [
        [
            "bash",
            str(refresh_script),
        ],
    ]


def test_wiki_cli_dispatches_custom_graph_refresh_pipeline_options(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "graph",
            "refresh",
            "--wiki-dir",
            "wiki",
            "--output-dir",
            ".graph_index",
            "--pdf-text-dir",
            ".graph_index/pdf_text",
            "--pdf-manifest",
            ".graph_index/pdf_text_manifest.jsonl",
            "--pdf-markdown-dir",
            ".graph_index/pdf_markdown",
            "--pdf-markdown-manifest",
            ".graph_index/pdf_markdown_manifest.jsonl",
            "--pdf-markdown-provider",
            "hybrid",
            "--pdf-facts",
            ".graph_index/pdf_facts.jsonl",
            "--limit-per-paper",
            "6",
            "--sample",
            "2",
            "--provider",
            "fastembed",
            "--model",
            DEFAULT_FASTEMBED_MODEL,
        ],
    )

    wiki.main()

    assert commands == [
        [
            sys.executable,
            "-m",
            "scripts.ingest.wiki_pdf_cache",
            "--wiki-dir",
            "wiki",
            "--output-dir",
            ".graph_index/pdf_text",
            "--manifest",
            ".graph_index/pdf_text_manifest.jsonl",
            "--sample",
            "2",
        ],
        [
            sys.executable,
            "-m",
            "scripts.ingest.wiki_pdf_markdown",
            "--wiki-dir",
            "wiki",
            "--output-dir",
            ".graph_index/pdf_markdown",
            "--manifest",
            ".graph_index/pdf_markdown_manifest.jsonl",
            "--provider",
            "hybrid",
            "--sample",
            "2",
        ],
        [
            sys.executable,
            "-m",
            "scripts.ingest.wiki_pdf_fact_extractor",
            "--manifest",
            ".graph_index/pdf_markdown_manifest.jsonl",
            "--output",
            ".graph_index/pdf_facts.jsonl",
            "--limit-per-paper",
            "6",
            "--sample",
            "2",
        ],
        [
            sys.executable,
            "-m",
            "legacy.graphrag.wiki_graph",
            "build",
            "--wiki-dir",
            "wiki",
            "--output-dir",
            ".graph_index",
            "--facts-file",
            ".graph_index/pdf_facts.jsonl",
        ],
        [
            sys.executable,
            "-m",
            "scripts.search.wiki_fact_embeddings",
            "build",
            "--graph-index",
            ".graph_index",
            "--provider",
            "fastembed",
            "--model",
            DEFAULT_FASTEMBED_MODEL,
        ],
        [
            sys.executable,
            "-m",
            "legacy.graphrag.wiki_communities",
            "--graph-index",
            ".graph_index",
            "--seed",
            "42",
        ],
    ]


def test_wiki_cli_detects_custom_graph_refresh_options_before_command(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "graph",
            "--wiki-dir",
            "alt-wiki",
            "refresh",
        ],
    )

    wiki.main()

    assert commands[0][:6] == [
        sys.executable,
        "-m",
        "scripts.ingest.wiki_pdf_cache",
        "--wiki-dir",
        "alt-wiki",
        "--output-dir",
    ]


def test_wiki_cli_dispatches_fact_extraction(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "facts",
            "extract",
            "--wiki-dir",
            "wiki",
            "--output",
            ".graph_index/facts.jsonl",
            "--paper",
            "paper:center",
            "--limit-per-paper",
            "4",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:4] == [
        sys.executable,
        "-m",
        "scripts.ingest.wiki_fact_extractor",
        "--wiki-dir",
    ]
    assert cmd[4:] == [
        "wiki",
        "--output",
        ".graph_index/facts.jsonl",
        "--limit-per-paper",
        "4",
        "--paper",
        "paper:center",
    ]


def test_wiki_cli_dispatches_fact_eval(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "facts",
            "eval",
            "--graph-index",
            ".graph_index",
            "--benchmark",
            "benchmarks/facts.json",
            "--limit",
            "10",
            "--match-mode",
            "semantic",
            "--min-recall",
            "1.0",
            "--min-precision",
            "0.9",
            "--max-unexpected",
            "0",
            "--max-forbidden",
            "0",
            "--json",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:4] == [
        sys.executable,
        "-m",
        "eval.wiki_fact_eval",
        "--graph-index",
    ]
    assert cmd[4:] == [
        ".graph_index",
        "--benchmark",
        "benchmarks/facts.json",
        "--limit",
        "10",
        "--match-mode",
        "semantic",
        "--min-recall",
        "1.0",
        "--min-precision",
        "0.9",
        "--max-unexpected",
        "0",
        "--max-forbidden",
        "0",
        "--json",
    ]


def test_wiki_cli_dispatches_fact_embedding_build(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "facts",
            "embed",
            "--graph-index",
            ".graph_index",
            "--provider",
            "fastembed",
            "--model",
            DEFAULT_FASTEMBED_MODEL,
            "--json",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:4] == [
        sys.executable,
        "-m",
        "scripts.search.wiki_fact_embeddings",
        "build",
    ]
    assert cmd[4:] == [
        "--graph-index",
        ".graph_index",
        "--provider",
        "fastembed",
        "--model",
        DEFAULT_FASTEMBED_MODEL,
        "--json",
    ]


def test_wiki_cli_dispatches_pdf_audit(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "pdf",
            "audit",
            "--wiki-dir",
            "wiki",
            "--sample",
            "3",
            "--json",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:4] == [
        sys.executable,
        "-m",
        "scripts.ingest.wiki_pdf_audit",
        "--wiki-dir",
    ]
    assert cmd[4:] == ["wiki", "--sample", "3", "--json"]


def test_wiki_cli_dispatches_pdf_cache(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "pdf",
            "cache",
            "--wiki-dir",
            "wiki",
            "--output-dir",
            ".graph_index/pdf_text",
            "--manifest",
            ".graph_index/pdf_text_manifest.jsonl",
            "--sample",
            "2",
            "--json",
            "--summary-only",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:4] == [
        sys.executable,
        "-m",
        "scripts.ingest.wiki_pdf_cache",
        "--wiki-dir",
    ]
    assert cmd[4:] == [
        "wiki",
        "--output-dir",
        ".graph_index/pdf_text",
        "--manifest",
        ".graph_index/pdf_text_manifest.jsonl",
        "--sample",
        "2",
        "--json",
        "--summary-only",
    ]


def test_wiki_cli_dispatches_pdf_markdown(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "pdf",
            "markdown",
            "--wiki-dir",
            "wiki",
            "--output-dir",
            ".graph_index/pdf_markdown",
            "--manifest",
            ".graph_index/pdf_markdown_manifest.jsonl",
            "--provider",
            "hybrid",
            "--sample",
            "2",
            "--incremental",
            "--json",
            "--summary-only",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:4] == [
        sys.executable,
        "-m",
        "scripts.ingest.wiki_pdf_markdown",
        "--wiki-dir",
    ]
    assert cmd[4:] == [
        "wiki",
        "--output-dir",
        ".graph_index/pdf_markdown",
        "--manifest",
        ".graph_index/pdf_markdown_manifest.jsonl",
        "--provider",
        "hybrid",
        "--sample",
        "2",
        "--incremental",
        "--json",
        "--summary-only",
    ]


def test_wiki_cli_dispatches_pdf_facts(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "pdf",
            "facts",
            "--manifest",
            ".graph_index/pdf_text_manifest.jsonl",
            "--output",
            ".graph_index/pdf_facts.jsonl",
            "--limit-per-paper",
            "3",
            "--sample",
            "2",
            "--json",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:4] == [
        sys.executable,
        "-m",
        "scripts.ingest.wiki_pdf_fact_extractor",
        "--manifest",
    ]
    assert cmd[4:] == [
        ".graph_index/pdf_text_manifest.jsonl",
        "--output",
        ".graph_index/pdf_facts.jsonl",
        "--limit-per-paper",
        "3",
        "--sample",
        "2",
        "--json",
    ]


def test_wiki_cli_dispatches_pdf_cache_incremental(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "pdf",
            "cache",
            "--wiki-dir",
            "wiki",
            "--output-dir",
            ".graph_index/pdf_text",
            "--manifest",
            ".graph_index/pdf_text_manifest.jsonl",
            "--incremental",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    assert commands[0][-1] == "--incremental"


def test_wiki_cli_dispatches_pdf_facts_incremental(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "pdf",
            "facts",
            "--manifest",
            ".graph_index/pdf_markdown_manifest.jsonl",
            "--output",
            ".graph_index/pdf_facts.jsonl",
            "--incremental",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    assert commands[0][-1] == "--incremental"


def test_wiki_cli_dispatches_pdf_quality(monkeypatch):
    commands = []
    monkeypatch.setattr(wiki, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki.py",
            "pdf",
            "quality",
            "--wiki-dir",
            "wiki",
            "--manifest",
            ".graph_index/pdf_markdown_manifest.jsonl",
            "--output",
            ".graph_index/pdf_markdown_quality.jsonl",
            "--report",
            ".graph_index/pdf_markdown_quality_report.md",
            "--bakeoff-dir",
            ".graph_index/pdf_markdown_bakeoff",
            "--providers",
            "plain,pymupdf4llm",
            "--sample",
            "3",
            "--sample-strategy",
            "stride",
            "--paper-ids",
            "sample.txt",
            "--low-score",
            "70",
            "--json",
        ],
    )

    wiki.main()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:4] == [
        sys.executable,
        "-m",
        "scripts.ingest.wiki_pdf_markdown_quality",
        "--wiki-dir",
    ]
    assert cmd[4:] == [
        "wiki",
        "--manifest",
        ".graph_index/pdf_markdown_manifest.jsonl",
        "--output",
        ".graph_index/pdf_markdown_quality.jsonl",
        "--report",
        ".graph_index/pdf_markdown_quality_report.md",
        "--bakeoff-dir",
        ".graph_index/pdf_markdown_bakeoff",
        "--providers",
        "plain,pymupdf4llm",
        "--sample-strategy",
        "stride",
        "--low-score",
        "70",
        "--paper-ids",
        "sample.txt",
        "--sample",
        "3",
        "--json",
    ]


def test_wiki_cli_pdf_subcommands_reject_unsupported_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["wiki.py", "pdf", "audit", "--output", ".graph_index/pdf_facts.jsonl"],
    )

    with pytest.raises(SystemExit) as excinfo:
        wiki.main()

    assert excinfo.value.code == 2
