import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legacy.graphrag.wiki_graph import (  # noqa: E402
    compute_wiki_signature,
    read_build_signature,
    write_build_signature,
)


def _seed_wiki(tmp_path: Path) -> Path:
    """Build a minimal wiki tree with one research note and one concept."""
    wiki = tmp_path / "wiki"
    (wiki / "research").mkdir(parents=True)
    (wiki / "concepts").mkdir()
    paper = wiki / "research" / "paperA.md"
    paper.write_text(
        "---\n"
        "aliases: [Paper A]\n"
        "domain: research\n"
        "category: materials\n"
        "tags: [Polymer]\n"
        "sources: []\n"
        "status: unread\n"
        "abstract: Abstract A.\n"
        "---\n"
        "# Paper A\n\n"
        "- **Authors:** Author A\n"
        "- **Year:** 2024\n\n"
        "## Abstract\nAbstract A.\n",
        encoding="utf-8",
    )
    concept = wiki / "concepts" / "Polymer.md"
    concept.write_text(
        "---\naliases: [Polymer]\ndomain: research\ncategory: concepts\ntags: []\nsources: []\nstatus: unread\n---\n# Polymer\nA concept.\n",
        encoding="utf-8",
    )
    return wiki


def test_signature_stable_for_identical_inputs(tmp_path):
    wiki = _seed_wiki(tmp_path)
    sig_a = compute_wiki_signature(wiki)
    sig_b = compute_wiki_signature(wiki)
    assert sig_a == sig_b
    assert len(sig_a) == 16  # 16 hex chars per the spec


def test_signature_changes_when_paper_added(tmp_path):
    wiki = _seed_wiki(tmp_path)
    sig_before = compute_wiki_signature(wiki)
    (wiki / "research" / "paperB.md").write_text(
        "---\n"
        "aliases: [B]\ndomain: research\ncategory: materials\ntags: []\nsources: []\nstatus: unread\nabstract: B.\n"
        "---\n# B\n- **Authors:** X\n- **Year:** 2024\n## Abstract\nB.\n",
        encoding="utf-8",
    )
    sig_after = compute_wiki_signature(wiki)
    assert sig_before != sig_after


def test_signature_changes_when_mtime_bumps(tmp_path):
    wiki = _seed_wiki(tmp_path)
    sig_before = compute_wiki_signature(wiki)
    target = wiki / "research" / "paperA.md"
    # Touch with a future mtime — bump mtime_ns so the signature changes
    # even if we don't modify content.
    new_mtime = time.time() + 10
    import os

    os.utime(target, (new_mtime, new_mtime))
    sig_after = compute_wiki_signature(wiki)
    assert sig_before != sig_after


def test_signature_changes_when_paper_removed(tmp_path):
    wiki = _seed_wiki(tmp_path)
    sig_before = compute_wiki_signature(wiki)
    (wiki / "research" / "paperA.md").unlink()
    sig_after = compute_wiki_signature(wiki)
    assert sig_before != sig_after


def test_signature_includes_facts_file(tmp_path):
    wiki = _seed_wiki(tmp_path)
    facts = tmp_path / "facts.jsonl"
    facts.write_text('{"paper_id":"paper:a"}\n', encoding="utf-8")
    sig_no_facts = compute_wiki_signature(wiki)
    sig_with_facts = compute_wiki_signature(wiki, facts_file=facts)
    assert sig_no_facts != sig_with_facts


def test_signature_roundtrip_on_disk(tmp_path):
    output_dir = tmp_path / ".graph_index"
    assert read_build_signature(output_dir) is None
    write_build_signature(output_dir, "abc123def4567890")
    assert read_build_signature(output_dir) == "abc123def4567890"


def test_signature_roundtrip_empty_file(tmp_path):
    output_dir = tmp_path / ".graph_index"
    output_dir.mkdir()
    (output_dir / "build_signature.txt").write_text("", encoding="utf-8")
    # Empty file should round-trip to None (treated as "no signature stored").
    assert read_build_signature(output_dir) is None


def test_main_build_incremental_skips_when_signature_matches(tmp_path, capsys, monkeypatch):
    from legacy.graphrag import wiki_graph

    wiki = _seed_wiki(tmp_path)
    output_dir = tmp_path / ".graph_index"

    # First build — populates graph.sqlite and writes the signature.
    wiki_graph.main(["build", "--wiki-dir", str(wiki), "--output-dir", str(output_dir)])
    capsys.readouterr()  # drain
    write_build_signature(output_dir, compute_wiki_signature(wiki))

    # Second build with --incremental — must NOT call write_index because
    # the signature matches.
    called = {"build_graph": 0, "write_index": 0}
    real_build = wiki_graph.build_graph
    real_write = wiki_graph.write_index

    def spy_build(*args, **kwargs):
        called["build_graph"] += 1
        return real_build(*args, **kwargs)

    def spy_write(*args, **kwargs):
        called["write_index"] += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(wiki_graph, "build_graph", spy_build)
    monkeypatch.setattr(wiki_graph, "write_index", spy_write)

    wiki_graph.main(
        [
            "build",
            "--wiki-dir",
            str(wiki),
            "--output-dir",
            str(output_dir),
            "--incremental",
        ]
    )
    out = capsys.readouterr().out
    assert "index_unchanged" in out
    assert called["build_graph"] == 0
    assert called["write_index"] == 0


def test_main_build_incremental_runs_full_when_signature_differs(tmp_path, capsys):
    from legacy.graphrag import wiki_graph

    wiki = _seed_wiki(tmp_path)
    output_dir = tmp_path / ".graph_index"

    # Initial build
    wiki_graph.main(["build", "--wiki-dir", str(wiki), "--output-dir", str(output_dir)])
    capsys.readouterr()

    # Add a paper → signature changes → --incremental must do full build.
    (wiki / "research" / "paperC.md").write_text(
        "---\n"
        "aliases: [C]\ndomain: research\ncategory: materials\ntags: []\nsources: []\nstatus: unread\nabstract: C.\n"
        "---\n# C\n- **Authors:** X\n- **Year:** 2024\n## Abstract\nC.\n",
        encoding="utf-8",
    )

    wiki_graph.main(
        [
            "build",
            "--wiki-dir",
            str(wiki),
            "--output-dir",
            str(output_dir),
            "--incremental",
        ]
    )
    out = capsys.readouterr().out
    # "index_written" path was taken (not "index_unchanged").
    assert "index_written" in out


def test_main_build_incremental_no_prior_signature_does_full_build(tmp_path, capsys):
    from legacy.graphrag import wiki_graph

    wiki = _seed_wiki(tmp_path)
    output_dir = tmp_path / ".graph_index"

    # First-ever --incremental run: no graph.sqlite exists yet → full build.
    wiki_graph.main(
        [
            "build",
            "--wiki-dir",
            str(wiki),
            "--output-dir",
            str(output_dir),
            "--incremental",
        ]
    )
    out = capsys.readouterr().out
    assert "index_written" in out
    assert (output_dir / "graph.sqlite").exists()
    assert (output_dir / "build_signature.txt").exists()
