import json
import sqlite3

import pytest

from legacy.graphrag.wiki_graph import (
    audit_graph,
    build_graph,
    duplicate_paper_groups,
    duplicate_paper_triage,
    render_duplicate_review_markdown,
    extract_wikilinks,
    graph_with_facts,
    main,
    parse_frontmatter,
    parse_note,
    write_index,
    write_sqlite,
    _format_list_diff,
    _safe_merge_tier,
    _action_hint,
    _assert_outside_source_layers,
)
from legacy.graphrag.wiki_semantic_schema import SemanticFact


def test_audit_graph_reports_counts():
    graph = {
        "nodes": [
            {"id": "paper:a", "kind": "Paper", "label": "A"},
            {"id": "concept:b", "kind": "Concept", "label": "B"},
            {"id": "unresolved:Missing", "kind": "Unresolved", "label": "Missing"},
        ],
        "edges": [
            {"source": "paper:a", "type": "LINKS_TO", "target": "concept:b"},
            {"source": "paper:a", "type": "LINKS_TO_UNRESOLVED", "target": "unresolved:Missing"},
        ],
    }

    summary = audit_graph(graph)

    assert summary["nodes"] == 3
    assert summary["edges"] == 2
    assert summary["kinds"]["Paper"] == 1
    assert summary["unresolved_links"] == 1


def test_main_audit_prints_summary(tmp_path, capsys):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "smith2024.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        """---
aliases: [Audit Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# Audit Paper
Links to [[MissingConcept]].
""",
        encoding="utf-8",
    )

    main(["audit", "--wiki-dir", str(wiki)])

    output = capsys.readouterr().out
    assert "nodes: 2\n" in output
    assert "edges: 1\n" in output
    assert "unresolved_links: 1\n" in output
    assert "  Paper: 1\n" in output
    assert "  LINKS_TO_UNRESOLVED: 1\n" in output


def test_extract_wikilinks_handles_aliases_and_sections():
    body = "See [[Charge_Trapping]] and [[paperKey#Methods|method section]]."

    assert extract_wikilinks(body) == ["Charge_Trapping", "paperKey"]


def test_extract_wikilinks_ignores_inline_and_fenced_code_examples():
    body = """Use `[[ExampleInline]]` as a placeholder.

```markdown
[[ExampleFence]]
```

    [[ExampleIndented]]
\t[[ExampleTabbed]]

Real link: [[Charge_Trapping]].
"""

    assert extract_wikilinks(body) == ["Charge_Trapping"]


def test_parse_note_extracts_frontmatter_body_and_node_id(tmp_path):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "smith2024.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        """---
aliases:
  - A Test Paper
domain: research
category: materials
tags:
  - Dielectric
sources:
  - zotero:smith2024
pdf_path: /tmp/smith2024.pdf
status: unread
---
# A Test Paper

## Abstract
This paper studies [[Charge_Trapping]] in elastomers.
""",
        encoding="utf-8",
    )

    parsed = parse_note(note, wiki)

    assert parsed.node_id == "paper:smith2024"
    assert parsed.kind == "Paper"
    assert parsed.title == "A Test Paper"
    assert parsed.rel_path == "research/smith2024.md"
    assert parsed.tags == ["Dielectric"]
    assert parsed.sources == ["zotero:smith2024"]
    assert parsed.pdf_path == "/tmp/smith2024.pdf"
    assert parsed.wikilinks == ["Charge_Trapping"]
    assert "## Abstract" in parsed.body
    assert "This paper studies [[Charge_Trapping]] in elastomers." in parsed.body


def test_research_materials_note_without_paper_evidence_is_not_paper(tmp_path):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "Response_to_reviewer.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        """---
aliases: [Response to reviewer]
domain: research
category: materials
tags: [review]
sources:
  - zotero:response2024
pdf_path: /tmp/response.pdf
---
# Response to reviewer

**Authors:** Unknown
**Year:** Unknown

## Abstract

""",
        encoding="utf-8",
    )

    parsed = parse_note(note, wiki)

    assert parsed.kind == "Note"
    assert parsed.node_id == "note:Response_to_reviewer"


def test_research_materials_note_with_numeric_year_is_paper(tmp_path):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "doe2024.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        """---
aliases: [Year Evidence Paper]
domain: research
category: materials
tags: [materials]
sources: []
---
# Year Evidence Paper

**Authors:** Unknown
**Year:** 2024

## Abstract

""",
        encoding="utf-8",
    )

    parsed = parse_note(note, wiki)

    assert parsed.kind == "Paper"
    assert parsed.node_id == "paper:doe2024"


def test_research_materials_note_with_bulleted_metadata_is_paper(tmp_path):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "smith2024.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        """---
aliases: [Bulleted Metadata Paper]
domain: research
category: materials
tags: [materials]
sources: []
---
# Bulleted Metadata Paper

- **Authors:** Jane Doe
- **Year:** 2023

## Abstract

""",
        encoding="utf-8",
    )

    parsed = parse_note(note, wiki)

    assert parsed.kind == "Paper"
    assert parsed.node_id == "paper:smith2024"


def test_research_paper_classifies_when_root_is_repo_not_wiki(tmp_path):
    repo_root = tmp_path / "repo"
    note = repo_root / "wiki" / "research" / "smith2024.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        """---
aliases: [Repo Root Paper]
domain: research
category: materials
tags: [materials]
sources: []
---
# Repo Root Paper

* **Authors:** Jane Doe
* **Year:** 2024

## Abstract

""",
        encoding="utf-8",
    )

    parsed = parse_note(note, repo_root)

    assert parsed.kind == "Paper"
    assert parsed.node_id == "paper:smith2024"
    assert parsed.rel_path == "wiki/research/smith2024.md"


def test_parse_frontmatter_invalid_yaml_returns_empty_metadata():
    text = """---
week: {{date:YYYY-[W]ww}}
---
# Template
"""

    metadata, body = parse_frontmatter(text)

    assert metadata == {}
    assert body == text


@pytest.mark.parametrize("category", ["index", "timeline", "moc"])
def test_research_meta_notes_are_not_classified_as_papers(tmp_path, category):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / f"01_Literature_{category}.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        f"""---
aliases: [Literature Index]
domain: research
category: {category}
tags: [{category}]
sources: []
---
# Literature Index
""",
        encoding="utf-8",
    )

    parsed = parse_note(note, wiki)

    assert parsed.kind == "Note"
    assert parsed.node_id == f"note:01_Literature_{category}"


def test_build_graph_creates_note_tag_source_pdf_and_wikilink_edges(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    concept = wiki / "concepts" / "Charge_Trapping.md"
    paper.parent.mkdir(parents=True)
    concept.parent.mkdir(parents=True)
    paper.write_text(
        """---
aliases: [A Test Paper]
domain: research
category: materials
tags: [Dielectric]
sources: [zotero:smith2024]
pdf_path: /tmp/smith2024.pdf
status: unread
abstract: Some abstract
---
# A Test Paper
Links to [[Charge_Trapping]].
""",
        encoding="utf-8",
    )
    concept.write_text(
        """---
aliases: [Charge Trapping]
domain: research
category: concepts
tags: [Charge]
sources: []
---
# Charge Trapping
""",
        encoding="utf-8",
    )

    graph = build_graph(wiki)
    node_ids = {node["id"] for node in graph["nodes"]}
    edge_keys = {(edge["source"], edge["type"], edge["target"]) for edge in graph["edges"]}

    assert "paper:smith2024" in node_ids
    assert "concept:Charge_Trapping" in node_ids
    assert "tag:Dielectric" in node_ids
    assert "source:zotero:smith2024" in node_ids
    assert "pdf:/tmp/smith2024.pdf" in node_ids
    assert ("paper:smith2024", "HAS_TAG", "tag:Dielectric") in edge_keys
    assert ("paper:smith2024", "HAS_SOURCE", "source:zotero:smith2024") in edge_keys
    assert ("paper:smith2024", "HAS_PDF", "pdf:/tmp/smith2024.pdf") in edge_keys
    assert ("paper:smith2024", "LINKS_TO", "concept:Charge_Trapping") in edge_keys


def test_build_graph_marks_duplicate_papers_by_bibliographic_identity(tmp_path):
    wiki = tmp_path / "wiki"
    research = wiki / "research"
    research.mkdir(parents=True)
    for stem, source, pdf_path in [
        ("levine2023", "zotero:first", "/tmp/first.pdf"),
        ("levine2023a", "zotero:second", "/tmp/second.pdf"),
    ]:
        (research / f"{stem}.md").write_text(
            f"""---
aliases:
  - A Low-Voltage, High-Force Capacity Electroadhesive Clutch
domain: research
category: materials
tags: []
sources:
  - {source}
pdf_path: {pdf_path}
---
# A Low-Voltage, High-Force Capacity Electroadhesive Clutch

- **Authors:** D. J. Levine, J. H. Pikul
- **Year:** 2023

## Abstract
Same abstract.
""",
            encoding="utf-8",
        )

    graph = build_graph(wiki)
    nodes = {node["id"]: node for node in graph["nodes"]}
    edge_keys = {(edge["source"], edge["type"], edge["target"]) for edge in graph["edges"]}

    assert nodes["paper:levine2023"]["canonical_paper_id"] == "paper:levine2023"
    assert nodes["paper:levine2023a"]["canonical_paper_id"] == "paper:levine2023"
    assert nodes["paper:levine2023a"]["duplicate_of"] == "paper:levine2023"
    assert ("paper:levine2023a", "DUPLICATE_OF", "paper:levine2023") in edge_keys


def test_build_graph_does_not_canonicalize_when_authors_are_missing(tmp_path):
    wiki = tmp_path / "wiki"
    research = wiki / "research"
    research.mkdir(parents=True)
    for stem in ["unknown2024a", "unknown2024b"]:
        (research / f"{stem}.md").write_text(
            """---
aliases: [Same Generic Title]
domain: research
category: materials
tags: []
sources: []
---
# Same Generic Title

- **Authors:** Unknown
- **Year:** 2024

## Abstract
Different papers can share a generic title when author metadata is missing.
""",
            encoding="utf-8",
        )

    graph = build_graph(wiki)
    nodes = {node["id"]: node for node in graph["nodes"]}
    edge_keys = {(edge["source"], edge["type"], edge["target"]) for edge in graph["edges"]}

    assert "canonical_paper_id" not in nodes["paper:unknown2024a"]
    assert "canonical_paper_id" not in nodes["paper:unknown2024b"]
    assert not any(edge_type == "DUPLICATE_OF" for _, edge_type, _ in edge_keys)


def test_build_graph_repo_root_only_indexes_wiki_directory(tmp_path):
    repo_root = tmp_path / "repo"
    paper = repo_root / "wiki" / "research" / "smith2024.md"
    docs = repo_root / "docs"
    paper.parent.mkdir(parents=True)
    docs.mkdir(parents=True)
    (repo_root / "CLAUDE.md").write_text("# Claude\n", encoding="utf-8")
    (docs / "plan.md").write_text("# Plan\n", encoding="utf-8")
    paper.write_text(
        """---
aliases: [Repo Root Graph Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# Repo Root Graph Paper
""",
        encoding="utf-8",
    )

    graph = build_graph(repo_root)
    node_ids = {node["id"] for node in graph["nodes"]}

    assert "paper:smith2024" in node_ids
    assert "note:CLAUDE" not in node_ids
    assert "note:plan" not in node_ids


def test_build_graph_skips_operational_and_generated_pages(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    literature_index = wiki / "research" / "01_Literature_Index.md"
    timeline = wiki / "research" / "02_Research_Timeline.md"
    dispatch = wiki / "dispatches" / "2026-05-01" / "example" / "draft.md"
    for path in (paper, literature_index, timeline, dispatch):
        path.parent.mkdir(parents=True, exist_ok=True)

    paper.write_text(
        """---
aliases: [Corpus Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# Corpus Paper
""",
        encoding="utf-8",
    )
    for path, title in (
        (literature_index, "Literature Index"),
        (timeline, "Research Timeline"),
        (dispatch, "Generated Draft"),
    ):
        path.write_text(
            f"""---
aliases: [{title}]
domain: research
category: index
tags: []
sources: []
---
# {title}
""",
            encoding="utf-8",
        )

    graph = build_graph(wiki)
    rel_paths = {node.get("rel_path") for node in graph["nodes"]}

    assert "research/smith2024.md" in rel_paths
    assert "research/01_Literature_Index.md" not in rel_paths
    assert "research/02_Research_Timeline.md" not in rel_paths
    assert "dispatches/2026-05-01/example/draft.md" not in rel_paths


def test_build_graph_skips_irrelevant_notes(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    irrelevant = wiki / "research" / "fixture2024.md"
    paper.parent.mkdir(parents=True)

    paper.write_text(
        """---
aliases: [Corpus Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# Corpus Paper
""",
        encoding="utf-8",
    )
    irrelevant.write_text(
        """---
aliases: [Fixture Paper]
domain: research
category: materials
tags: []
sources: []
status: irrelevant
abstract: Fixture abstract
---
# Fixture Paper
""",
        encoding="utf-8",
    )

    graph = build_graph(wiki)
    node_ids = {node["id"] for node in graph["nodes"]}

    assert "paper:smith2024" in node_ids
    assert "paper:fixture2024" not in node_ids


def test_build_graph_creates_unresolved_nodes(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    paper.parent.mkdir(parents=True)
    paper.write_text(
        """---
aliases: [Unresolved Link Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# Unresolved Link Paper
Links to [[MissingConcept]].
""",
        encoding="utf-8",
    )

    graph = build_graph(wiki)
    node_ids = {node["id"] for node in graph["nodes"]}
    edge_keys = {(edge["source"], edge["type"], edge["target"]) for edge in graph["edges"]}

    assert "unresolved:MissingConcept" in node_ids
    assert (
        "paper:smith2024",
        "LINKS_TO_UNRESOLVED",
        "unresolved:MissingConcept",
    ) in edge_keys


def test_build_graph_rejects_distinct_notes_with_same_node_id(tmp_path):
    wiki = tmp_path / "wiki"
    research_note = wiki / "research" / "same.md"
    general_note = wiki / "research" / "nested" / "same.md"
    research_note.parent.mkdir(parents=True)
    general_note.parent.mkdir(parents=True)
    research_note.write_text(
        """---
aliases: [Research Same]
domain: research
abstract: Some abstract
---
# Research Same
""",
        encoding="utf-8",
    )
    general_note.write_text(
        """---
aliases: [General Same]
domain: general
abstract: Other abstract
---
# General Same
""",
        encoding="utf-8",
    )

    try:
        build_graph(wiki)
    except ValueError as exc:
        assert "Duplicate node id" in str(exc)
    else:
        raise AssertionError("Expected duplicate node id to fail instead of silently merging")


def test_graph_with_facts_keeps_base_graph_unchanged_without_facts(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    paper.parent.mkdir(parents=True)
    paper.write_text(
        """---
aliases: [Factless Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# Factless Paper
""",
        encoding="utf-8",
    )

    graph = build_graph(wiki)

    assert graph_with_facts(graph, []) == graph


def test_graph_with_facts_adds_fact_node_and_edges_for_existing_concept_target(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    concept = wiki / "concepts" / "Charge_Trapping.md"
    paper.parent.mkdir(parents=True)
    concept.parent.mkdir(parents=True)
    paper.write_text(
        """---
aliases: [Semantic Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# Semantic Paper
Links to [[Charge_Trapping]].
""",
        encoding="utf-8",
    )
    concept.write_text(
        """---
aliases: [Charge Trapping]
domain: research
category: concepts
tags: []
sources: []
---
# Charge Trapping
""",
        encoding="utf-8",
    )
    graph = build_graph(wiki)
    facts = [
        SemanticFact(
            paper_id="paper:smith2024",
            fact_id="fact:smith2024:claim:1",
            fact_type="Claim",
            text="Charge trapping increases actuation hysteresis.",
            evidence="Observed during cycling.",
            confidence=0.81,
            targets=["concept:Charge_Trapping"],
        )
    ]

    graph_with_semantics = graph_with_facts(graph, facts)
    nodes_by_id = {node["id"]: node for node in graph_with_semantics["nodes"]}
    edge_keys = {
        (edge["source"], edge["type"], edge["target"]) for edge in graph_with_semantics["edges"]
    }

    assert nodes_by_id["fact:smith2024:claim:1"] == {
        "id": "fact:smith2024:claim:1",
        "kind": "Fact",
        "title": "Charge trapping increases actuation hysteresis.",
        "fact_type": "Claim",
        "evidence": "Observed during cycling.",
        "confidence": 0.81,
        "paper_id": "paper:smith2024",
    }
    assert ("paper:smith2024", "HAS_FACT", "fact:smith2024:claim:1") in edge_keys
    assert ("fact:smith2024:claim:1", "ABOUT", "concept:Charge_Trapping") in edge_keys
    assert (
        sum(1 for node in graph_with_semantics["nodes"] if node["id"] == "concept:Charge_Trapping")
        == 1
    )


def test_graph_with_facts_creates_placeholder_target_nodes(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    paper.parent.mkdir(parents=True)
    paper.write_text(
        """---
aliases: [Placeholder Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# Placeholder Paper
""",
        encoding="utf-8",
    )
    graph = build_graph(wiki)
    facts = [
        SemanticFact(
            paper_id="paper:smith2024",
            fact_id="fact:smith2024:metric:1",
            fact_type="Metric",
            text="Peak strain reached 253%.",
            evidence="Results table.",
            confidence=0.74,
            targets=["metric:strain"],
        )
    ]

    graph_with_semantics = graph_with_facts(graph, facts)
    nodes_by_id = {node["id"]: node for node in graph_with_semantics["nodes"]}
    edge_keys = {
        (edge["source"], edge["type"], edge["target"]) for edge in graph_with_semantics["edges"]
    }

    assert nodes_by_id["metric:strain"] == {
        "id": "metric:strain",
        "kind": "Metric",
        "title": "strain",
    }
    assert ("fact:smith2024:metric:1", "ABOUT", "metric:strain") in edge_keys


def test_graph_with_facts_rejects_unknown_fact_paper():
    graph = {"nodes": [], "edges": []}
    facts = [
        SemanticFact(
            paper_id="paper:external",
            fact_id="fact:external:claim:1",
            fact_type="Claim",
            text="External paper fact.",
            evidence="Pilot fixture.",
            confidence=0.6,
        )
    ]

    with pytest.raises(ValueError, match="unknown paper_id"):
        graph_with_facts(graph, facts)


def test_write_index_outputs_jsonl_and_sqlite(tmp_path):
    graph = {
        "nodes": [
            {
                "id": "paper:smith2024",
                "kind": "Paper",
                "title": "A Test Paper",
                "rel_path": "research/smith2024.md",
            },
            {"id": "tag:Dielectric", "kind": "Tag", "label": "Dielectric"},
        ],
        "edges": [
            {"source": "paper:smith2024", "type": "HAS_TAG", "target": "tag:Dielectric"},
        ],
    }
    output_dir = tmp_path / ".graph_index"

    write_index(graph, output_dir)

    nodes = [json.loads(line) for line in (output_dir / "nodes.jsonl").read_text().splitlines()]
    edges = [json.loads(line) for line in (output_dir / "edges.jsonl").read_text().splitlines()]
    assert nodes[0]["id"] == "paper:smith2024"
    assert edges[0]["type"] == "HAS_TAG"

    conn = sqlite3.connect(output_dir / "graph.sqlite")
    try:
        row = conn.execute(
            "select kind, label, props_json from nodes where id = ?",
            ("paper:smith2024",),
        ).fetchone()
        assert row[:2] == ("Paper", "A Test Paper")
        assert json.loads(row[2]) == {
            "title": "A Test Paper",
            "rel_path": "research/smith2024.md",
        }
        edge_row = conn.execute("select source, type, target from edges").fetchone()
        assert edge_row == ("paper:smith2024", "HAS_TAG", "tag:Dielectric")
        index_names = {row[1] for row in conn.execute("pragma index_list('edges')")}
        assert {"idx_edges_source", "idx_edges_target", "idx_edges_type"} <= index_names
    finally:
        conn.close()


def test_write_index_writes_facts_jsonl_and_fact_rows_when_facts_supplied(tmp_path):
    graph = {
        "nodes": [
            {"id": "paper:smith2024", "kind": "Paper", "title": "A Test Paper"},
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:smith2024",
            fact_id="fact:smith2024:claim:1",
            fact_type="Claim",
            text="Claim text.",
            evidence="Evidence text.",
            confidence=0.66,
            targets=["metric:strain"],
        )
    ]
    output_dir = tmp_path / ".graph_index"

    write_index(graph, output_dir, facts=facts)

    fact_rows = [
        json.loads(line)
        for line in (output_dir / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert fact_rows == [facts[0].to_dict()]

    conn = sqlite3.connect(output_dir / "graph.sqlite")
    try:
        node_ids = {row[0] for row in conn.execute("select id from nodes")}
        edge_keys = {
            (row[0], row[1], row[2])
            for row in conn.execute("select source, type, target from edges")
        }
        assert "fact:smith2024:claim:1" in node_ids
        assert "metric:strain" in node_ids
        assert ("paper:smith2024", "HAS_FACT", "fact:smith2024:claim:1") in edge_keys
        assert ("fact:smith2024:claim:1", "ABOUT", "metric:strain") in edge_keys
    finally:
        conn.close()


def test_duplicate_paper_groups_reads_generated_index(tmp_path):
    graph = {
        "nodes": [
            {
                "id": "paper:canonical",
                "kind": "Paper",
                "title": "Canonical Paper",
                "rel_path": "research/canonical.md",
                "canonical_paper_id": "paper:canonical",
            },
            {
                "id": "paper:duplicate",
                "kind": "Paper",
                "title": "Canonical Paper",
                "rel_path": "research/duplicate.md",
                "canonical_paper_id": "paper:canonical",
                "duplicate_of": "paper:canonical",
            },
        ],
        "edges": [
            {
                "source": "paper:duplicate",
                "type": "DUPLICATE_OF",
                "target": "paper:canonical",
            }
        ],
    }
    output_dir = tmp_path / ".graph_index"
    write_index(graph, output_dir)

    report = duplicate_paper_groups(output_dir)

    assert report == {
        "summary": {"groups": 1, "duplicate_papers": 1},
        "groups": [
            {
                "canonical_paper_id": "paper:canonical",
                "canonical_title": "Canonical Paper",
                "canonical_path": "research/canonical.md",
                "duplicates": [
                    {
                        "paper_id": "paper:duplicate",
                        "title": "Canonical Paper",
                        "path": "research/duplicate.md",
                    }
                ],
            }
        ],
    }


def test_main_duplicates_prints_json_report(tmp_path, capsys):
    graph = {
        "nodes": [
            {"id": "paper:a", "kind": "Paper", "title": "A", "rel_path": "research/a.md"},
            {"id": "paper:b", "kind": "Paper", "title": "A", "rel_path": "research/b.md"},
        ],
        "edges": [{"source": "paper:b", "type": "DUPLICATE_OF", "target": "paper:a"}],
    }
    output_dir = tmp_path / ".graph_index"
    write_index(graph, output_dir)

    main(["duplicates", "--output-dir", str(output_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"] == {"groups": 1, "duplicate_papers": 1}
    assert payload["groups"][0]["canonical_paper_id"] == "paper:a"


def test_duplicate_paper_triage_prioritizes_metadata_differences(tmp_path):
    graph = {
        "nodes": [
            {
                "id": "paper:canonical",
                "kind": "Paper",
                "title": "Canonical Paper",
                "rel_path": "research/canonical.md",
                "status": "unread",
                "canonical_paper_id": "paper:canonical",
            },
            {
                "id": "paper:duplicate",
                "kind": "Paper",
                "title": "Canonical Paper",
                "rel_path": "research/duplicate.md",
                "status": "read",
                "canonical_paper_id": "paper:canonical",
                "duplicate_of": "paper:canonical",
            },
            {"id": "source:zotero:a", "kind": "Source", "title": "zotero:a"},
            {"id": "source:zotero:b", "kind": "Source", "title": "zotero:b"},
            {"id": "pdf:/tmp/a.pdf", "kind": "PDF", "title": "/tmp/a.pdf", "path": "/tmp/a.pdf"},
            {"id": "pdf:/tmp/b.pdf", "kind": "PDF", "title": "/tmp/b.pdf", "path": "/tmp/b.pdf"},
            {"id": "tag:A", "kind": "Tag", "title": "A"},
            {"id": "tag:B", "kind": "Tag", "title": "B"},
        ],
        "edges": [
            {"source": "paper:duplicate", "type": "DUPLICATE_OF", "target": "paper:canonical"},
            {"source": "paper:canonical", "type": "HAS_SOURCE", "target": "source:zotero:a"},
            {"source": "paper:duplicate", "type": "HAS_SOURCE", "target": "source:zotero:b"},
            {"source": "paper:canonical", "type": "HAS_PDF", "target": "pdf:/tmp/a.pdf"},
            {"source": "paper:duplicate", "type": "HAS_PDF", "target": "pdf:/tmp/b.pdf"},
            {"source": "paper:canonical", "type": "HAS_TAG", "target": "tag:A"},
            {"source": "paper:duplicate", "type": "HAS_TAG", "target": "tag:B"},
        ],
    }
    output_dir = tmp_path / ".graph_index"
    write_index(graph, output_dir)

    report = duplicate_paper_triage(output_dir)

    assert report["summary"] == {
        "groups": 1,
        "duplicate_papers": 1,
        "manual_merge_review": 1,
        "generated_only_ok": 0,
    }
    item = report["items"][0]
    assert item["action"] == "manual_merge_review"
    assert item["priority"] == "high"
    assert item["reasons"] == [
        "different_sources",
        "different_pdf_paths",
        "different_tags",
        "different_status",
    ]
    assert item["canonical"]["paper_id"] == "paper:canonical"
    assert item["duplicate"]["paper_id"] == "paper:duplicate"


def test_duplicate_paper_triage_marks_matching_metadata_as_generated_only(tmp_path):
    graph = {
        "nodes": [
            {
                "id": "paper:a",
                "kind": "Paper",
                "title": "A",
                "rel_path": "research/a.md",
                "canonical_paper_id": "paper:a",
            },
            {
                "id": "paper:b",
                "kind": "Paper",
                "title": "A",
                "rel_path": "research/b.md",
                "canonical_paper_id": "paper:a",
                "duplicate_of": "paper:a",
            },
        ],
        "edges": [{"source": "paper:b", "type": "DUPLICATE_OF", "target": "paper:a"}],
    }
    output_dir = tmp_path / ".graph_index"
    write_index(graph, output_dir)

    report = duplicate_paper_triage(output_dir)

    assert report["summary"]["generated_only_ok"] == 1
    assert report["items"][0]["action"] == "generated_only_ok"
    assert report["items"][0]["priority"] == "low"
    assert report["items"][0]["reasons"] == []


def test_main_duplicate_triage_prints_json_report(tmp_path, capsys):
    graph = {
        "nodes": [
            {"id": "paper:a", "kind": "Paper", "title": "A", "rel_path": "research/a.md"},
            {"id": "paper:b", "kind": "Paper", "title": "A", "rel_path": "research/b.md"},
        ],
        "edges": [{"source": "paper:b", "type": "DUPLICATE_OF", "target": "paper:a"}],
    }
    output_dir = tmp_path / ".graph_index"
    write_index(graph, output_dir)

    main(["duplicate-triage", "--output-dir", str(output_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["duplicate_papers"] == 1
    assert payload["items"][0]["action"] == "generated_only_ok"


def test_render_duplicate_review_markdown_sorts_safe_first_and_formats_diff():
    report = {
        "summary": {
            "groups": 2,
            "duplicate_papers": 2,
            "manual_merge_review": 1,
            "generated_only_ok": 1,
        },
        "items": [
            {
                "canonical_paper_id": "paper:canonical",
                "duplicate_paper_id": "paper:duplicate",
                "action": "manual_merge_review",
                "priority": "high",
                "reasons": ["different_sources", "different_pdf_paths"],
                "canonical": {
                    "paper_id": "paper:canonical",
                    "title": "Canonical Paper",
                    "path": "research/canonical.md",
                    "status": "stage",
                    "sources": ["zotero:a"],
                    "pdf_paths": ["/tmp/a.pdf"],
                    "tags": ["A"],
                    "fact_count": 3,
                },
                "duplicate": {
                    "paper_id": "paper:duplicate",
                    "title": "Canonical Paper",
                    "path": "research/duplicate.md",
                    "status": "unread",
                    "sources": ["zotero:b"],
                    "pdf_paths": ["/tmp/b.pdf"],
                    "tags": ["B"],
                    "fact_count": 1,
                },
            },
            {
                "canonical_paper_id": "paper:a",
                "duplicate_paper_id": "paper:b",
                "action": "generated_only_ok",
                "priority": "low",
                "reasons": [],
                "canonical": {
                    "paper_id": "paper:a",
                    "title": "A",
                    "path": "research/a.md",
                    "fact_count": 0,
                },
                "duplicate": {
                    "paper_id": "paper:b",
                    "title": "A",
                    "path": "research/b.md",
                    "fact_count": 0,
                },
            },
        ],
    }

    markdown = render_duplicate_review_markdown(report)

    assert markdown.startswith("# Duplicate Paper Review\n")
    assert "- duplicate papers: 2" in markdown
    assert "- displayed items: 2 / 2" in markdown
    safe_idx = markdown.index("## 1. A")
    manual_idx = markdown.index("## 2. Canonical Paper")
    assert safe_idx < manual_idx
    assert "**Suggestion:** Safe to fold (generated layer only)" in markdown
    assert "[[canonical]]" in markdown
    assert "[[duplicate]]" in markdown
    assert "[research/canonical.md](research/canonical.md)" not in markdown
    assert "**Reasons:** different_sources, different_pdf_paths" in markdown
    assert "**+** zotero:a" in markdown
    assert "**+** zotero:b" in markdown
    assert "| Facts | 3 | 1 |" in markdown


def test_render_duplicate_review_markdown_limit_truncates_after_sort():
    report = {
        "summary": {
            "groups": 2,
            "duplicate_papers": 2,
            "manual_merge_review": 1,
            "generated_only_ok": 1,
        },
        "items": [
            {
                "canonical_paper_id": "paper:canonical",
                "duplicate_paper_id": "paper:duplicate",
                "action": "manual_merge_review",
                "priority": "high",
                "reasons": ["different_sources", "different_pdf_paths"],
                "canonical": {
                    "paper_id": "paper:canonical",
                    "title": "Canonical Paper",
                    "path": "research/canonical.md",
                },
                "duplicate": {
                    "paper_id": "paper:duplicate",
                    "title": "Canonical Paper",
                    "path": "research/duplicate.md",
                },
            },
            {
                "canonical_paper_id": "paper:a",
                "duplicate_paper_id": "paper:b",
                "action": "generated_only_ok",
                "priority": "low",
                "reasons": [],
                "canonical": {"paper_id": "paper:a", "title": "A", "path": "research/a.md"},
                "duplicate": {"paper_id": "paper:b", "title": "A", "path": "research/b.md"},
            },
        ],
    }

    markdown = render_duplicate_review_markdown(report, limit=1)

    assert "- displayed items: 1 / 2" in markdown
    assert "## 1. A" in markdown
    assert "## 2." not in markdown
    assert "Canonical Paper" not in markdown


def test_main_duplicate_report_writes_markdown_file(tmp_path):
    graph = {
        "nodes": [
            {"id": "paper:a", "kind": "Paper", "title": "A", "rel_path": "research/a.md"},
            {"id": "paper:b", "kind": "Paper", "title": "A", "rel_path": "research/b.md"},
        ],
        "edges": [{"source": "paper:b", "type": "DUPLICATE_OF", "target": "paper:a"}],
    }
    index_dir = tmp_path / ".graph_index"
    output = tmp_path / "duplicate-review.md"
    write_index(graph, index_dir)

    main(
        [
            "duplicate-report",
            "--output-dir",
            str(index_dir),
            "--output",
            str(output),
            "--limit",
            "1",
        ]
    )

    markdown = output.read_text(encoding="utf-8")
    assert markdown.startswith("# Duplicate Paper Review\n")
    assert "paper:b" in markdown


def test_write_index_without_facts_removes_existing_facts_jsonl(tmp_path):
    graph = {
        "nodes": [{"id": "paper:smith2024", "kind": "Paper", "title": "A Test Paper"}],
        "edges": [],
    }
    output_dir = tmp_path / ".graph_index"
    facts_path = output_dir / "facts.jsonl"
    output_dir.mkdir()
    fact = SemanticFact(
        paper_id="paper:smith2024",
        fact_id="fact:smith2024:claim:1",
        fact_type="Claim",
        text="Claim text.",
        evidence="Evidence text.",
        confidence=0.66,
        targets=["metric:strain"],
    )
    facts_path.write_text(json.dumps(fact.to_dict(), sort_keys=True) + "\n", encoding="utf-8")

    write_index(graph, output_dir)

    assert not facts_path.exists()

    conn = sqlite3.connect(output_dir / "graph.sqlite")
    try:
        node_ids = {row[0] for row in conn.execute("select id from nodes")}
        edge_keys = {
            (row[0], row[1], row[2])
            for row in conn.execute("select source, type, target from edges")
        }
        assert "fact:smith2024:claim:1" not in node_ids
        assert ("paper:smith2024", "HAS_FACT", "fact:smith2024:claim:1") not in edge_keys
    finally:
        conn.close()


def test_write_index_round_trips_build_graph_output(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    concept = wiki / "concepts" / "Charge_Trapping.md"
    paper.parent.mkdir(parents=True)
    concept.parent.mkdir(parents=True)
    paper.write_text(
        """---
aliases: [A Test Paper]
domain: research
category: materials
tags: [Dielectric]
sources: []
abstract: Some abstract
---
# A Test Paper
Links to [[Charge_Trapping]].
""",
        encoding="utf-8",
    )
    concept.write_text(
        """---
aliases: [Charge Trapping]
domain: research
category: concepts
tags: []
sources: []
---
# Charge Trapping
""",
        encoding="utf-8",
    )
    graph = build_graph(wiki)

    write_index(graph, tmp_path / ".graph_index")

    conn = sqlite3.connect(tmp_path / ".graph_index" / "graph.sqlite")
    try:
        node_count = conn.execute("select count(*) from nodes").fetchone()[0]
        edge_count = conn.execute("select count(*) from edges").fetchone()[0]
        assert node_count == len(graph["nodes"])
        assert edge_count == len(graph["edges"])
        row = conn.execute(
            "select label, props_json from nodes where id = ?",
            ("paper:smith2024",),
        ).fetchone()
        assert row[0] == "A Test Paper"
        assert json.loads(row[1])["rel_path"] == "research/smith2024.md"
    finally:
        conn.close()


def test_write_sqlite_preserves_existing_database_on_failure(tmp_path):
    sqlite_path = tmp_path / "graph.sqlite"
    write_sqlite(
        sqlite_path,
        {
            "nodes": [{"id": "paper:smith2024", "kind": "Paper", "label": "Original"}],
            "edges": [],
        },
    )

    with pytest.raises(TypeError):
        write_sqlite(
            sqlite_path,
            {
                "nodes": [
                    {
                        "id": "paper:broken",
                        "kind": "Paper",
                        "label": "Broken",
                        "bad": object(),
                    }
                ],
                "edges": [],
            },
        )

    conn = sqlite3.connect(sqlite_path)
    try:
        row = conn.execute("select id, label from nodes").fetchone()
        assert row == ("paper:smith2024", "Original")
    finally:
        conn.close()
    assert not sqlite_path.with_suffix(sqlite_path.suffix + ".tmp").exists()


def test_main_build_reads_facts_file_when_supplied(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    paper.parent.mkdir(parents=True)
    paper.write_text(
        """---
aliases: [CLI Semantic Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# CLI Semantic Paper
""",
        encoding="utf-8",
    )
    facts_file = tmp_path / "facts.jsonl"
    fact = SemanticFact(
        paper_id="paper:smith2024",
        fact_id="fact:smith2024:claim:1",
        fact_type="Claim",
        text="CLI claim text.",
        evidence="CLI evidence.",
        confidence=0.77,
        targets=["metric:strain"],
    )
    facts_file.write_text(json.dumps(fact.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
    output_dir = tmp_path / ".graph_index"

    main(
        [
            "build",
            "--wiki-dir",
            str(wiki),
            "--output-dir",
            str(output_dir),
            "--facts-file",
            str(facts_file),
        ]
    )

    node_rows = [
        json.loads(line)
        for line in (output_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["id"] == "fact:smith2024:claim:1" for row in node_rows)
    assert (output_dir / "facts.jsonl").exists()


def test_main_build_reads_exact_custom_facts_file_path(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    paper.parent.mkdir(parents=True)
    paper.write_text(
        """---
aliases: [CLI Semantic Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# CLI Semantic Paper
""",
        encoding="utf-8",
    )
    custom_facts_file = tmp_path / "custom_facts.jsonl"
    fact = SemanticFact(
        paper_id="paper:smith2024",
        fact_id="fact:smith2024:claim:custom",
        fact_type="Claim",
        text="Custom facts-file claim.",
        evidence="Custom facts-file evidence.",
        confidence=0.77,
        targets=["metric:strain"],
    )
    custom_facts_file.write_text(
        json.dumps(fact.to_dict(), sort_keys=True) + "\n", encoding="utf-8"
    )
    sibling_fact = SemanticFact(
        paper_id="paper:smith2024",
        fact_id="fact:smith2024:claim:sibling",
        fact_type="Claim",
        text="Sibling claim should not be loaded for custom files.",
        evidence="Sibling evidence.",
        confidence=0.77,
        targets=["metric:strain"],
    )
    (tmp_path / "pdf_facts.jsonl").write_text(
        json.dumps(sibling_fact.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / ".graph_index"

    main(
        [
            "build",
            "--wiki-dir",
            str(wiki),
            "--output-dir",
            str(output_dir),
            "--facts-file",
            str(custom_facts_file),
        ]
    )

    node_rows = [
        json.loads(line)
        for line in (output_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["id"] == "fact:smith2024:claim:custom" for row in node_rows)
    assert not any(row["id"] == "fact:smith2024:claim:sibling" for row in node_rows)


def test_main_build_merges_sibling_pdf_and_llm_fact_files(tmp_path):
    wiki = tmp_path / "wiki"
    paper = wiki / "research" / "smith2024.md"
    paper.parent.mkdir(parents=True)
    paper.write_text(
        """---
aliases: [CLI Semantic Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Some abstract
---
# CLI Semantic Paper
""",
        encoding="utf-8",
    )
    graph_index = tmp_path / ".graph_index"
    graph_index.mkdir()
    for filename, suffix in [
        ("facts.jsonl", "wiki"),
        ("pdf_facts.jsonl", "pdf"),
        ("facts_llm_augment.jsonl", "llm"),
    ]:
        fact = SemanticFact(
            paper_id="paper:smith2024",
            fact_id=f"fact:smith2024:claim:{suffix}",
            fact_type="Claim",
            text=f"{suffix} fact.",
            evidence=f"{suffix} evidence.",
            confidence=0.77,
            targets=["metric:strain"],
        )
        (graph_index / filename).write_text(
            json.dumps(fact.to_dict(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    main(
        [
            "build",
            "--wiki-dir",
            str(wiki),
            "--output-dir",
            str(graph_index),
            "--facts-file",
            str(graph_index / "facts.jsonl"),
        ]
    )

    written_facts = {
        json.loads(line)["fact_id"]
        for line in (graph_index / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert written_facts == {
        "fact:smith2024:claim:wiki",
        "fact:smith2024:claim:pdf",
        "fact:smith2024:claim:llm",
    }


def test_format_list_diff_marks_added_and_removed():
    canonical_cell, duplicate_cell = _format_list_diff(["a", "b"], ["b", "c"])
    assert "b" in canonical_cell and "b" in duplicate_cell
    assert "**+** a" in canonical_cell
    assert "**−** a" in duplicate_cell
    assert "**−** c" in canonical_cell
    assert "**+** c" in duplicate_cell


def test_format_list_diff_collapses_equal_lists():
    canonical_cell, duplicate_cell = _format_list_diff(["a"], ["a"])
    assert "_(same)_" in canonical_cell
    assert canonical_cell == duplicate_cell


def test_safe_merge_tier_orders_safest_first():
    assert _safe_merge_tier([]) == 0
    assert _safe_merge_tier(["different_tags"]) == 1
    assert _safe_merge_tier(["different_status"]) == 2
    assert _safe_merge_tier(["different_pdf_paths"]) == 3
    assert _safe_merge_tier(["different_sources"]) == 4
    assert _safe_merge_tier(["different_tags", "different_pdf_paths"]) == 5


def test_action_hint_prefers_paper_with_abstract():
    canonical = {"body_words": 100, "has_abstract": False, "doi": ""}
    duplicate = {"body_words": 100, "has_abstract": True, "doi": ""}
    hint = _action_hint(canonical, duplicate, ["different_tags"])
    assert "keep duplicate" in hint
    assert "has abstract" in hint


def test_action_hint_recommends_longer_body_when_abstracts_match():
    canonical = {"body_words": 200, "has_abstract": True, "doi": ""}
    duplicate = {"body_words": 50, "has_abstract": True, "doi": ""}
    hint = _action_hint(canonical, duplicate, ["different_tags"])
    assert "keep canonical" in hint
    assert "longer body" in hint


def test_action_hint_safe_for_no_reasons():
    hint = _action_hint({"body_words": 0}, {"body_words": 0}, [])
    assert hint == "Safe to fold (generated layer only)"


def test_assert_outside_source_layers_rejects_path_inside_wiki(tmp_path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    target = wiki_dir / "research" / "duplicate-review.md"
    with pytest.raises(SystemExit, match="source layer"):
        _assert_outside_source_layers(target, wiki_dir)


def test_assert_outside_source_layers_rejects_raw_layer(tmp_path):
    wiki_dir = tmp_path / "wiki"
    raw_dir = tmp_path / "raw"
    wiki_dir.mkdir()
    raw_dir.mkdir()
    target = raw_dir / "leak.md"
    with pytest.raises(SystemExit, match="source layer"):
        _assert_outside_source_layers(target, wiki_dir)


def test_assert_outside_source_layers_allows_generated_index(tmp_path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    target = tmp_path / ".graph_index" / "duplicate-review.md"
    _assert_outside_source_layers(target, wiki_dir)


def test_main_duplicate_report_refuses_writing_into_wiki(tmp_path):
    wiki_dir = tmp_path / "wiki"
    research_dir = wiki_dir / "research"
    research_dir.mkdir(parents=True)
    graph = {
        "nodes": [
            {"id": "paper:a", "kind": "Paper", "title": "A", "rel_path": "research/a.md"},
            {"id": "paper:b", "kind": "Paper", "title": "A", "rel_path": "research/b.md"},
        ],
        "edges": [{"source": "paper:b", "type": "DUPLICATE_OF", "target": "paper:a"}],
    }
    index_dir = tmp_path / ".graph_index"
    write_index(graph, index_dir)
    leak_target = research_dir / "duplicate-review.md"

    with pytest.raises(SystemExit, match="source layer"):
        main(
            [
                "duplicate-report",
                "--wiki-dir",
                str(wiki_dir),
                "--output-dir",
                str(index_dir),
                "--output",
                str(leak_target),
            ]
        )
    assert not leak_target.exists()


def test_main_duplicate_report_without_limit_emits_all_groups(tmp_path):
    graph = {
        "nodes": [
            {"id": "paper:a", "kind": "Paper", "title": "A", "rel_path": "research/a.md"},
            {"id": "paper:b", "kind": "Paper", "title": "A", "rel_path": "research/b.md"},
            {"id": "paper:c", "kind": "Paper", "title": "C", "rel_path": "research/c.md"},
            {"id": "paper:d", "kind": "Paper", "title": "C", "rel_path": "research/d.md"},
        ],
        "edges": [
            {"source": "paper:b", "type": "DUPLICATE_OF", "target": "paper:a"},
            {"source": "paper:d", "type": "DUPLICATE_OF", "target": "paper:c"},
        ],
    }
    index_dir = tmp_path / ".graph_index"
    output = tmp_path / "duplicate-review.md"
    write_index(graph, index_dir)

    main(
        [
            "duplicate-report",
            "--output-dir",
            str(index_dir),
            "--output",
            str(output),
        ]
    )

    markdown = output.read_text(encoding="utf-8")
    assert "- displayed items: 2 / 2" in markdown
    assert "[[a]]" in markdown
    assert "[[c]]" in markdown
