import json
import sqlite3
import sys

import pytest

from scripts.search.fact_query import (
    normalize_query_token,
    query_facts,
    tokenize_fact_query,
)
from scripts.search.wiki_fact_embeddings import build_fact_embedding_index
from legacy.graphrag.wiki_graph import write_index
from legacy.graphrag.wiki_index_manifest import write_manifest
from legacy.graphrag.wiki_query import (
    enrich_fact_results,
    fact_results_to_context_bundle,
    main,
    query_near,
)
import legacy.graphrag.wiki_query as wiki_query
from legacy.graphrag.wiki_semantic_schema import SemanticFact


class FakeEmbeddingProvider:
    model_name = "fake-embeddings"

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lower = text.lower()
        if "droplet" in lower or "liquid" in lower:
            return [1.0, 0.0]
        if "charge" in lower:
            return [0.0, 1.0]
        return [0.5, 0.5]


def test_normalize_query_token_keeps_scientific_plural_stems():
    assert normalize_query_token("electrodes") == "electrode"


def test_tokenize_fact_query_keeps_hyphenated_terms_and_parts():
    assert tokenize_fact_query("Low-Voltage High-Force") == [
        "low-voltage",
        "low",
        "voltage",
        "high-force",
        "high",
        "force",
    ]


def test_query_near_returns_neighbor_nodes(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:smith2024",
                "kind": "Paper",
                "label": "A Test Paper",
                "rel_path": "research/smith2024.md",
            },
            {
                "id": "concept:Charge_Trapping",
                "kind": "Concept",
                "label": "Charge Trapping",
                "rel_path": "concepts/Charge_Trapping.md",
            },
        ],
        "edges": [
            {
                "source": "paper:smith2024",
                "type": "LINKS_TO",
                "target": "concept:Charge_Trapping",
            },
        ],
    }
    write_index(graph, index_dir)

    results = query_near(index_dir, "paper:smith2024")

    assert results == [
        {
            "id": "concept:Charge_Trapping",
            "kind": "Concept",
            "label": "Charge Trapping",
            "rel_path": "concepts/Charge_Trapping.md",
            "edge_type": "LINKS_TO",
            "direction": "out",
        }
    ]


def test_query_near_returns_outgoing_before_incoming(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:center",
                "kind": "Paper",
                "label": "Center Paper",
                "rel_path": "research/center.md",
            },
            {
                "id": "concept:out",
                "kind": "Concept",
                "label": "Out Concept",
                "rel_path": "concepts/out.md",
            },
            {
                "id": "paper:incoming",
                "kind": "Paper",
                "label": "Incoming Paper",
                "rel_path": "research/incoming.md",
            },
        ],
        "edges": [
            {"source": "paper:center", "type": "LINKS_TO", "target": "concept:out"},
            {"source": "paper:incoming", "type": "LINKS_TO", "target": "paper:center"},
        ],
    }
    write_index(graph, index_dir)

    results = query_near(index_dir, "paper:center")

    assert [result["direction"] for result in results] == ["out", "in"]
    assert [result["id"] for result in results] == ["concept:out", "paper:incoming"]


def test_query_near_limit_applies_across_directions(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:center", "kind": "Paper", "label": "Center Paper"},
            {"id": "concept:out", "kind": "Concept", "label": "Out Concept"},
            {"id": "paper:incoming", "kind": "Paper", "label": "Incoming Paper"},
        ],
        "edges": [
            {"source": "paper:center", "type": "LINKS_TO", "target": "concept:out"},
            {"source": "paper:incoming", "type": "LINKS_TO", "target": "paper:center"},
        ],
    }
    write_index(graph, index_dir)

    results = query_near(index_dir, "paper:center", limit=1)

    assert [result["direction"] for result in results] == ["out"]
    assert [result["id"] for result in results] == ["concept:out"]


def test_main_dispatches_near_query(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:center", "kind": "Paper", "label": "Center Paper"},
            {"id": "concept:out", "kind": "Concept", "label": "Out Concept"},
        ],
        "edges": [
            {"source": "paper:center", "type": "LINKS_TO", "target": "concept:out"},
        ],
    }
    write_index(graph, index_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["wiki_query.py", "--near", "paper:center", "--graph-index", str(index_dir)],
    )

    main()

    output = capsys.readouterr().out
    assert "Direction" in output
    assert "LINKS_TO" in output
    assert "Out Concept" in output


def test_main_dispatches_near_query_as_json(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:center", "kind": "Paper", "label": "Center Paper"},
            {"id": "concept:out", "kind": "Concept", "label": "Out Concept"},
        ],
        "edges": [
            {"source": "paper:center", "type": "LINKS_TO", "target": "concept:out"},
        ],
    }
    write_index(graph, index_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_query.py",
            "--near",
            "paper:center",
            "--graph-index",
            str(index_dir),
            "--json",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "near": "paper:center",
        "results": [
            {
                "direction": "out",
                "edge_type": "LINKS_TO",
                "id": "concept:out",
                "kind": "Concept",
                "label": "Out Concept",
                "rel_path": "",
            }
        ],
    }


def test_main_facts_rejects_near_and_facts_together(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:center", "kind": "Paper", "label": "Center Paper"},
            {"id": "concept:out", "kind": "Concept", "label": "Out Concept"},
        ],
        "edges": [{"source": "paper:center", "type": "LINKS_TO", "target": "concept:out"}],
    }
    write_index(graph, index_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_query.py",
            "--near",
            "paper:center",
            "--facts",
            "--graph-index",
            str(index_dir),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    assert (
        "Cannot combine --near with --facts. Use one query mode at a time."
        in capsys.readouterr().out
    )


def test_main_near_missing_graph_index_prints_error(monkeypatch, capsys, tmp_path):
    missing_index = tmp_path / "missing_graph_index"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_query.py",
            "--near",
            "paper:center",
            "--graph-index",
            str(missing_index),
        ],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected missing graph index to exit non-zero")

    output = capsys.readouterr().out
    assert "Graph index not found" in output


def test_query_facts_filters_by_paper_target_type_and_text(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:center", "kind": "Paper", "label": "Center Paper"},
            {"id": "paper:other", "kind": "Paper", "label": "Other Paper"},
            {"id": "concept:Charge_Trapping", "kind": "Concept", "label": "Charge Trapping"},
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:center",
            fact_id="fact:center:claim:1",
            fact_type="Claim",
            text="Charge trapping increases hysteresis.",
            evidence="Hysteresis increased after cycling.",
            confidence=0.82,
            targets=["concept:Charge_Trapping"],
        ),
        SemanticFact(
            paper_id="paper:other",
            fact_id="fact:other:method:1",
            fact_type="Method",
            text="Used solvent casting.",
            evidence="Methods section.",
            confidence=0.7,
            targets=["method:casting"],
        ),
    ]
    write_index(graph, index_dir, facts=facts)

    results = query_facts(
        index_dir,
        paper_id="paper:center",
        target="concept:Charge_Trapping",
        fact_type="Claim",
        text="hysteresis",
    )

    assert results == [
        {
            "fact_id": "fact:center:claim:1",
            "paper_id": "paper:center",
            "fact_type": "Claim",
            "text": "Charge trapping increases hysteresis.",
            "evidence": "Hysteresis increased after cycling.",
            "confidence": 0.82,
            "targets": ["concept:Charge_Trapping"],
        }
    ]


def test_query_facts_bulk_loads_about_targets(monkeypatch, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:center", "kind": "Paper", "label": "Center Paper"},
            {"id": "concept:Charge_Trapping", "kind": "Concept", "label": "Charge Trapping"},
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:center",
            fact_id=f"fact:center:claim:{index}",
            fact_type="Claim",
            text=f"Charge trapping finding {index}.",
            evidence="Observed in cycling.",
            confidence=0.8,
            targets=["concept:Charge_Trapping"],
        )
        for index in range(10)
    ]
    write_index(graph, index_dir, facts=facts)

    original_connect = sqlite3.connect
    executed_sql: list[str] = []

    class CountingConnection:
        def __init__(self, path):
            self._conn = original_connect(path)

        @property
        def row_factory(self):
            return self._conn.row_factory

        @row_factory.setter
        def row_factory(self, value):
            self._conn.row_factory = value

        def execute(self, sql, parameters=()):
            executed_sql.append(" ".join(sql.split()))
            return self._conn.execute(sql, parameters)

        def close(self):
            self._conn.close()

    monkeypatch.setattr(wiki_query.sqlite3, "connect", CountingConnection)

    results = query_facts(index_dir, target="concept:Charge_Trapping", text="charge")

    per_fact_about_queries = [
        sql
        for sql in executed_sql
        if sql == "select target from edges where source = ? and type = 'ABOUT' order by target"
    ]
    assert len(results) == 10
    assert per_fact_about_queries == []


def test_query_facts_filters_by_canonical_paper_id(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:canonical",
                "kind": "Paper",
                "label": "Canonical Paper",
                "canonical_paper_id": "paper:canonical",
            },
            {
                "id": "paper:duplicate",
                "kind": "Paper",
                "label": "Duplicate Paper",
                "canonical_paper_id": "paper:canonical",
                "duplicate_of": "paper:canonical",
            },
            {"id": "concept:Charge_Trapping", "kind": "Concept", "label": "Charge Trapping"},
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:canonical",
            fact_id="fact:canonical:claim:1",
            fact_type="Claim",
            text="Canonical fact text.",
            evidence="Canonical evidence.",
            confidence=0.8,
            targets=["concept:Charge_Trapping"],
        ),
        SemanticFact(
            paper_id="paper:duplicate",
            fact_id="fact:duplicate:claim:1",
            fact_type="Claim",
            text="Duplicate fact text.",
            evidence="Duplicate evidence.",
            confidence=0.8,
            targets=["concept:Charge_Trapping"],
        ),
    ]
    write_index(graph, index_dir, facts=facts)

    results = query_facts(index_dir, paper_id="paper:canonical")

    assert [result["fact_id"] for result in results] == [
        "fact:canonical:claim:1",
        "fact:duplicate:claim:1",
    ]


def test_enrich_fact_results_adds_paper_note_and_pdf_metadata(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:center",
                "kind": "Paper",
                "label": "Center Paper",
                "rel_path": "research/center.md",
                "status": "unread",
                "canonical_paper_id": "paper:canonical",
                "duplicate_of": "paper:canonical",
            },
            {
                "id": "pdf:/tmp/center.pdf",
                "kind": "PDF",
                "label": "/tmp/center.pdf",
                "path": "/tmp/center.pdf",
            },
        ],
        "edges": [{"source": "paper:center", "type": "HAS_PDF", "target": "pdf:/tmp/center.pdf"}],
    }
    facts = [
        SemanticFact(
            paper_id="paper:center",
            fact_id="fact:center:claim:1",
            fact_type="Claim",
            text="Low voltage actuation was reported.",
            evidence="Reported at low voltage.",
            confidence=0.76,
            targets=["metric:voltage"],
        )
    ]
    write_index(graph, index_dir, facts=facts)

    enriched = enrich_fact_results(
        index_dir,
        [
            {
                "fact_id": "fact:center:claim:1",
                "paper_id": "paper:center",
                "fact_type": "Claim",
                "text": "Low voltage actuation was reported.",
                "evidence": "Reported at low voltage.",
                "confidence": 0.76,
                "targets": ["metric:voltage"],
            }
        ],
    )

    assert enriched[0]["paper_title"] == "Center Paper"
    assert enriched[0]["paper_path"] == "research/center.md"
    assert enriched[0]["paper_status"] == "unread"
    assert enriched[0]["canonical_paper_id"] == "paper:canonical"
    assert enriched[0]["duplicate_of"] == "paper:canonical"
    assert enriched[0]["pdf_path"] == "/tmp/center.pdf"


def test_query_facts_matches_domain_paraphrase_aliases(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:charge", "kind": "Paper", "label": "Charge Paper"},
            {"id": "paper:lens", "kind": "Paper", "label": "Lens Paper"},
            {"id": "concept:Charge_Trapping", "kind": "Concept", "label": "Charge Trapping"},
            {"id": "metric:voltage", "kind": "Metric", "label": "Voltage"},
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:charge",
            fact_id="fact:charge:claim:1",
            fact_type="Claim",
            text="The energy level of the charge trap depends on crystallite structure.",
            evidence="Energy level and crystallite structure.",
            confidence=0.8,
            targets=["concept:Charge_Trapping"],
        ),
        SemanticFact(
            paper_id="paper:lens",
            fact_id="fact:lens:claim:1",
            fact_type="Claim",
            text="A soft tunable bilayer DE lens achieves focal length variation at 500 V.",
            evidence="Soft tunable bilayer lens operated with voltage.",
            confidence=0.8,
            targets=["metric:voltage"],
        ),
    ]
    write_index(graph, index_dir, facts=facts)

    charge_results = query_facts(
        index_dir,
        target="concept:Charge_Trapping",
        text="trap depth changes with crystalline structure",
    )
    lens_results = query_facts(
        index_dir,
        target="metric:voltage",
        text="soft adjustable lens driven with a small electrical input",
    )

    assert [result["paper_id"] for result in charge_results] == ["paper:charge"]
    assert [result["paper_id"] for result in lens_results] == ["paper:lens"]


def test_query_facts_matches_expanded_research_paraphrases(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:fluid", "kind": "Paper", "label": "Fluid DEA"},
            {"id": "paper:ipmc", "kind": "Paper", "label": "IPMC"},
            {"id": "paper:strain", "kind": "Paper", "label": "EAP"},
            {
                "id": "concept:Dielectric_Elastomer_Actuator",
                "kind": "Concept",
                "label": "DEA",
            },
            {"id": "metric:voltage", "kind": "Metric", "label": "Voltage"},
            {"id": "metric:strain", "kind": "Metric", "label": "Strain"},
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:fluid",
            fact_id="fact:fluid:claim:1",
            fact_type="Claim",
            text="Translucent soft robots use a fluid electrode DEA bimorph.",
            evidence="fluid electrode dielectric elastomer actuator",
            confidence=0.8,
            targets=["concept:Dielectric_Elastomer_Actuator"],
        ),
        SemanticFact(
            paper_id="paper:ipmc",
            fact_id="fact:ipmc:claim:1",
            fact_type="Claim",
            text="IPMCs are ionic polymer actuators with actuation voltage of 1-5 V.",
            evidence="low actuation voltage 1-5 V",
            confidence=0.8,
            targets=["metric:voltage"],
        ),
        SemanticFact(
            paper_id="paper:strain",
            fact_id="fact:strain:claim:1",
            fact_type="Claim",
            text="Electroactive polymer materials have exceeded 100 percent strain.",
            evidence="strain levels exceeded 100 percent",
            confidence=0.8,
            targets=["metric:strain"],
        ),
    ]
    write_index(graph, index_dir, facts=facts)

    fluid_results = query_facts(
        index_dir,
        target="concept:Dielectric_Elastomer_Actuator",
        text="transparent soft robot using fluid electrodes and dielectric elastomer bending",
        match_mode="semantic",
    )
    ipmc_results = query_facts(
        index_dir,
        target="metric:voltage",
        text="ionic polymer actuator review reporting one to five volt operation",
        match_mode="semantic",
    )
    strain_results = query_facts(
        index_dir,
        target="metric:strain",
        text="electroactive polymer actuator class capable of more than one hundred percent strain",
        match_mode="semantic",
    )

    assert [result["paper_id"] for result in fluid_results] == ["paper:fluid"]
    assert [result["paper_id"] for result in ipmc_results] == ["paper:ipmc"]
    assert [result["paper_id"] for result in strain_results] == ["paper:strain"]


def test_query_facts_semantic_mode_ranks_partial_paraphrase_matches(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:expected", "kind": "Paper", "label": "Expected Paper"},
            {"id": "paper:other", "kind": "Paper", "label": "Other Paper"},
            {
                "id": "mechanism:electrothermal_actuation",
                "kind": "Mechanism",
                "label": "Electrothermal",
            },
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:other",
            fact_id="fact:other:claim:1",
            fact_type="Claim",
            text="A generic electrothermal actuator is described.",
            evidence="Generic electrothermal actuator.",
            confidence=0.6,
            targets=["mechanism:electrothermal_actuation"],
        ),
        SemanticFact(
            paper_id="paper:expected",
            fact_id="fact:expected:claim:1",
            fact_type="Claim",
            text=(
                "Twisting LCE/CNT composite films into a fiber increases uniaxial "
                "muscle stroke for electrothermal actuation under DC voltage."
            ),
            evidence="LCE CNT composite muscle stroke electrothermal DC voltage.",
            confidence=0.8,
            targets=["mechanism:electrothermal_actuation"],
        ),
    ]
    write_index(graph, index_dir, facts=facts)

    keyword_results = query_facts(
        index_dir,
        target="mechanism:electrothermal_actuation",
        text="carbon nanotube liquid crystal elastomer muscle heated by current network",
    )
    semantic_results = query_facts(
        index_dir,
        target="mechanism:electrothermal_actuation",
        text="carbon nanotube liquid crystal elastomer muscle heated by current network",
        match_mode="semantic",
    )

    assert keyword_results == []
    assert [result["paper_id"] for result in semantic_results] == [
        "paper:expected",
        "paper:other",
    ]
    assert semantic_results[0]["score"] > semantic_results[1]["score"]


def test_query_facts_limit_and_missing_index(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [{"id": "paper:center", "kind": "Paper", "label": "Center Paper"}],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:center",
            fact_id=f"fact:center:claim:{idx}",
            fact_type="Claim",
            text=f"Claim {idx}",
            evidence="Evidence",
            confidence=0.8,
        )
        for idx in range(2)
    ]
    write_index(graph, index_dir, facts=facts)

    assert len(query_facts(index_dir, limit=1)) == 1
    assert query_facts(index_dir, limit=0) == []
    try:
        query_facts(tmp_path / "missing")
    except FileNotFoundError as exc:
        assert "Graph index not found" in str(exc)
    else:
        raise AssertionError("Expected missing graph index to raise FileNotFoundError")


def test_fact_results_to_context_bundle_returns_citable_items():
    bundle = fact_results_to_context_bundle(
        "low voltage",
        "hybrid",
        [
            {
                "paper_id": "paper:center",
                "paper_title": "Center Paper",
                "paper_path": "research/center.md",
                "pdf_path": "/tmp/center.pdf",
                "fact_id": "fact:center:1",
                "fact_type": "Claim",
                "text": "Low voltage actuation was reported.",
                "evidence": "Reported at low voltage.",
                "targets": ["metric:voltage"],
                "score": 0.91,
                "debug_only": "not copied",
            }
        ],
    )

    assert bundle == {
        "query": "low voltage",
        "match_mode": "hybrid",
        "context_items": [
            {
                "citation_id": "F1",
                "paper_id": "paper:center",
                "canonical_paper_id": "",
                "citation_key": "",
                "duplicate_of": "",
                "paper_title": "Center Paper",
                "paper_path": "research/center.md",
                "pdf_name": "center.pdf",
                "fact_id": "fact:center:1",
                "fact_type": "Claim",
                "text": "Low voltage actuation was reported.",
                "evidence": "Reported at low voltage.",
                "targets": ["metric:voltage"],
                "score": 0.91,
            }
        ],
    }


def test_main_dispatches_fact_query(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [{"id": "paper:center", "kind": "Paper", "label": "Center Paper"}],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:center",
            fact_id="fact:center:claim:1",
            fact_type="Claim",
            text="Low voltage actuation was reported.",
            evidence="Reported at low voltage.",
            confidence=0.76,
            targets=["metric:voltage"],
        )
    ]
    write_index(graph, index_dir, facts=facts)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_query.py",
            "--facts",
            "--paper",
            "paper:center",
            "--graph-index",
            str(index_dir),
            "--text",
            "voltage",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "semantic fact(s)" in output
    assert "Low voltage actuation" in output
    assert "paper: Center Paper" in output
    assert "metric:voltage" in output


def test_main_prints_json_fact_query_with_provenance(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:center",
                "kind": "Paper",
                "label": "Center Paper",
                "rel_path": "research/center.md",
            }
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:center",
                fact_id="fact:center:claim:1",
                fact_type="Claim",
                text="Low voltage actuation was reported.",
                evidence="Reported at low voltage.",
                confidence=0.76,
                targets=["metric:voltage"],
            )
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_query.py",
            "voltage",
            "--facts",
            "--graph-index",
            str(index_dir),
            "--json",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["match_mode"] == "keyword"
    assert payload["query"] == "voltage"
    assert payload["results"][0]["paper_id"] == "paper:center"
    assert payload["results"][0]["paper_title"] == "Center Paper"
    assert payload["results"][0]["paper_path"] == "research/center.md"


def test_main_prints_context_fact_query_bundle(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:center",
                "kind": "Paper",
                "label": "Center Paper",
                "rel_path": "research/center.md",
            }
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:center",
                fact_id="fact:center:claim:1",
                fact_type="Claim",
                text="Low voltage actuation was reported.",
                evidence="Reported at low voltage.",
                confidence=0.76,
                targets=["metric:voltage"],
            )
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_query.py",
            "voltage",
            "--facts",
            "--graph-index",
            str(index_dir),
            "--context",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["match_mode"] == "keyword"
    assert payload["query"] == "voltage"
    assert payload["context_items"][0]["citation_id"] == "F1"
    assert payload["context_items"][0]["paper_id"] == "paper:center"
    assert payload["context_items"][0]["paper_title"] == "Center Paper"
    assert payload["context_items"][0]["text"] == "Low voltage actuation was reported."


def test_main_auto_fact_query_uses_hybrid_when_embeddings_exist(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:actuator", "kind": "Paper", "label": "Actuator"},
            {"id": "paper:far", "kind": "Paper", "label": "Far"},
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:actuator",
                fact_id="fact:actuator:1",
                fact_type="Mechanism",
                text="Liquid droplets expand during electrothermal actuation.",
                evidence="liquid droplets",
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            ),
            SemanticFact(
                paper_id="paper:far",
                fact_id="fact:far:1",
                fact_type="Claim",
                text="Charge trap energy changes with morphology.",
                evidence="charge trap",
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            ),
        ],
    )
    provider = FakeEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)
    # Building embeddings and refreshing the staleness manifest are now two steps
    # (refresh-graph step 7); wiki_query's embedding_index_is_current reads the latter.
    write_manifest(index_dir)
    monkeypatch.setattr("legacy.graphrag.wiki_query._provider_from_name", lambda *_args: provider)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_query.py",
            "heated liquid droplet actuator",
            "--facts",
            "--target",
            "mechanism:electrothermal_actuation",
            "--graph-index",
            str(index_dir),
            "--limit",
            "5",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "semantic fact(s)" in output
    assert "paper:actuator" in output
    assert "paper: Actuator" in output
    assert "Liquid droplets expand" in output
    assert "paper:far" not in output


def test_main_auto_fact_query_falls_back_to_keyword_without_embeddings(
    monkeypatch, capsys, tmp_path
):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [{"id": "paper:center", "kind": "Paper", "label": "Center Paper"}],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:center",
                fact_id="fact:center:claim:1",
                fact_type="Claim",
                text="Low voltage actuation was reported.",
                evidence="Reported at low voltage.",
                confidence=0.76,
                targets=["metric:voltage"],
            )
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_query.py",
            "voltage",
            "--facts",
            "--graph-index",
            str(index_dir),
            "--limit",
            "5",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "semantic fact(s)" in output
    assert "Low voltage actuation" in output
    assert "paper:center" in output


def test_main_facts_rejects_unsupported_legacy_filters(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["wiki_query.py", "--facts", "--domain", "research"])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    output = capsys.readouterr().out
    assert "Semantic fact queries do not support filters: --domain" in output


def test_main_preserves_legacy_query_without_near(monkeypatch, capsys, tmp_path):
    note = tmp_path / "wiki" / "research" / "legacy.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        """---
domain: research
tags: [test]
year: 2024
---
# Legacy Query Note

## Abstract
Existing query behavior should still work.
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["wiki_query.py", "--domain", "research"])

    main()

    output = capsys.readouterr().out
    assert "Legacy Query Note" in output
    assert "1 result(s)" in output
