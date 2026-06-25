import json
import sys

from legacy.graphrag.wiki_answer import answer_from_context_bundle, main
from legacy.graphrag.wiki_graph import write_index
from legacy.graphrag.wiki_semantic_schema import SemanticFact


def test_answer_from_context_bundle_renders_cited_local_answer():
    bundle = {
        "query": "low voltage actuator",
        "match_mode": "hybrid",
        "context_items": [
            {
                "citation_id": "F1",
                "paper_id": "paper:center",
                "paper_title": "Center Paper",
                "paper_path": "research/center.md",
                "fact_id": "fact:center:1",
                "fact_type": "Claim",
                "text": "Low voltage actuation was reported.",
                "evidence": "Reported at low voltage.",
                "targets": ["metric:voltage"],
                "score": 0.91,
            }
        ],
    }

    answer = answer_from_context_bundle(bundle)

    assert "low voltage actuator" in answer
    assert "Low voltage actuation was reported. [F1]" in answer
    assert "[F1] Center Paper" in answer
    assert "research/center.md" in answer


def test_answer_from_context_bundle_reports_no_evidence():
    answer = answer_from_context_bundle(
        {"query": "unknown", "match_mode": "hybrid", "context_items": []}
    )

    assert "No local evidence found" in answer
    assert "unknown" in answer


def test_answer_from_context_bundle_deduplicates_identical_fact_text():
    bundle = {
        "query": "low voltage actuator",
        "match_mode": "hybrid",
        "context_items": [
            {
                "citation_id": "F1",
                "paper_title": "Center Paper",
                "paper_path": "research/center.md",
                "fact_id": "fact:center:1",
                "text": "Low voltage actuation was reported.",
            },
            {
                "citation_id": "F2",
                "paper_title": "Center Paper Duplicate",
                "paper_path": "research/center-duplicate.md",
                "fact_id": "fact:center:2",
                "text": "Low voltage actuation was reported.",
            },
        ],
    }

    answer = answer_from_context_bundle(bundle)

    assert answer.count("Low voltage actuation was reported.") == 1
    assert "[F1] Center Paper" in answer
    assert "[F2] Center Paper Duplicate" not in answer


def test_answer_from_context_bundle_deduplicates_by_canonical_paper():
    bundle = {
        "query": "low voltage actuator",
        "match_mode": "hybrid",
        "context_items": [
            {
                "citation_id": "F1",
                "paper_id": "paper:canonical",
                "canonical_paper_id": "paper:canonical",
                "paper_title": "Center Paper",
                "paper_path": "research/center.md",
                "fact_id": "fact:center:1",
                "text": "Low voltage actuation was reported.",
            },
            {
                "citation_id": "F2",
                "paper_id": "paper:duplicate",
                "canonical_paper_id": "paper:canonical",
                "duplicate_of": "paper:canonical",
                "paper_title": "Center Paper Duplicate",
                "paper_path": "research/center-duplicate.md",
                "fact_id": "fact:duplicate:2",
                "text": "Force capacity was also reported in the duplicate note.",
            },
        ],
    }

    answer = answer_from_context_bundle(bundle)

    assert "Low voltage actuation was reported. [F1]" in answer
    assert "Force capacity was also reported" not in answer
    assert "[F1] Center Paper" in answer
    assert "[F2] Center Paper Duplicate" not in answer


def test_main_prints_json_answer_from_local_facts(monkeypatch, capsys, tmp_path):
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
                text="Low voltage actuator actuation was reported.",
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
            "wiki_answer.py",
            "low voltage actuator",
            "--graph-index",
            str(index_dir),
            "--target",
            "metric:voltage",
            "--match-mode",
            "keyword",
            "--json",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "low voltage actuator"
    assert payload["match_mode"] == "keyword"
    assert payload["context_items"][0]["citation_id"] == "F1"
    assert "Low voltage actuator actuation was reported. [F1]" in payload["answer"]
