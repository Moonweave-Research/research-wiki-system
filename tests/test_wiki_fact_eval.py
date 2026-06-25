import json
import sys

import pytest

from eval.wiki_fact_eval import (
    BenchmarkCase,
    evaluate_cases,
    load_benchmark_cases,
    main,
    quality_gate_failures,
)
from scripts.search.wiki_fact_embeddings import build_fact_embedding_index
from legacy.graphrag.wiki_graph import write_index
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


def test_load_benchmark_cases_reads_json_list(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "voltage-low",
                    "query": "low voltage",
                    "target": "metric:voltage",
                    "expected_papers": ["paper:cao2019"],
                    "forbidden_papers": ["paper:false"],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_benchmark_cases(benchmark) == [
        BenchmarkCase(
            case_id="voltage-low",
            query="low voltage",
            target="metric:voltage",
            expected_papers=["paper:cao2019"],
            forbidden_papers=["paper:false"],
        )
    ]


def test_evaluate_cases_reports_recall_missing_and_unexpected(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:cao2019", "kind": "Paper", "label": "Cao 2019"},
            {"id": "paper:lee2021", "kind": "Paper", "label": "Lee 2021"},
            {"id": "paper:false", "kind": "Paper", "label": "False Hit"},
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:cao2019",
            fact_id="fact:cao2019:claim:1",
            fact_type="Claim",
            text="Low voltage actuation was demonstrated.",
            evidence="low voltage actuation",
            confidence=0.8,
            targets=["metric:voltage"],
        ),
        SemanticFact(
            paper_id="paper:false",
            fact_id="fact:false:claim:1",
            fact_type="Claim",
            text="Low voltage wording appears but is not expected.",
            evidence="low voltage wording",
            confidence=0.5,
            targets=["metric:voltage"],
        ),
    ]
    write_index(graph, index_dir, facts=facts)
    cases = [
        BenchmarkCase(
            case_id="voltage",
            query="low voltage",
            target="metric:voltage",
            expected_papers=["paper:cao2019", "paper:lee2021"],
            forbidden_papers=["paper:false"],
        )
    ]

    report = evaluate_cases(index_dir, cases, limit=5)

    assert report["summary"] == {
        "cases": 1,
        "expected_papers": 2,
        "retrieved_papers": 2,
        "found_expected_papers": 1,
        "unexpected_papers": 1,
        "forbidden_papers": 1,
        "recall_at_k": 0.5,
        "precision_at_k": 0.5,
    }
    assert report["cases"][0]["found_expected_papers"] == ["paper:cao2019"]
    assert report["cases"][0]["missing_expected_papers"] == ["paper:lee2021"]
    assert report["cases"][0]["unexpected_papers"] == ["paper:false"]
    assert report["cases"][0]["forbidden_papers"] == ["paper:false"]
    assert report["cases"][0]["found_forbidden_papers"] == ["paper:false"]


def test_quality_gate_failures_reports_threshold_misses():
    report = {
        "summary": {
            "recall_at_k": 0.9,
            "precision_at_k": 0.8,
            "unexpected_papers": 2,
            "forbidden_papers": 1,
        }
    }

    failures = quality_gate_failures(
        report,
        min_recall=1.0,
        min_precision=0.9,
        max_unexpected=0,
        max_forbidden=0,
    )

    assert failures == [
        "recall_at_k 0.900 is below 1.000",
        "precision_at_k 0.800 is below 0.900",
        "unexpected_papers 2 exceeds 0",
        "forbidden_papers 1 exceeds 0",
    ]


def test_evaluate_cases_applies_limit_to_distinct_papers_not_fact_rows(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:a", "kind": "Paper", "label": "A"},
            {"id": "paper:b", "kind": "Paper", "label": "B"},
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:a",
            fact_id=f"fact:a:claim:{idx}",
            fact_type="Claim",
            text=f"Low voltage duplicate fact {idx}.",
            evidence="low voltage",
            confidence=0.8,
            targets=["metric:voltage"],
        )
        for idx in range(2)
    ] + [
        SemanticFact(
            paper_id="paper:b",
            fact_id="fact:b:claim:1",
            fact_type="Claim",
            text="Low voltage second paper.",
            evidence="low voltage",
            confidence=0.8,
            targets=["metric:voltage"],
        )
    ]
    write_index(graph, index_dir, facts=facts)

    report = evaluate_cases(
        index_dir,
        [
            BenchmarkCase(
                case_id="voltage",
                query="low voltage",
                target="metric:voltage",
                expected_papers=["paper:a", "paper:b"],
            )
        ],
        limit=2,
    )

    assert report["summary"]["recall_at_k"] == 1.0
    assert report["cases"][0]["found_expected_papers"] == ["paper:a", "paper:b"]


def test_evaluate_cases_counts_canonical_duplicate_papers_once(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:canonical",
                "kind": "Paper",
                "label": "Canonical",
                "canonical_paper_id": "paper:canonical",
            },
            {
                "id": "paper:duplicate",
                "kind": "Paper",
                "label": "Duplicate",
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
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:canonical",
                fact_id="fact:canonical:claim:1",
                fact_type="Claim",
                text="Low voltage actuator evidence.",
                evidence="low voltage",
                confidence=0.8,
                targets=["metric:voltage"],
            ),
            SemanticFact(
                paper_id="paper:duplicate",
                fact_id="fact:duplicate:claim:1",
                fact_type="Claim",
                text="Low voltage actuator evidence.",
                evidence="low voltage",
                confidence=0.8,
                targets=["metric:voltage"],
            ),
        ],
    )

    report = evaluate_cases(
        index_dir,
        [
            BenchmarkCase(
                case_id="voltage",
                query="low voltage",
                target="metric:voltage",
                expected_papers=["paper:canonical"],
            )
        ],
        limit=5,
    )

    assert report["summary"]["retrieved_papers"] == 1
    assert report["summary"]["precision_at_k"] == 1.0
    assert report["cases"][0]["found_papers"] == ["paper:canonical"]


def test_evaluate_cases_can_use_semantic_match_mode(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:expected", "kind": "Paper", "label": "Expected Paper"},
            {"id": "mechanism:electrothermal_actuation", "kind": "Mechanism", "label": "Electrothermal"},
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:expected",
                fact_id="fact:expected:claim:1",
                fact_type="Claim",
                text="LCE/CNT fibers show muscle stroke under electrothermal actuation.",
                evidence="LCE CNT muscle stroke electrothermal actuation.",
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            )
        ],
    )

    keyword_report = evaluate_cases(
        index_dir,
        [
            BenchmarkCase(
                case_id="cnt-muscle",
                query="carbon nanotube liquid crystal elastomer muscle heated current network",
                target="mechanism:electrothermal_actuation",
                expected_papers=["paper:expected"],
            )
        ],
        limit=5,
    )
    semantic_report = evaluate_cases(
        index_dir,
        [
            BenchmarkCase(
                case_id="cnt-muscle",
                query="carbon nanotube liquid crystal elastomer muscle heated current network",
                target="mechanism:electrothermal_actuation",
                expected_papers=["paper:expected"],
            )
        ],
        limit=5,
        match_mode="semantic",
    )

    assert keyword_report["summary"]["recall_at_k"] == 0.0
    assert semantic_report["summary"]["recall_at_k"] == 1.0


def test_evaluate_cases_can_use_embedding_match_mode(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:actuator", "kind": "Paper", "label": "Actuator"},
            {"id": "paper:charge", "kind": "Paper", "label": "Charge"},
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:charge",
                fact_id="fact:charge:claim:1",
                fact_type="Claim",
                text="Charge trap energy changes with morphology.",
                evidence="charge trap",
                confidence=0.8,
                targets=["concept:Charge_Trapping"],
            ),
            SemanticFact(
                paper_id="paper:actuator",
                fact_id="fact:actuator:mechanism:1",
                fact_type="Mechanism",
                text="Liquid droplets expand during electrothermal actuation.",
                evidence="liquid droplets",
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            ),
        ],
    )
    provider = FakeEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    report = evaluate_cases(
        index_dir,
        [
            BenchmarkCase(
                case_id="droplet-actuator",
                query="heated droplet actuator",
                target="mechanism:electrothermal_actuation",
                expected_papers=["paper:actuator"],
            )
        ],
        limit=5,
        match_mode="embedding",
        embedding_provider=provider,
    )

    assert report["summary"]["recall_at_k"] == 1.0
    assert report["summary"]["precision_at_k"] == 1.0
    assert report["cases"][0]["found_expected_papers"] == ["paper:actuator"]


def test_evaluate_cases_embedding_match_mode_requires_provider(tmp_path):
    with pytest.raises(ValueError, match="embedding provider"):
        evaluate_cases(tmp_path / ".graph_index", [], match_mode="embedding")


def test_evaluate_cases_can_use_hybrid_match_mode(tmp_path):
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
                fact_id="fact:actuator:mechanism:1",
                fact_type="Mechanism",
                text="Liquid droplets expand during electrothermal actuation.",
                evidence="liquid droplets",
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            ),
            SemanticFact(
                paper_id="paper:far",
                fact_id="fact:far:claim:1",
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

    report = evaluate_cases(
        index_dir,
        [
            BenchmarkCase(
                case_id="droplet-actuator",
                query="heated liquid droplet actuator",
                target="mechanism:electrothermal_actuation",
                expected_papers=["paper:actuator"],
            )
        ],
        limit=5,
        match_mode="hybrid",
        embedding_provider=provider,
        candidate_limit=5,
        score_margin=0.04,
    )

    assert report["summary"]["recall_at_k"] == 1.0
    assert report["summary"]["precision_at_k"] == 1.0


def test_evaluate_cases_rejects_non_positive_limit(tmp_path):
    with pytest.raises(ValueError, match="limit must be positive"):
        evaluate_cases(tmp_path / ".graph_index", [], limit=0)


def test_load_benchmark_cases_requires_expected_papers(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps([{"id": "missing-expected", "query": "low voltage"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing expected_papers"):
        load_benchmark_cases(benchmark)


def test_load_benchmark_cases_rejects_non_string_optional_filters(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "bad-target",
                    "query": "low voltage",
                    "target": ["metric:voltage"],
                    "expected_papers": ["paper:cao2019"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target"):
        load_benchmark_cases(benchmark)


def test_load_benchmark_cases_rejects_invalid_forbidden_papers(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "bad-forbidden",
                    "query": "low voltage",
                    "expected_papers": ["paper:cao2019"],
                    "forbidden_papers": ["paper:false", 123],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden_papers"):
        load_benchmark_cases(benchmark)


def test_main_prints_human_readable_report(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {"nodes": [{"id": "paper:cao2019", "kind": "Paper", "label": "Cao 2019"}], "edges": []}
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:cao2019",
                fact_id="fact:cao2019:claim:1",
                fact_type="Claim",
                text="Low voltage actuation was demonstrated.",
                evidence="low voltage actuation",
                confidence=0.8,
                targets=["metric:voltage"],
            )
        ],
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "voltage",
                    "query": "low voltage",
                    "target": "metric:voltage",
                    "expected_papers": ["paper:cao2019"],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_fact_eval.py",
            "--graph-index",
            str(index_dir),
            "--benchmark",
            str(benchmark),
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "Recall@5: 1.000" in output
    assert "Precision@5: 1.000" in output
    assert "0 unexpected paper hit(s)" in output
    assert "0 forbidden paper hit(s)" in output
    assert "voltage" in output
    assert "paper:cao2019" in output


def test_main_prints_json_report(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {"nodes": [{"id": "paper:cao2019", "kind": "Paper", "label": "Cao 2019"}], "edges": []}
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:cao2019",
                fact_id="fact:cao2019:claim:1",
                fact_type="Claim",
                text="Low voltage actuation was demonstrated.",
                evidence="low voltage actuation",
                confidence=0.8,
                targets=["metric:voltage"],
            )
        ],
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "voltage",
                    "query": "low voltage",
                    "target": "metric:voltage",
                    "expected_papers": ["paper:cao2019"],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_fact_eval.py",
            "--graph-index",
            str(index_dir),
            "--benchmark",
            str(benchmark),
            "--json",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["recall_at_k"] == 1.0
    assert payload["summary"]["precision_at_k"] == 1.0
    assert payload["summary"]["retrieved_papers"] == 1
    assert payload["summary"]["unexpected_papers"] == 0
    assert payload["summary"]["forbidden_papers"] == 0
    assert payload["cases"][0]["id"] == "voltage"


def test_main_fails_quality_gate(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:cao2019", "kind": "Paper", "label": "Cao 2019"},
            {"id": "paper:false", "kind": "Paper", "label": "False Hit"},
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:cao2019",
                fact_id="fact:cao2019:claim:1",
                fact_type="Claim",
                text="Low voltage actuation was demonstrated.",
                evidence="low voltage actuation",
                confidence=0.8,
                targets=["metric:voltage"],
            ),
            SemanticFact(
                paper_id="paper:false",
                fact_id="fact:false:claim:1",
                fact_type="Claim",
                text="Low voltage false positive.",
                evidence="low voltage",
                confidence=0.8,
                targets=["metric:voltage"],
            ),
        ],
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "voltage",
                    "query": "low voltage",
                    "target": "metric:voltage",
                    "expected_papers": ["paper:cao2019"],
                    "forbidden_papers": ["paper:false"],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_fact_eval.py",
            "--graph-index",
            str(index_dir),
            "--benchmark",
            str(benchmark),
            "--max-forbidden",
            "0",
            "--max-unexpected",
            "0",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    output = capsys.readouterr().out
    assert "Quality gate failed" in output
    assert "forbidden_papers 1 exceeds 0" in output
    assert "unexpected_papers 1 exceeds 0" in output


def test_main_rejects_non_positive_limit(monkeypatch, capsys, tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps([{"id": "voltage", "expected_papers": ["paper:cao2019"]}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_fact_eval.py",
            "--graph-index",
            str(tmp_path / ".graph_index"),
            "--benchmark",
            str(benchmark),
            "--limit",
            "0",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    assert "limit must be positive" in capsys.readouterr().out


def test_main_rejects_invalid_benchmark_without_traceback(monkeypatch, capsys, tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps([{"id": "bad"}]), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_fact_eval.py",
            "--graph-index",
            str(tmp_path / ".graph_index"),
            "--benchmark",
            str(benchmark),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    output = capsys.readouterr().out
    assert "missing expected_papers" in output
    assert "Traceback" not in output
