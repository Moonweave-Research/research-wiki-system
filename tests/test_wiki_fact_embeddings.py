import json
import builtins
import time
import sqlite3
from dataclasses import dataclass, asdict

import numpy as np
import pytest

from scripts.search import wiki_fact_embeddings
from scripts.search.wiki_fact_embeddings import (
    EMBEDDING_CACHE_DB_FILE,
    EMBEDDING_META_FILE,
    build_fact_embedding_index,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_SCORE_MARGIN,
    exact_token_overlap_score,
    FastEmbedProvider,
    query_fact_embeddings,
    query_hybrid_fact_embeddings,
    query_specificity_bonus,
    read_fact_records,
)
@dataclass
class SemanticFact:
    paper_id: str
    fact_id: str
    fact_type: str
    text: str
    evidence: str
    confidence: float
    targets: list[str]


def write_index(graph, index_dir, *, facts):
    """Create synthetic JSONL facts and SQLite compatibility metadata locally."""
    facts = list(facts)
    _write_facts_jsonl(index_dir, facts)
    with sqlite3.connect(index_dir / "graph.sqlite") as connection:
        connection.executescript(
            "CREATE TABLE IF NOT EXISTS nodes (id TEXT, kind TEXT, label TEXT, props_json TEXT);"
            "CREATE TABLE IF NOT EXISTS edges (source TEXT, target TEXT, type TEXT);"
            "DELETE FROM nodes; DELETE FROM edges;"
        )
        nodes = [(n["id"], n["kind"], n.get("label", ""), json.dumps(n)) for n in graph["nodes"]]
        nodes.extend((f.fact_id, "Fact", f.text, json.dumps(asdict(f))) for f in facts)
        connection.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?)", nodes)
        edges = [(e["source"], e["target"], e["type"]) for e in graph["edges"]]
        edges.extend((f.fact_id, target, "ABOUT") for f in facts for target in f.targets)
        connection.executemany("INSERT INTO edges VALUES (?, ?, ?)", edges)


class FakeEmbeddingProvider:
    model_name = "fake-embeddings"

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lower = text.lower()
        if "microliquid" in lower or "droplet" in lower:
            return [1.0, 0.0]
        if "charge" in lower or "trap" in lower:
            return [0.0, 1.0]
        return [0.5, 0.5]


class RecordingEmbeddingProvider:
    model_name = "recording-embeddings"

    def __init__(self):
        self.document_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[float(len(self.document_calls)), 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FailingSecondBatchEmbeddingProvider(RecordingEmbeddingProvider):
    def embed_documents(self, texts):
        if self.document_calls:
            raise RuntimeError("simulated embedding interruption")
        return super().embed_documents(texts)


def _write_facts_jsonl(index_dir, facts, filename="facts.jsonl"):
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        for fact in facts:
            handle.write(
                json.dumps(
                    {
                        "fact_id": fact.fact_id,
                        "paper_id": fact.paper_id,
                        "fact_type": fact.fact_type,
                        "text": fact.text,
                        "evidence": fact.evidence,
                        "confidence": fact.confidence,
                        "targets": list(fact.targets),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def test_default_score_margin_is_precision_oriented():
    assert DEFAULT_SCORE_MARGIN == 0.02


def test_default_keyword_weight_prefers_fact_evidence():
    assert DEFAULT_KEYWORD_WEIGHT == 0.3


def test_fastembed_provider_defers_model_loading_until_embedding(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "fastembed":
            raise AssertionError("fastembed should not load during provider construction")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    provider = FastEmbedProvider("fake-model")

    assert provider.model_name == "fake-model"


def test_build_fact_embedding_index_writes_vectors_and_manifest(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:actuator", "kind": "Paper", "label": "Actuator"},
            {"id": "paper:charge", "kind": "Paper", "label": "Charge"},
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id="paper:actuator",
            fact_id="fact:actuator:1",
            fact_type="Mechanism",
            text="Microliquid droplets expand under resistive heating.",
            evidence="microliquid droplets",
            confidence=0.8,
            targets=["mechanism:electrothermal_actuation"],
        ),
        SemanticFact(
            paper_id="paper:charge",
            fact_id="fact:charge:1",
            fact_type="Claim",
            text="Charge trap energy level depends on crystallite structure.",
            evidence="charge trap",
            confidence=0.8,
            targets=["concept:Charge_Trapping"],
        ),
    ]
    write_index(graph, index_dir, facts=facts)

    report = build_fact_embedding_index(index_dir, FakeEmbeddingProvider())

    assert report == {
        "model": "fake-embeddings",
        "facts": 2,
        "dimensions": 2,
        "vectors": str(index_dir / "fact_embeddings.npz"),
        "manifest": str(index_dir / "fact_embeddings_manifest.jsonl"),
        "reused": 0,
        "embedded": 2,
        "skipped_current": False,
    }
    vectors = np.load(index_dir / "fact_embeddings.npz")["vectors"]
    assert vectors.shape == (2, 2)
    assert (index_dir / "fact_embeddings_manifest.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 2


def test_build_fact_embedding_index_batches_document_embedding_calls(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": f"paper:item{i}", "kind": "Paper", "label": f"Paper {i}"} for i in range(5)
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id=f"paper:item{i}",
            fact_id=f"fact:item{i}:1",
            fact_type="Claim",
            text=f"Fact text {i}",
            evidence=f"Evidence {i}",
            confidence=0.8,
            targets=[],
        )
        for i in range(5)
    ]
    write_index(graph, index_dir, facts=facts)
    provider = RecordingEmbeddingProvider()

    report = build_fact_embedding_index(index_dir, provider, batch_size=2)

    assert report["facts"] == 5
    assert [len(call) for call in provider.document_calls] == [2, 2, 1]
    vectors = np.load(index_dir / "fact_embeddings.npz")["vectors"]
    assert vectors.shape == (5, 2)


def test_build_fact_embedding_index_reuses_unchanged_existing_vectors(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": f"paper:item{i}", "kind": "Paper", "label": f"Paper {i}"} for i in range(3)
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id=f"paper:item{i}",
            fact_id=f"fact:item{i}:1",
            fact_type="Claim",
            text=f"Fact text {i}",
            evidence=f"Evidence {i}",
            confidence=0.8,
            targets=[],
        )
        for i in range(2)
    ]
    write_index(graph, index_dir, facts=facts)
    build_fact_embedding_index(index_dir, RecordingEmbeddingProvider(), batch_size=2)
    first_vectors = np.load(index_dir / "fact_embeddings.npz")["vectors"]

    write_index(
        graph,
        index_dir,
        facts=[
            *facts,
            SemanticFact(
                paper_id="paper:item2",
                fact_id="fact:item2:1",
                fact_type="Claim",
                text="Fact text 2",
                evidence="Evidence 2",
                confidence=0.8,
                targets=[],
            ),
        ],
    )
    provider = RecordingEmbeddingProvider()

    build_fact_embedding_index(index_dir, provider, batch_size=2)

    assert [len(call) for call in provider.document_calls] == [1]
    vectors = np.load(index_dir / "fact_embeddings.npz")["vectors"]
    assert vectors.shape == (3, 2)
    np.testing.assert_array_equal(vectors[:2], first_vectors)


def _facts_for(n):
    return [
        SemanticFact(
            paper_id=f"paper:item{i}",
            fact_id=f"fact:item{i}:1",
            fact_type="Claim",
            text=f"Fact text {i}",
            evidence=f"Evidence {i}",
            confidence=0.8,
            targets=[],
        )
        for i in range(n)
    ]


def test_build_records_fastembed_version_in_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(wiki_fact_embeddings, "_fastembed_version", lambda: "9.9.9")
    index_dir = tmp_path / ".graph_index"
    graph = {"nodes": [{"id": "paper:item0", "kind": "Paper", "label": "P0"}], "edges": []}
    write_index(graph, index_dir, facts=_facts_for(1))

    build_fact_embedding_index(index_dir, RecordingEmbeddingProvider(), batch_size=2)

    meta = json.loads((index_dir / EMBEDDING_META_FILE).read_text(encoding="utf-8"))
    assert meta["fastembed_version"] == "9.9.9"


def test_build_rebuilds_all_when_fastembed_version_changes(tmp_path, monkeypatch):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [{"id": f"paper:item{i}", "kind": "Paper", "label": f"P{i}"} for i in range(2)],
        "edges": [],
    }
    write_index(graph, index_dir, facts=_facts_for(2))

    monkeypatch.setattr(wiki_fact_embeddings, "_fastembed_version", lambda: "0.5.1")
    build_fact_embedding_index(index_dir, RecordingEmbeddingProvider(), batch_size=2)
    assert (index_dir / EMBEDDING_CACHE_DB_FILE).exists()

    # A fastembed upgrade changes the pooling (CLS -> mean); stale vectors must not
    # be reused. Same facts, new version -> every fact is re-embedded, none skipped.
    monkeypatch.setattr(wiki_fact_embeddings, "_fastembed_version", lambda: "0.8.0")
    provider = RecordingEmbeddingProvider()
    report = build_fact_embedding_index(index_dir, provider, batch_size=2)

    assert sum(len(call) for call in provider.document_calls) == 2
    assert report.get("skipped_current") is not True
    assert report["embedded"] == 2
    assert report["reused"] == 0


def test_build_reuses_within_same_fastembed_version(tmp_path, monkeypatch):
    monkeypatch.setattr(wiki_fact_embeddings, "_fastembed_version", lambda: "1.0.0")
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [{"id": f"paper:item{i}", "kind": "Paper", "label": f"P{i}"} for i in range(2)],
        "edges": [],
    }
    write_index(graph, index_dir, facts=_facts_for(1))
    build_fact_embedding_index(index_dir, RecordingEmbeddingProvider(), batch_size=2)

    write_index(graph, index_dir, facts=_facts_for(2))
    provider = RecordingEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider, batch_size=2)

    # Only the new fact is embedded; the version is unchanged so the first is reused.
    assert sum(len(call) for call in provider.document_calls) == 1


def test_build_fact_embedding_index_reports_reused_and_embedded_counts(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": f"paper:item{i}", "kind": "Paper", "label": f"Paper {i}"} for i in range(2)
        ],
        "edges": [],
    }
    first_fact = SemanticFact(
        paper_id="paper:item0",
        fact_id="fact:item0:1",
        fact_type="Claim",
        text="Fact text 0",
        evidence="Evidence 0",
        confidence=0.8,
        targets=[],
    )
    write_index(graph, index_dir, facts=[first_fact])
    build_fact_embedding_index(index_dir, RecordingEmbeddingProvider(), batch_size=2)
    write_index(
        graph,
        index_dir,
        facts=[
            first_fact,
            SemanticFact(
                paper_id="paper:item1",
                fact_id="fact:item1:1",
                fact_type="Claim",
                text="Fact text 1",
                evidence="Evidence 1",
                confidence=0.8,
                targets=[],
            ),
        ],
    )

    report = build_fact_embedding_index(index_dir, RecordingEmbeddingProvider(), batch_size=2)

    assert report["reused"] == 1
    assert report["embedded"] == 1


def test_build_fact_embedding_index_skips_current_outputs_without_rewriting(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [{"id": "paper:item0", "kind": "Paper", "label": "Paper 0"}],
        "edges": [],
    }
    fact = SemanticFact(
        paper_id="paper:item0",
        fact_id="fact:item0:1",
        fact_type="Claim",
        text="Fact text 0",
        evidence="Evidence 0",
        confidence=0.8,
        targets=[],
    )
    write_index(graph, index_dir, facts=[fact])
    build_fact_embedding_index(index_dir, RecordingEmbeddingProvider(), batch_size=2)
    vector_path = index_dir / "fact_embeddings.npz"
    meta_path = index_dir / EMBEDDING_META_FILE
    vector_mtime_ns = vector_path.stat().st_mtime_ns
    meta_text = meta_path.read_text(encoding="utf-8")
    time.sleep(0.01)
    provider = RecordingEmbeddingProvider()

    report = build_fact_embedding_index(index_dir, provider, batch_size=2)

    assert report["skipped_current"] is True
    assert provider.document_calls == []
    assert vector_path.stat().st_mtime_ns == vector_mtime_ns
    assert meta_path.read_text(encoding="utf-8") == meta_text


def test_build_fact_embedding_index_fast_path_skips_graph_scan_when_current(tmp_path, monkeypatch):
    index_dir = _minimal_index(tmp_path)
    build_fact_embedding_index(index_dir, _FixedDimProvider(model_name="fake-A", dim=8))

    from scripts.search import wiki_fact_embeddings as mod

    def fail_read_fact_records(index_dir):
        raise AssertionError("current embedding fast path should not scan graph facts")

    monkeypatch.setattr(mod, "read_fact_records", fail_read_fact_records)

    report = build_fact_embedding_index(index_dir, _FixedDimProvider(model_name="fake-A", dim=8))

    assert report["skipped_current"] is True
    assert report["facts"] == 1
    assert report["dimensions"] == 8


def test_read_fact_records_merges_wiki_and_pdf_facts(tmp_path):
    index_dir = tmp_path / ".graph_index"
    wiki_facts = [
        SemanticFact(
            paper_id="paper:a",
            fact_id="fact:a:1",
            fact_type="Claim",
            text="Wiki fact one.",
            evidence="ev1",
            confidence=0.72,
            targets=["concept:X", "metric:strain"],
        )
    ]
    pdf_facts = [
        SemanticFact(
            paper_id="paper:a",
            fact_id="fact:a:pdf:1",
            fact_type="Claim",
            text="Pdf fact one.",
            evidence="ev2",
            confidence=0.6,
            targets=["metric:pressure"],
        )
    ]
    _write_facts_jsonl(index_dir, wiki_facts, "facts.jsonl")
    _write_facts_jsonl(index_dir, pdf_facts, "pdf_facts.jsonl")

    records = read_fact_records(index_dir)

    assert [r["fact_id"] for r in records] == ["fact:a:1", "fact:a:pdf:1"]
    assert records[0]["targets"] == ["concept:X", "metric:strain"]
    assert records[1]["text"] == "Pdf fact one."
    assert records[1]["confidence"] == 0.6


def test_read_fact_records_tolerates_missing_pdf_facts(tmp_path):
    index_dir = tmp_path / ".graph_index"
    _write_facts_jsonl(
        index_dir,
        [
            SemanticFact(
                paper_id="paper:a",
                fact_id="fact:a:1",
                fact_type="Claim",
                text="Only wiki facts here.",
                evidence="ev",
                confidence=0.72,
                targets=[],
            )
        ],
        "facts.jsonl",
    )
    records = read_fact_records(index_dir)
    assert len(records) == 1
    assert records[0]["targets"] == []


def test_build_fact_embedding_index_resumes_from_partial_batch_cache(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": f"paper:item{i}", "kind": "Paper", "label": f"Paper {i}"} for i in range(3)
        ],
        "edges": [],
    }
    facts = [
        SemanticFact(
            paper_id=f"paper:item{i}",
            fact_id=f"fact:item{i}:1",
            fact_type="Claim",
            text=f"Fact text {i}",
            evidence=f"Evidence {i}",
            confidence=0.8,
            targets=[],
        )
        for i in range(3)
    ]
    write_index(graph, index_dir, facts=facts)

    with pytest.raises(RuntimeError, match="simulated embedding interruption"):
        build_fact_embedding_index(
            index_dir,
            FailingSecondBatchEmbeddingProvider(),
            batch_size=2,
        )

    provider = RecordingEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider, batch_size=2)

    assert [len(call) for call in provider.document_calls] == [1]
    vectors = np.load(index_dir / "fact_embeddings.npz")["vectors"]
    assert vectors.shape == (3, 2)


def test_query_fact_embeddings_ranks_by_local_vector_similarity(tmp_path):
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
                fact_id="fact:charge:1",
                fact_type="Claim",
                text="Charge trap energy level depends on crystallite structure.",
                evidence="charge trap",
                confidence=0.8,
                targets=["concept:Charge_Trapping"],
            ),
            SemanticFact(
                paper_id="paper:actuator",
                fact_id="fact:actuator:1",
                fact_type="Mechanism",
                text="Microliquid droplets expand under resistive heating.",
                evidence="microliquid droplets",
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            ),
        ],
    )
    provider = FakeEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    results = query_fact_embeddings(
        index_dir,
        "heated liquid droplets in elastomer",
        provider,
        target="mechanism:electrothermal_actuation",
        limit=1,
    )

    assert len(results) == 1
    assert results[0]["paper_id"] == "paper:actuator"
    assert results[0]["fact_id"] == "fact:actuator:1"
    assert results[0]["score"] == 1.0


def test_query_fact_embeddings_deduplicates_canonical_paper_group(tmp_path):
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
                fact_id="fact:canonical:1",
                fact_type="Mechanism",
                text="Microliquid droplets expand under resistive heating.",
                evidence="microliquid droplets",
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            ),
            SemanticFact(
                paper_id="paper:duplicate",
                fact_id="fact:duplicate:1",
                fact_type="Mechanism",
                text="Liquid droplet heating creates a duplicate mechanism result.",
                evidence="liquid droplet heating",
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            ),
        ],
    )
    provider = FakeEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    results = query_fact_embeddings(
        index_dir,
        "heated liquid droplets in elastomer",
        provider,
        target="mechanism:electrothermal_actuation",
        limit=5,
    )

    assert [result["canonical_paper_id"] for result in results] == ["paper:canonical"]
    assert len(results) == 1


def test_exact_token_overlap_score_counts_query_specific_terms():
    assert exact_token_overlap_score(
        "organic semiconductor crystallinity controls charge trap relationships",
        "Crystallinity in organic semiconductors affects charge-trapping relationships.",
    ) > exact_token_overlap_score(
        "organic semiconductor crystallinity controls charge trap relationships",
        "Intrinsic charge trapping in organic semiconductors.",
    )


def test_query_specificity_bonus_matches_decisive_scientific_clues():
    assert (
        query_specificity_bonus(
            "actuator design that boosts force while greatly reducing drive voltage",
            "A Low-Voltage, High-Force Capacity Electroadhesive Clutch",
            "",
            "",
        )
        > 0
    )
    assert (
        query_specificity_bonus(
            "electroactive polymer actuator class capable of more than one hundred percent strain",
            "",
            "The generated strain levels have exceeded 100%.",
            "",
        )
        > 0
    )
    assert (
        query_specificity_bonus(
            "aligned carbon nanotube sheet actuator changes shape when electrically heated",
            "",
            "Designable electrothermal actuators based on carbon nanotube sheets.",
            "",
        )
        > 0
    )


def test_query_hybrid_fact_embeddings_deduplicates_and_applies_score_margin(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:actuator", "kind": "Paper", "label": "Actuator"},
            {"id": "paper:near", "kind": "Paper", "label": "Near"},
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
                paper_id="paper:actuator",
                fact_id="fact:actuator:2",
                fact_type="Mechanism",
                text="A duplicate liquid droplet mechanism row.",
                evidence="droplet mechanism",
                confidence=0.7,
                targets=["mechanism:electrothermal_actuation"],
            ),
            SemanticFact(
                paper_id="paper:near",
                fact_id="fact:near:1",
                fact_type="Mechanism",
                text="Microliquid heating creates a nearby actuator result.",
                evidence="microliquid heating",
                confidence=0.7,
                targets=["mechanism:electrothermal_actuation"],
            ),
            SemanticFact(
                paper_id="paper:far",
                fact_id="fact:far:1",
                fact_type="Mechanism",
                text="Charge trap energy level depends on crystallite structure.",
                evidence="charge trap",
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            ),
        ],
    )
    provider = FakeEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    results = query_hybrid_fact_embeddings(
        index_dir,
        "heated liquid droplet actuator",
        provider,
        target="mechanism:electrothermal_actuation",
        limit=5,
        score_margin=0.25,
    )

    assert [result["paper_id"] for result in results] == ["paper:actuator", "paper:near"]
    assert results[0]["score"] >= results[0]["embedding_score"]
    assert results[0]["lexical_score"] > 0


def test_query_hybrid_fact_embeddings_deduplicates_canonical_paper_group(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:levine2023",
                "kind": "Paper",
                "label": "Low Voltage Clutch",
                "canonical_paper_id": "paper:levine2023",
            },
            {
                "id": "paper:levine2023a",
                "kind": "Paper",
                "label": "Low Voltage Clutch",
                "canonical_paper_id": "paper:levine2023",
                "duplicate_of": "paper:levine2023",
            },
        ],
        "edges": [
            {
                "source": "paper:levine2023a",
                "type": "DUPLICATE_OF",
                "target": "paper:levine2023",
            }
        ],
    }
    fact_text = "Low voltage actuator force improves with liquid droplet heating."
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:levine2023",
                fact_id="fact:levine2023:1",
                fact_type="Mechanism",
                text=fact_text,
                evidence=fact_text,
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            ),
            SemanticFact(
                paper_id="paper:levine2023a",
                fact_id="fact:levine2023a:1",
                fact_type="Mechanism",
                text=fact_text,
                evidence=fact_text,
                confidence=0.8,
                targets=["mechanism:electrothermal_actuation"],
            ),
        ],
    )
    provider = FakeEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    results = query_hybrid_fact_embeddings(
        index_dir,
        "low voltage liquid droplet actuator",
        provider,
        target="mechanism:electrothermal_actuation",
        limit=5,
        score_margin=1.0,
    )

    assert [result["paper_id"] for result in results] == ["paper:levine2023"]
    assert results[0]["canonical_paper_id"] == "paper:levine2023"


def test_query_hybrid_fact_embeddings_includes_strong_lexical_candidates(tmp_path):
    class MisleadingEmbeddingProvider:
        model_name = "misleading-embeddings"

        def embed_documents(self, texts):
            vectors = []
            for text in texts:
                if "Fivefold" in text:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([1.0, 0.0])
            return vectors

        def embed_query(self, text):
            return [1.0, 0.0]

    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {"id": "paper:semantic", "kind": "Paper", "label": "Semantic"},
            {"id": "paper:lexical", "kind": "Paper", "label": "Lexical"},
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:semantic",
                fact_id="fact:semantic:1",
                fact_type="Claim",
                text="Microliquid droplets expand under resistive heating.",
                evidence="microliquid droplets",
                confidence=0.8,
                targets=["metric:force"],
            ),
            SemanticFact(
                paper_id="paper:lexical",
                fact_id="fact:lexical:1",
                fact_type="Claim",
                text="Fivefold force capacity with a 40-fold reduction in voltage.",
                evidence="fivefold force capacity and 40-fold reduction",
                confidence=0.8,
                targets=["metric:force"],
            ),
        ],
    )
    provider = MisleadingEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    results = query_hybrid_fact_embeddings(
        index_dir,
        "fivefold 40-fold force reduction",
        provider,
        target="metric:force",
        limit=1,
        candidate_limit=1,
        keyword_weight=0.1,
        score_margin=1.0,
    )

    assert [result["paper_id"] for result in results] == ["paper:lexical"]


def test_query_hybrid_fact_embeddings_uses_paper_title_signal(tmp_path):
    class ArticleBiasedEmbeddingProvider:
        model_name = "article-biased-embeddings"

        def embed_documents(self, texts):
            vectors = []
            for text in texts:
                if "1-5 V" in text:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([1.0, 0.0])
            return vectors

        def embed_query(self, text):
            return [1.0, 0.0]

    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:generic",
                "kind": "Paper",
                "label": "Generic Dielectric Elastomer Review",
            },
            {
                "id": "paper:title_match",
                "kind": "Paper",
                "label": "Translucent soft robots driven by fluid electrode dielectric elastomer actuators",
            },
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:generic",
                fact_id="fact:generic:1",
                fact_type="Claim",
                text="Dielectric elastomer actuators are used in soft robotics.",
                evidence="soft robotics dielectric elastomer",
                confidence=0.8,
                targets=["concept:Dielectric_Elastomer_Actuator"],
            ),
            SemanticFact(
                paper_id="paper:title_match",
                fact_id="fact:title:1",
                fact_type="Claim",
                text="Fluid electrode DEA bimorphs can actuate soft robots.",
                evidence="fluid electrode DEA",
                confidence=0.8,
                targets=["concept:Dielectric_Elastomer_Actuator"],
            ),
        ],
    )
    provider = ArticleBiasedEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    results = query_hybrid_fact_embeddings(
        index_dir,
        "transparent soft robot using fluid electrodes and dielectric elastomer bending",
        provider,
        target="concept:Dielectric_Elastomer_Actuator",
        limit=1,
    )

    assert [result["paper_id"] for result in results] == ["paper:title_match"]
    assert results[0]["title_match_score"] > 0


def test_query_hybrid_fact_embeddings_does_not_let_title_only_decoy_win(tmp_path):
    class TitleDecoyEmbeddingProvider:
        model_name = "title-decoy-embeddings"

        def embed_documents(self, texts):
            vectors = []
            for text in texts:
                if "exact fact" in text:
                    vectors.append([0.9, 0.1])
                else:
                    vectors.append([1.0, 0.0])
            return vectors

        def embed_query(self, text):
            return [1.0, 0.0]

    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:title_decoy",
                "kind": "Paper",
                "label": "Transparent soft robot using fluid electrodes and dielectric elastomer bending",
            },
            {
                "id": "paper:exact_fact",
                "kind": "Paper",
                "label": "Specific actuator paper",
            },
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:title_decoy",
                fact_id="fact:title_decoy:1",
                fact_type="Claim",
                text="This article introduces a materials platform.",
                evidence="general platform",
                confidence=0.8,
                targets=["concept:Dielectric_Elastomer_Actuator"],
            ),
            SemanticFact(
                paper_id="paper:exact_fact",
                fact_id="fact:exact:1",
                fact_type="Claim",
                text="Transparent soft robot uses fluid electrodes for dielectric elastomer bending.",
                evidence="exact fact: fluid electrodes dielectric elastomer bending",
                confidence=0.8,
                targets=["concept:Dielectric_Elastomer_Actuator"],
            ),
        ],
    )
    provider = TitleDecoyEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    results = query_hybrid_fact_embeddings(
        index_dir,
        "transparent soft robot using fluid electrodes and dielectric elastomer bending",
        provider,
        target="concept:Dielectric_Elastomer_Actuator",
        limit=1,
        score_margin=1.0,
    )

    assert [result["paper_id"] for result in results] == ["paper:exact_fact"]


def test_query_hybrid_fact_embeddings_boosts_review_title_for_review_queries(tmp_path):
    class GenericEmbeddingProvider:
        model_name = "generic-embeddings"

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:article",
                "kind": "Paper",
                "label": "Low-voltage ionic actuator article",
            },
            {
                "id": "paper:review",
                "kind": "Paper",
                "label": "A Review of Ionic Polymeric Soft Actuators and Sensors",
            },
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:article",
                fact_id="fact:article:1",
                fact_type="Claim",
                text="A generic ionic polymer actuator article mentions low-voltage operation.",
                evidence="low voltage operation",
                confidence=0.8,
                targets=["metric:voltage"],
            ),
            SemanticFact(
                paper_id="paper:review",
                fact_id="fact:review:1",
                fact_type="Claim",
                text="IPMCs are ionic polymer actuators with actuation voltage of 1-5 V.",
                evidence="actuation voltage 1-5 V",
                confidence=0.8,
                targets=["metric:voltage"],
            ),
        ],
    )
    provider = GenericEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    results = query_hybrid_fact_embeddings(
        index_dir,
        "ionic polymer actuator review reporting one to five volt operation",
        provider,
        target="metric:voltage",
        limit=1,
    )

    assert [result["paper_id"] for result in results] == ["paper:review"]


def test_query_hybrid_fact_embeddings_penalizes_review_title_without_review_intent(tmp_path):
    class GenericEmbeddingProvider:
        model_name = "generic-embeddings"

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:specific",
                "kind": "Paper",
                "label": "Low-voltage high-force actuator article",
            },
            {
                "id": "paper:review",
                "kind": "Paper",
                "label": "Review of electrically driven soft actuators",
            },
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:specific",
                fact_id="fact:specific:1",
                fact_type="Claim",
                text="Force capacity improves while drive voltage is reduced.",
                evidence="force capacity reduced drive voltage",
                confidence=0.8,
                targets=["metric:force"],
            ),
            SemanticFact(
                paper_id="paper:review",
                fact_id="fact:review:1",
                fact_type="Claim",
                text="Force and voltage are discussed for soft actuators.",
                evidence="force voltage soft actuators",
                confidence=0.8,
                targets=["metric:force"],
            ),
        ],
    )
    provider = GenericEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    results = query_hybrid_fact_embeddings(
        index_dir,
        "actuator design that boosts force while greatly reducing drive voltage",
        provider,
        target="metric:force",
        limit=1,
        score_margin=1.0,
    )

    assert [result["paper_id"] for result in results] == ["paper:specific"]
    assert results[0]["review_penalty"] == 0.0


def test_query_hybrid_fact_embeddings_uses_body_overlap_for_specific_claims(tmp_path):
    class ReviewBiasedEmbeddingProvider:
        model_name = "review-biased-embeddings"

        def embed_documents(self, texts):
            vectors = []
            for text in texts:
                if "Intrinsic charge trapping" in text:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.9, 0.1])
            return vectors

        def embed_query(self, text):
            return [1.0, 0.0]

    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [
            {
                "id": "paper:review",
                "kind": "Paper",
                "label": "Intrinsic charge trapping in organic semiconductors",
            },
            {
                "id": "paper:specific",
                "kind": "Paper",
                "label": "Charge Trapping in a Low-Crystalline Conjugated Polymer",
            },
        ],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:review",
                fact_id="fact:review:1",
                fact_type="Claim",
                text="Intrinsic charge trapping in organic semiconductors.",
                evidence="organic semiconductors",
                confidence=0.8,
                targets=["concept:Charge_Trapping"],
            ),
            SemanticFact(
                paper_id="paper:specific",
                fact_id="fact:specific:1",
                fact_type="Claim",
                text=(
                    "Crystallinity in organic semiconductors controls "
                    "charge-trapping relationships."
                ),
                evidence="organic semiconductor crystallinity charge trap relationships",
                confidence=0.8,
                targets=["concept:Charge_Trapping"],
            ),
        ],
    )
    provider = ReviewBiasedEmbeddingProvider()
    build_fact_embedding_index(index_dir, provider)

    results = query_hybrid_fact_embeddings(
        index_dir,
        "organic semiconductor crystallinity controls charge trap relationships",
        provider,
        target="concept:Charge_Trapping",
        limit=1,
        score_margin=1.0,
    )

    assert [result["paper_id"] for result in results] == ["paper:specific"]
    assert results[0]["body_match_score"] > 0


class _FixedDimProvider:
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim

    def embed_documents(self, texts):
        return [[0.5] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.5] * self.dim


def _minimal_index(tmp_path):
    index_dir = tmp_path / ".graph_index"
    graph = {
        "nodes": [{"id": "paper:x", "kind": "Paper", "label": "X"}],
        "edges": [],
    }
    write_index(
        graph,
        index_dir,
        facts=[
            SemanticFact(
                paper_id="paper:x",
                fact_id="fact:x:1",
                fact_type="Claim",
                text="Some claim.",
                evidence="some evidence",
                confidence=0.9,
                targets=[],
            )
        ],
    )
    return index_dir


def test_build_writes_sidecar_meta(tmp_path):
    index_dir = _minimal_index(tmp_path)
    provider = _FixedDimProvider(model_name="fake-A", dim=8)
    build_fact_embedding_index(index_dir, provider)

    meta_path = index_dir / EMBEDDING_META_FILE
    assert meta_path.exists(), "fact_embeddings_meta.json was not written"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["model"] == "fake-A"
    assert meta["dim"] == 8


def test_query_raises_on_model_and_dim_mismatch(tmp_path):
    import pytest

    index_dir = _minimal_index(tmp_path)
    build_provider = _FixedDimProvider(model_name="fake-A", dim=8)
    build_fact_embedding_index(index_dir, build_provider)

    query_provider = _FixedDimProvider(model_name="fake-B", dim=16)
    with pytest.raises(ValueError, match="mismatch"):
        query_fact_embeddings(index_dir, "some query", query_provider, limit=1)
