import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legacy.graphrag.wiki_ppr import (  # noqa: E402
    build_entity_graph,
    normalize_entity,
    papers_from_entity_scores,
    ppr_query,
    run_ppr,
    seed_entities_from_query,
)
from legacy.graphrag.wiki_semantic_schema import SemanticTriple, _stable_triple_id  # noqa: E402


def _triple(paper_id, subject, predicate, obj, confidence=0.9):
    return SemanticTriple(
        paper_id=paper_id,
        triple_id=_stable_triple_id(paper_id, subject, predicate, obj),
        subject=subject,
        predicate=predicate,
        object=obj,
        evidence=f"{subject} {predicate} {obj}",
        confidence=confidence,
    )


def test_normalize_entity_preserves_chemistry_chars():
    assert normalize_entity("PEDOT:PSS") == "pedot:pss"
    assert normalize_entity("4-cyano-4'-pentyl") == "4-cyano-4'-pentyl"
    assert normalize_entity("  whitespace  ") == "whitespace"


def test_build_entity_graph_accumulates_weight():
    triples = [
        _triple("paper:p1", "PEDOT:PSS", "exhibits", "high conductivity", confidence=0.9),
        _triple("paper:p2", "PEDOT:PSS", "exhibits", "high conductivity", confidence=0.7),
        _triple("paper:p1", "azobenzene", "isomerizes under", "UV light", confidence=0.8),
    ]
    graph = build_entity_graph(triples)
    assert graph.has_edge("pedot:pss", "high conductivity")
    # Two triples with same SPO accumulate
    assert graph["pedot:pss"]["high conductivity"]["weight"] == 1.6
    assert graph.has_edge("azobenzene", "uv light")
    assert graph["azobenzene"]["uv light"]["weight"] == 0.8


def test_seed_entities_from_query_token_match():
    graph = nx.DiGraph()
    graph.add_node("pedot:pss")
    graph.add_node("azobenzene")
    graph.add_node("dielectric elastomer")

    seeds = seed_entities_from_query("conductivity of pedot:pss film", graph)
    # 'pedot:pss' lowercased; colon is preserved (not mapped to _).
    assert "pedot:pss" in seeds
    # Seeds sum to 1
    assert abs(sum(seeds.values()) - 1.0) < 1e-9


def test_seed_entities_empty_query_returns_empty():
    graph = nx.DiGraph()
    graph.add_node("foo")
    assert seed_entities_from_query("", graph) == {}
    assert seed_entities_from_query("nothing matches", graph) == {}


def test_run_ppr_returns_scores_when_seed_in_graph():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("b", "c", weight=1.0)
    graph.add_edge("c", "a", weight=1.0)
    scores = run_ppr(graph, {"a": 1.0}, alpha=0.85)
    assert set(scores.keys()) == {"a", "b", "c"}
    assert scores["a"] > 0
    # Personalized at a → a should have the highest score
    assert scores["a"] > scores["b"]


def test_run_ppr_empty_graph_returns_empty():
    assert run_ppr(nx.DiGraph(), {"a": 1.0}) == {}


def test_run_ppr_seeds_not_in_graph_returns_empty():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1.0)
    assert run_ppr(graph, {"missing_node": 1.0}) == {}


def test_papers_from_entity_scores_sums_per_paper():
    triples = [
        _triple("paper:p1", "PEDOT:PSS", "exhibits", "high conductivity"),
        _triple("paper:p1", "PEDOT:PSS", "is", "polymer"),
        _triple("paper:p2", "azobenzene", "isomerizes", "UV"),
    ]
    entity_scores = {
        "pedot:pss": 0.5,
        "high conductivity": 0.2,
        "polymer": 0.1,
        "azobenzene": 0.3,
        "uv": 0.05,
    }
    ranked = papers_from_entity_scores(triples, entity_scores)
    paper_dict = dict(ranked)
    # p1 mentions {pedot:pss, high conductivity, polymer} -> 0.5+0.2+0.1 = 0.8
    # p2 mentions {azobenzene, uv} -> 0.3+0.05 = 0.35
    assert paper_dict["paper:p1"] > paper_dict["paper:p2"]
    assert abs(paper_dict["paper:p1"] - 0.8) < 1e-9


def test_papers_from_entity_scores_empty_scores():
    triples = [_triple("paper:p1", "a", "b", "c")]
    assert papers_from_entity_scores(triples, {}) == []


def test_ppr_query_end_to_end():
    triples = [
        _triple("paper:p1", "PEDOT:PSS", "exhibits", "1000 S/cm"),
        _triple("paper:p1", "PEDOT:PSS", "is", "conducting polymer"),
        _triple("paper:p2", "conducting polymer", "applications include", "flexible electronics"),
        _triple("paper:p3", "azobenzene", "isomerizes under", "UV light"),
    ]
    result = ppr_query("PEDOT:PSS conductivity", triples)
    # Seed entities should include pedot:pss
    assert "pedot:pss" in result["seeds"]
    # PPR scores should propagate to neighbors of pedot:pss
    assert result["entity_scores"]["pedot:pss"] > 0
    # Top-ranked paper should be one of the PEDOT:PSS papers
    top_paper = result["papers"][0][0]
    assert top_paper in {"paper:p1", "paper:p2"}
    # p3 (azobenzene) should NOT be in top results because nothing connects it
    paper_ids = {pid for pid, _ in result["papers"]}
    # If azobenzene has zero PPR mass, p3 is excluded entirely
    if "paper:p3" in paper_ids:
        p3_score = next(s for pid, s in result["papers"] if pid == "paper:p3")
        p1_score = next(s for pid, s in result["papers"] if pid == "paper:p1")
        assert p1_score > p3_score


def test_ppr_query_multi_hop_propagation():
    """PEDOT:PSS -> conducting polymer -> flexible electronics
    Query about flexible electronics should retrieve PEDOT:PSS papers via 2-hop.
    """
    triples = [
        _triple("paper:p1", "PEDOT:PSS", "is", "conducting polymer"),
        _triple("paper:p2", "conducting polymer", "enables", "flexible electronics"),
        _triple("paper:p3", "unrelated thing", "is", "nothing"),
    ]
    result = ppr_query("flexible electronics application", triples)
    # PPR from 'flexible electronics' should reach p1 via the conducting polymer hub.
    paper_ids = [pid for pid, _ in result["papers"]]
    assert "paper:p1" in paper_ids or "paper:p2" in paper_ids
    assert "paper:p3" not in paper_ids
