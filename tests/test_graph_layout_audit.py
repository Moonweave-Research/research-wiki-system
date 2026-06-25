import sqlite3
from pathlib import Path

from scripts.search import graph_layout_audit


def _build_linkgraph(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE notes (path TEXT PRIMARY KEY, title TEXT);
        CREATE TABLE links (src_path TEXT, dst_target TEXT, dst_resolved INTEGER);
        """
    )
    conn.executemany(
        "INSERT INTO notes VALUES (?, ?)",
        [
            ("concepts/A.md", "A"),
            ("concepts/B.md", "B"),
            ("research/p1.md", "P1"),
            ("research/p2.md", "P2"),
            ("research/p3.md", "P3"),
            ("meta/Tags/A.md", "Tag A"),
        ],
    )
    conn.executemany(
        "INSERT INTO links VALUES (?, ?, ?)",
        [
            ("concepts/A.md", "research/p1.md", 1),
            ("concepts/A.md", "research/p2.md", 1),
            ("meta/Tags/A.md", "research/p3.md", 1),
        ],
    )
    conn.commit()
    conn.close()


def _build_related(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE related (src_path TEXT, dst_path TEXT, score REAL)")
    conn.executemany(
        "INSERT INTO related VALUES (?, ?, ?)",
        [
            ("research/p1.md", "research/p2.md", 0.9),
            ("research/p1.md", "research/p3.md", 0.8),
            ("research/p2.md", "research/p3.md", 0.7),
        ],
    )
    conn.commit()
    conn.close()


def _build_communities(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE community (path TEXT PRIMARY KEY, community INTEGER)")
    conn.executemany(
        "INSERT INTO community VALUES (?, ?)",
        [
            ("concepts/A.md", 0),
            ("concepts/B.md", 1),
            ("research/p1.md", 0),
            ("research/p2.md", 0),
            ("research/p3.md", 1),
        ],
    )
    conn.commit()
    conn.close()


def test_build_layout_report_measures_global_graph_inputs(tmp_path: Path) -> None:
    linkgraph = tmp_path / "linkgraph.sqlite"
    related = tmp_path / "related.sqlite"
    communities = tmp_path / "communities.sqlite"
    _build_linkgraph(linkgraph)
    _build_related(related)
    _build_communities(communities)

    report = graph_layout_audit.build_layout_report(
        linkgraph_db=linkgraph,
        related_db=related,
        communities_db=communities,
        similar_top_k=2,
    )

    assert report["summary"]["nodes"] == 5
    assert report["summary"]["research_nodes"] == 3
    assert report["summary"]["concept_nodes"] == 2
    assert report["summary"]["explicit_edges"] == 2
    assert report["summary"]["similar_edges_top_k"] == 3
    assert report["summary"]["zero_degree_explicit"] == 2
    assert report["summary"]["zero_degree_with_similar"] == 1
    assert report["communities"]["count"] == 2
    assert report["communities"]["top_12_coverage_percent"] == 100
    assert report["layout_parameters"]["global_link_distance"] == 36
    assert report["layout_parameters"]["global_layout_algorithm"] == "community-ring-v1"
    assert report["layout_parameters"]["local_layout_algorithm"] == "local-radial-v1"
    assert "no_coordinate_cache_or_layout_quality_gate" not in report["risks"]
    assert "no_coordinate_cache" not in report["risks"]
    assert "non_deterministic_force_positions" not in report["risks"]


def test_format_markdown_documents_layout_baseline() -> None:
    report = {
        "summary": {
            "nodes": 5,
            "research_nodes": 3,
            "concept_nodes": 2,
            "explicit_edges": 2,
            "similar_edges_top_k": 3,
            "zero_degree_explicit": 2,
            "zero_degree_with_similar": 1,
        },
        "communities": {
            "count": 2,
            "largest": 3,
            "top_12_coverage_percent": 100,
            "top_12_nodes": 5,
            "long_tail_nodes": 0,
        },
        "layout_parameters": {"global_link_distance": 36, "global_charge": -160},
        "risks": ["ring_anchor_geometry_is_not_semantic_coordinates"],
    }

    markdown = graph_layout_audit.format_markdown(report)

    assert "## Global Graph Inputs" in markdown
    assert "explicit_edges" in markdown
    assert "## Current Layout Parameters" in markdown
    assert "ring_anchor_geometry_is_not_semantic_coordinates" in markdown


def test_quality_gate_reports_pass_and_thresholds() -> None:
    report = {
        "summary": {
            "nodes": 100,
            "zero_degree_explicit": 5,
            "zero_degree_with_similar": 2,
        },
        "communities": {
            "largest": 20,
            "top_12_coverage_percent": 90,
        },
    }

    gate = graph_layout_audit.evaluate_quality_gate(report)

    assert gate["passed"] is True
    assert gate["failures"] == []
    assert gate["thresholds"]["max_zero_degree_with_similar_percent"] == 5


def test_quality_gate_reports_failures_for_blob_and_isolation_risk() -> None:
    report = {
        "summary": {
            "nodes": 100,
            "zero_degree_explicit": 30,
            "zero_degree_with_similar": 12,
        },
        "communities": {
            "largest": 58,
            "top_12_coverage_percent": 97,
        },
    }

    gate = graph_layout_audit.evaluate_quality_gate(report)

    assert gate["passed"] is False
    failed_metrics = {failure["metric"] for failure in gate["failures"]}
    assert failed_metrics == {
        "zero_degree_with_similar_percent",
        "largest_community_percent",
        "top_12_coverage_percent",
    }


def test_build_layout_report_embeds_quality_gate(tmp_path: Path) -> None:
    linkgraph = tmp_path / "linkgraph.sqlite"
    related = tmp_path / "related.sqlite"
    communities = tmp_path / "communities.sqlite"
    _build_linkgraph(linkgraph)
    _build_related(related)
    _build_communities(communities)

    report = graph_layout_audit.build_layout_report(
        linkgraph_db=linkgraph,
        related_db=related,
        communities_db=communities,
        similar_top_k=2,
    )

    assert report["quality_gate"]["passed"] is False
    assert report["quality_gate"]["failures"][0]["metric"] == "zero_degree_with_similar_percent"


def test_build_layout_report_can_benchmark_semantic_layout_candidates(tmp_path: Path) -> None:
    linkgraph = tmp_path / "linkgraph.sqlite"
    related = tmp_path / "related.sqlite"
    communities = tmp_path / "communities.sqlite"
    _build_linkgraph(linkgraph)
    _build_related(related)
    _build_communities(communities)

    report = graph_layout_audit.build_layout_report(
        linkgraph_db=linkgraph,
        related_db=related,
        communities_db=communities,
        similar_top_k=2,
        include_candidates=True,
    )

    candidates = report["layout_candidates"]

    assert candidates["baseline"]["algorithm"] == "community-ring-v1"
    assert candidates["semantic_force"]["algorithm"] == "semantic-force-v1"
    assert candidates["semantic_force"]["status"] == "benchmark_only"
    assert "force_directed_edges" in candidates["semantic_force"]["basis"]
    assert candidates["semantic_force"]["metrics"]["node_count"] == 5
    assert candidates["semantic_force"]["metrics"]["edge_count"] == 5
    assert "median_edge_length_delta" in candidates["semantic_force"]["delta_vs_baseline"]
    assert candidates["recommendation"]["ui_adoption"] in {
        "eligible_for_browser_smoke",
        "keep_current",
    }


def test_format_markdown_includes_layout_candidate_benchmark() -> None:
    report = {
        "summary": {
            "nodes": 5,
            "research_nodes": 3,
            "concept_nodes": 2,
            "explicit_edges": 2,
            "similar_edges_top_k": 3,
            "zero_degree_explicit": 2,
            "zero_degree_with_similar": 1,
        },
        "communities": {
            "count": 2,
            "largest": 3,
            "top_12_coverage_percent": 100,
            "top_12_nodes": 5,
            "long_tail_nodes": 0,
        },
        "layout_parameters": {"global_link_distance": 36, "global_charge": -160},
        "quality_gate": {
            "passed": True,
            "measurements": {
                "zero_degree_with_similar_percent": 1,
                "largest_community_percent": 3,
                "top_12_coverage_percent": 90,
            },
            "failures": [],
        },
        "layout_candidates": {
            "baseline": {
                "algorithm": "community-ring-v1",
                "metrics": {"community_separation_ratio": 2.1},
            },
            "semantic_force": {
                "algorithm": "semantic-force-v1",
                "status": "benchmark_only",
                "basis": ["force_directed_edges", "community_gravity"],
                "metrics": {
                    "median_edge_length": 42.0,
                    "community_separation_ratio": 2.4,
                    "overlap_risk_percent": 0.0,
                },
                "delta_vs_baseline": {
                    "median_edge_length_delta": -8.0,
                    "community_separation_ratio_delta": 0.3,
                    "overlap_risk_percent_delta": 0.0,
                },
            },
            "recommendation": {
                "ui_adoption": "eligible_for_browser_smoke",
                "reason": "candidate_improves_edge_compactness_without_new_overlap_risk",
            },
        },
        "risks": ["ring_anchor_geometry_is_not_semantic_coordinates"],
    }

    markdown = graph_layout_audit.format_markdown(report)

    assert "## Layout Candidate Benchmark" in markdown
    assert "semantic-force-v1" in markdown
    assert "eligible_for_browser_smoke" in markdown
