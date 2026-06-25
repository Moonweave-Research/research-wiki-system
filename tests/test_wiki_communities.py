import json
import sqlite3


from legacy.graphrag.wiki_communities import (
    _build_paper_concept_map,
    _build_projection,
    run,
)
from legacy.graphrag.wiki_graph import write_sqlite


def _make_graph(papers: list[str], concepts: list[str], links: list[tuple[str, str]]) -> dict:
    nodes = [{"id": f"paper:{p}", "kind": "Paper", "title": p} for p in papers]
    nodes += [{"id": f"concept:{c}", "kind": "Concept", "title": c} for c in concepts]
    edges = []
    for paper, concept in links:
        fact_id = f"fact:{paper}_{concept}"
        nodes.append({"id": fact_id, "kind": "Fact", "title": f"{paper}-{concept}"})
        edges.append({"source": f"paper:{paper}", "type": "HAS_FACT", "target": fact_id})
        edges.append({"source": fact_id, "type": "ABOUT", "target": f"concept:{concept}"})
    return {"nodes": nodes, "edges": edges}


def test_paper_paper_projection_weights(tmp_path):
    graph = _make_graph(
        papers=["a", "b", "c", "d"],
        concepts=["X", "Y", "Z"],
        links=[
            ("a", "X"),
            ("a", "Y"),
            ("b", "X"),
            ("b", "Y"),
            ("b", "Z"),
            ("c", "X"),
        ],
    )
    db_path = tmp_path / "graph.sqlite"
    write_sqlite(db_path, graph)

    conn = sqlite3.connect(db_path)
    try:
        paper_concepts = _build_paper_concept_map(conn)
    finally:
        conn.close()

    assert "paper:d" not in paper_concepts
    assert paper_concepts["paper:a"] == {"concept:X", "concept:Y"}
    assert paper_concepts["paper:b"] == {"concept:X", "concept:Y", "concept:Z"}
    assert paper_concepts["paper:c"] == {"concept:X"}

    sorted_papers, edges = _build_projection(paper_concepts, min_shared=1)
    edge_weights = {tuple(sorted([sorted_papers[u], sorted_papers[v]])): w for u, v, w in edges}

    assert edge_weights[("paper:a", "paper:b")] == 2
    assert edge_weights[("paper:a", "paper:c")] == 1
    assert edge_weights[("paper:b", "paper:c")] == 1
    assert ("paper:a", "paper:d") not in edge_weights
    assert ("paper:b", "paper:d") not in edge_weights
    assert ("paper:c", "paper:d") not in edge_weights


def test_leiden_assigns_all_papers(tmp_path, capsys):
    graph = _make_graph(
        papers=["p1", "p2", "p3", "p4", "p5", "p6", "isolated"],
        concepts=["X", "Y", "Z", "W"],
        links=[
            ("p1", "X"),
            ("p1", "Y"),
            ("p2", "X"),
            ("p2", "Y"),
            ("p3", "X"),
            ("p3", "Y"),
            ("p4", "Z"),
            ("p4", "W"),
            ("p5", "Z"),
            ("p5", "W"),
            ("p6", "Z"),
            ("p6", "W"),
        ],
    )
    db_path = tmp_path / "graph.sqlite"
    write_sqlite(db_path, graph)

    run(graph_index=tmp_path, resolution=1.0, min_shared_concepts=1, seed=42)

    captured = capsys.readouterr().out
    modularity_line = next(line for line in captured.splitlines() if line.startswith("modularity:"))
    modularity = float(modularity_line.split(":")[1].strip())
    assert modularity >= 0

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("select id, props_json from nodes where kind='Paper'").fetchall()
    finally:
        conn.close()

    community_ids = {}
    for paper_id, props_json_str in rows:
        props = json.loads(props_json_str or "{}")
        assert "community_id" in props, f"{paper_id} missing community_id"
        community_ids[paper_id] = props["community_id"]

    assert community_ids["paper:isolated"] == -1

    connected_cids = {cid for pid, cid in community_ids.items() if pid != "paper:isolated"}
    assert len(set(connected_cids)) > 1


def test_community_persistence_idempotent(tmp_path, capsys):
    graph = _make_graph(
        papers=["p1", "p2", "p3", "p4", "p5", "p6"],
        concepts=["X", "Y", "Z", "W"],
        links=[
            ("p1", "X"),
            ("p1", "Y"),
            ("p2", "X"),
            ("p2", "Y"),
            ("p3", "X"),
            ("p3", "Y"),
            ("p4", "Z"),
            ("p4", "W"),
            ("p5", "Z"),
            ("p5", "W"),
            ("p6", "Z"),
            ("p6", "W"),
        ],
    )
    db_path = tmp_path / "graph.sqlite"
    write_sqlite(db_path, graph)

    run(graph_index=tmp_path, resolution=1.0, min_shared_concepts=1, seed=42)

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("select id, props_json from nodes where kind='Paper'").fetchall()
    finally:
        conn.close()
    first_run = {pid: json.loads(pj or "{}").get("community_id") for pid, pj in rows}

    capsys.readouterr()

    run(graph_index=tmp_path, resolution=1.0, min_shared_concepts=1, seed=42)

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("select id, props_json from nodes where kind='Paper'").fetchall()
    finally:
        conn.close()
    second_run = {pid: json.loads(pj or "{}").get("community_id") for pid, pj in rows}

    assert first_run == second_run
