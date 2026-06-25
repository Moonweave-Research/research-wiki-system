import sqlite3
import time

from scripts.search import graph_communities


def test_detect_communities_separates_clusters():
    # two disjoint triangles -> two communities
    nodes = ["a", "b", "c", "d", "e", "f"]
    edges = {("a", "b"), ("b", "c"), ("a", "c"), ("d", "e"), ("e", "f"), ("d", "f")}
    comm = graph_communities.detect_communities(nodes, edges)
    assert comm["a"] == comm["b"] == comm["c"]
    assert comm["d"] == comm["e"] == comm["f"]
    assert comm["a"] != comm["d"]


def _minimal_indexes(tmp_path):
    lg = tmp_path / "lg.sqlite"
    conn = sqlite3.connect(lg)
    conn.executescript(
        "CREATE TABLE notes (path TEXT);"
        "CREATE TABLE links (src_path TEXT, dst_target TEXT, dst_resolved INTEGER);"
    )
    paths = [f"research/{x}.md" for x in "abcdef"]
    conn.executemany("INSERT INTO notes VALUES (?)", [(p,) for p in paths])
    tri = [("a", "b"), ("b", "c"), ("a", "c"), ("d", "e"), ("e", "f"), ("d", "f")]
    conn.executemany(
        "INSERT INTO links VALUES (?, ?, 1)",
        [(f"research/{a}.md", f"research/{b}.md") for a, b in tri],
    )
    conn.commit()
    conn.close()
    related = tmp_path / "related.sqlite"
    rconn = sqlite3.connect(related)
    rconn.execute("CREATE TABLE related (src_path TEXT, dst_path TEXT, score REAL)")
    rconn.commit()
    rconn.close()
    return lg, related


def test_build_index_writes_community_per_node(tmp_path):
    lg, related = _minimal_indexes(tmp_path)
    db = tmp_path / "communities.sqlite"
    n_nodes, n_comms = graph_communities.build_index(lg, related, db)
    assert n_nodes == 6
    assert n_comms == 2  # the two triangles
    conn = sqlite3.connect(db)
    try:
        rows = dict(conn.execute("SELECT path, community FROM community"))
    finally:
        conn.close()
    assert len(rows) == 6
    assert rows["research/a.md"] == rows["research/b.md"]
    assert rows["research/a.md"] != rows["research/d.md"]


def test_build_index_skips_current_output_without_rewriting(tmp_path):
    lg, related = _minimal_indexes(tmp_path)
    db = tmp_path / "communities.sqlite"
    first = graph_communities.build_index(lg, related, db)
    mtime_ns = db.stat().st_mtime_ns
    time.sleep(0.01)

    second = graph_communities.build_index(lg, related, db)

    assert second == first
    assert db.stat().st_mtime_ns == mtime_ns
