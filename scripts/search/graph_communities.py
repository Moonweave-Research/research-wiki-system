#!/usr/bin/env python3
"""Precompute graph communities (Leiden) over the rendered graph — wikilinks plus
each node's top-K similar edges — so the UI can color nodes by cluster without
running community detection per request. Mirrors the backend graph builder's edge
set (research/concepts nodes, similar capped to top-K). Writes
.graph_index/communities.sqlite.

    uv run python -m scripts.search.graph_communities build
"""

import argparse
import collections
import hashlib
import json
import sqlite3
from pathlib import Path

import igraph as ig
import leidenalg as la

DEFAULT_LINKGRAPH = ".graph_index/linkgraph.sqlite"
DEFAULT_RELATED = ".graph_index/related.sqlite"
DEFAULT_DB = ".graph_index/communities.sqlite"
GRAPH_FOLDERS = ("research", "concepts")
SIMILAR_TOP_K = 2  # must match backend services.GRAPH_SIMILAR_TOP_K
RESOLUTION = 1.0

SCHEMA = "CREATE TABLE community (path TEXT PRIMARY KEY, community INTEGER);"


def _file_signature(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _meta_path(db_path: Path) -> Path:
    return db_path.with_suffix(db_path.suffix + ".meta.json")


def _input_signature(
    linkgraph_db: Path,
    related_db: Path,
    top_k: int,
    resolution: float,
) -> dict:
    return {
        "linkgraph_sha256": _file_signature(linkgraph_db),
        "related_sha256": _file_signature(related_db),
        "top_k": top_k,
        "resolution": resolution,
    }


def _current_index_counts(db_path: Path, signature: dict) -> tuple[int, int] | None:
    meta_path = _meta_path(db_path)
    if not db_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signature") != signature:
            return None
        return int(meta["nodes"]), int(meta["communities"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, sqlite3.Error):
        return None


def _write_meta(db_path: Path, signature: dict, nodes: int, communities: int) -> None:
    _meta_path(db_path).write_text(
        json.dumps(
            {"signature": signature, "nodes": nodes, "communities": communities},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _in_set(path: str) -> bool:
    return path.split("/", 1)[0] in GRAPH_FOLDERS


def graph_edges(
    linkgraph_db: Path, related_db: Path, top_k: int = SIMILAR_TOP_K
) -> tuple[list[str], set[tuple[str, str]]]:
    """research/concepts nodes + undirected (wikilink ∪ top-K similar) edges."""
    lconn = sqlite3.connect(linkgraph_db)
    try:
        nodes = sorted(p for (p,) in lconn.execute("SELECT path FROM notes") if _in_set(p))
        node_set = set(nodes)
        edges: set[tuple[str, str]] = set()
        for src, dst in lconn.execute(
            "SELECT src_path, dst_target FROM links WHERE dst_resolved=1"
        ):
            if src in node_set and dst in node_set and src != dst:
                edges.add((src, dst) if src < dst else (dst, src))
    finally:
        lconn.close()

    rconn = sqlite3.connect(related_db)
    try:
        kept: collections.Counter[str] = collections.Counter()
        for src, dst in rconn.execute("SELECT src_path, dst_path FROM related ORDER BY score DESC"):
            if src not in node_set or dst not in node_set or src == dst:
                continue
            if kept[src] >= top_k:
                continue
            kept[src] += 1
            edges.add((src, dst) if src < dst else (dst, src))
    finally:
        rconn.close()
    return nodes, edges


def detect_communities(
    nodes: list[str],
    edges: set[tuple[str, str]],
    resolution: float = RESOLUTION,
    seed: int = 42,
) -> dict[str, int]:
    index = {path: i for i, path in enumerate(nodes)}
    graph = ig.Graph(n=len(nodes), edges=[(index[a], index[b]) for a, b in edges])
    partition = la.find_partition(
        graph, la.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=seed
    )
    return {nodes[i]: partition.membership[i] for i in range(len(nodes))}


def build_index(
    linkgraph_db: Path,
    related_db: Path,
    db_path: Path,
    top_k: int = SIMILAR_TOP_K,
    resolution: float = RESOLUTION,
) -> tuple[int, int]:
    signature = _input_signature(linkgraph_db, related_db, top_k, resolution)
    current = _current_index_counts(db_path, signature)
    if current is not None:
        return current
    nodes, edges = graph_edges(linkgraph_db, related_db, top_k)
    membership = detect_communities(nodes, edges, resolution)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO community VALUES (?, ?)", list(membership.items()))
        conn.commit()
    finally:
        conn.close()
    communities = len(set(membership.values()))
    _write_meta(db_path, signature, len(nodes), communities)
    return len(nodes), communities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precompute graph communities (Leiden).")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Rebuild .graph_index/communities.sqlite")
    build.add_argument("--linkgraph", default=DEFAULT_LINKGRAPH)
    build.add_argument("--related", default=DEFAULT_RELATED)
    build.add_argument("--db", default=DEFAULT_DB)
    build.add_argument("--top-k", type=int, default=SIMILAR_TOP_K)
    build.add_argument("--resolution", type=float, default=RESOLUTION)
    args = parser.parse_args(argv)

    nodes, comms = build_index(
        Path(args.linkgraph), Path(args.related), Path(args.db), args.top_k, args.resolution
    )
    print(f"graph communities built: {nodes} nodes, {comms} communities -> {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
