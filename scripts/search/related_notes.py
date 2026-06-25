#!/usr/bin/env python3
"""Build the semantic relatedness index from existing fact embeddings.

Each paper's embedding is the L2-normalized mean of its fact vectors (the validated
e5-large vectors in .graph_index/fact_embeddings.npz — no recompute). For each paper
we keep its top-K cosine-nearest other papers above a threshold and store them as
`similar` edges in .graph_index/related.sqlite. This is the Phase 3 "organic"
paper-to-paper layer; explicit wikilinks (linkgraph.sqlite) are untouched.

    uv run python -m scripts.search.related_notes build [--top-k 8] [--min-score 0.5]
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_NPZ = ".graph_index/fact_embeddings.npz"
DEFAULT_MANIFEST = ".graph_index/fact_embeddings_manifest.jsonl"
DEFAULT_DB = ".graph_index/related.sqlite"
DEFAULT_TOP_K = 8
DEFAULT_MIN_SCORE = 0.5

SCHEMA = """
CREATE TABLE related (
    src_path TEXT,
    dst_path TEXT,
    score REAL
);
CREATE INDEX idx_related_src ON related(src_path);
CREATE INDEX idx_related_dst ON related(dst_path);
"""


def _file_signature(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vault_signature(vault_root: Path) -> list[list[str | int]]:
    research_dir = vault_root / "research"
    if not research_dir.is_dir():
        return []
    return [
        [str(path.relative_to(vault_root)), path.stat().st_size, path.stat().st_mtime_ns]
        for path in sorted(research_dir.glob("*.md"))
    ]


def _file_stat_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def _meta_path(db_path: Path) -> Path:
    return db_path.with_suffix(db_path.suffix + ".meta.json")


def _fast_signature(
    npz_path: Path,
    manifest_path: Path,
    top_k: int,
    min_score: float,
    vault_root: Path,
) -> dict:
    return {
        "npz": _file_stat_signature(npz_path),
        "manifest": _file_stat_signature(manifest_path),
        "top_k": top_k,
        "min_score": min_score,
        "vault": _vault_signature(vault_root),
    }


def _input_signature(
    npz_path: Path,
    manifest_path: Path,
    top_k: int,
    min_score: float,
    vault_root: Path,
) -> dict:
    return {
        "npz_sha256": _file_signature(npz_path),
        "manifest_sha256": _file_signature(manifest_path),
        "top_k": top_k,
        "min_score": min_score,
        "vault": _vault_signature(vault_root),
    }


def _current_index_counts(db_path: Path, signature: dict) -> tuple[int, int] | None:
    meta_path = _meta_path(db_path)
    if not db_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signature") != signature:
            return None
        return int(meta["papers"]), int(meta["edges"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, sqlite3.Error):
        return None


def _current_index_counts_fast(db_path: Path, fast_signature: dict) -> tuple[int, int] | None:
    meta_path = _meta_path(db_path)
    if not db_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fast_signature") != fast_signature:
            return None
        return int(meta["papers"]), int(meta["edges"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, sqlite3.Error):
        return None


def _write_meta(
    db_path: Path,
    signature: dict,
    fast_signature: dict,
    papers: int,
    edges: int,
) -> None:
    _meta_path(db_path).write_text(
        json.dumps(
            {
                "signature": signature,
                "fast_signature": fast_signature,
                "papers": papers,
                "edges": edges,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _paper_id_to_path(paper_id: str) -> str:
    citekey = paper_id.split(":", 1)[-1] if paper_id.startswith("paper:") else paper_id
    return f"research/{citekey}.md"


def paper_vectors(
    npz_path: Path, manifest_path: Path, vault_root: Path = Path("wiki")
) -> tuple[list[str], np.ndarray]:
    """Return (paper_paths, normalized_matrix) — one mean vector per paper.

    The cached fact embeddings outlive note deletions (papers archived as
    duplicates), so a paper whose note no longer exists under vault_root is
    skipped — otherwise its stale path would re-enter the graph as a phantom edge.
    """
    vectors = np.load(npz_path)["vectors"].astype(np.float32)
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for row, line in enumerate(open(manifest_path, encoding="utf-8")):
        if not line.strip():
            continue
        paper_id = json.loads(line)["paper_id"]
        if paper_id not in sums:
            sums[paper_id] = np.zeros(vectors.shape[1], dtype=np.float32)
            counts[paper_id] = 0
        sums[paper_id] += vectors[row]
        counts[paper_id] += 1

    paths: list[str] = []
    means = []
    for paper_id, total in sums.items():
        path = _paper_id_to_path(paper_id)
        if not (vault_root / path).exists():
            continue
        mean = total / counts[paper_id]
        norm = np.linalg.norm(mean)
        if norm == 0:
            continue
        paths.append(path)
        means.append(mean / norm)
    return paths, np.vstack(means)


def compute_related(
    paths: list[str], matrix: np.ndarray, top_k: int, min_score: float
) -> list[tuple[str, str, float]]:
    """Top-K nearest other papers per paper, above min_score."""
    sims = matrix @ matrix.T  # cosine (rows are L2-normalized)
    np.fill_diagonal(sims, -1.0)  # exclude self
    edges: list[tuple[str, str, float]] = []
    for i, src in enumerate(paths):
        order = np.argsort(-sims[i])[:top_k]
        for j in order:
            score = float(sims[i, j])
            if score < min_score:
                break
            edges.append((src, paths[j], round(score, 4)))
    return edges


def build_index(
    npz_path: Path,
    manifest_path: Path,
    db_path: Path,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    vault_root: Path = Path("wiki"),
) -> tuple[int, int]:
    fast_signature = _fast_signature(npz_path, manifest_path, top_k, min_score, vault_root)
    current = _current_index_counts_fast(db_path, fast_signature)
    if current is not None:
        return current
    signature = _input_signature(npz_path, manifest_path, top_k, min_score, vault_root)
    current = _current_index_counts(db_path, signature)
    if current is not None:
        _write_meta(db_path, signature, fast_signature, current[0], current[1])
        return current
    paths, matrix = paper_vectors(npz_path, manifest_path, vault_root)
    edges = compute_related(paths, matrix, top_k, min_score)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO related VALUES (?, ?, ?)", edges)
        conn.commit()
    finally:
        conn.close()
    _write_meta(db_path, signature, fast_signature, len(paths), len(edges))
    return len(paths), len(edges)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the semantic relatedness index.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Rebuild .graph_index/related.sqlite")
    build.add_argument("--npz", default=DEFAULT_NPZ)
    build.add_argument("--manifest", default=DEFAULT_MANIFEST)
    build.add_argument("--db", default=DEFAULT_DB)
    build.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    build.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    build.add_argument("--vault", default="wiki")
    args = parser.parse_args(argv)

    papers, edges = build_index(
        Path(args.npz),
        Path(args.manifest),
        Path(args.db),
        args.top_k,
        args.min_score,
        Path(args.vault),
    )
    print(f"related index built: {papers} papers, {edges} edges -> {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
