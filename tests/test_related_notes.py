import json
import sqlite3
import time

import numpy as np

from scripts.search import related_notes


def _make_inputs(tmp_path, notes=("A", "B", "C")):
    # 3 papers: A and B share a direction (near), C orthogonal (far).
    # 2 facts each so the mean-of-facts path is exercised.
    vecs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # paper:A
            [0.95, 0.05, 0.0],
            [0.85, 0.15, 0.0],  # paper:B (close to A)
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],  # paper:C (far)
        ],
        dtype=np.float32,
    )
    npz = tmp_path / "fe.npz"
    np.savez(npz, vectors=vecs)
    manifest = tmp_path / "manifest.jsonl"
    ids = ["paper:A", "paper:A", "paper:B", "paper:B", "paper:C", "paper:C"]
    manifest.write_text(
        "".join(json.dumps({"paper_id": pid}) + "\n" for pid in ids), encoding="utf-8"
    )
    research = tmp_path / "wiki" / "research"
    research.mkdir(parents=True)
    for name in notes:
        (research / f"{name}.md").write_text("note", encoding="utf-8")
    return npz, manifest, tmp_path / "wiki"


def test_paper_vectors_means_and_normalizes(tmp_path):
    npz, manifest, vault = _make_inputs(tmp_path)
    paths, matrix = related_notes.paper_vectors(npz, manifest, vault)
    assert set(paths) == {"research/A.md", "research/B.md", "research/C.md"}
    # rows are L2-normalized
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_paper_vectors_skips_missing_note(tmp_path):
    # C's note is absent (e.g. archived as a duplicate) -> dropped from the graph.
    npz, manifest, vault = _make_inputs(tmp_path, notes=("A", "B"))
    paths, _ = related_notes.paper_vectors(npz, manifest, vault)
    assert set(paths) == {"research/A.md", "research/B.md"}


def test_compute_related_ranks_near_paper_first(tmp_path):
    npz, manifest, vault = _make_inputs(tmp_path)
    paths, matrix = related_notes.paper_vectors(npz, manifest, vault)
    edges = related_notes.compute_related(paths, matrix, top_k=1, min_score=0.5)
    top = {src: dst for src, dst, _ in edges}
    assert top["research/A.md"] == "research/B.md"  # A's nearest is B, not C
    assert top["research/B.md"] == "research/A.md"


def test_build_index_writes_sqlite(tmp_path):
    npz, manifest, vault = _make_inputs(tmp_path)
    db = tmp_path / "related.sqlite"
    papers, edge_count = related_notes.build_index(
        npz, manifest, db, top_k=2, min_score=0.5, vault_root=vault
    )
    assert papers == 3
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT dst_path, score FROM related WHERE src_path='research/A.md' ORDER BY score DESC"
        ).fetchall()
    finally:
        conn.close()
    assert rows[0][0] == "research/B.md"
    assert rows[0][1] > 0.9  # A–B are very close


def test_build_index_skips_current_output_without_rewriting(tmp_path):
    npz, manifest, vault = _make_inputs(tmp_path)
    db = tmp_path / "related.sqlite"
    first = related_notes.build_index(
        npz, manifest, db, top_k=2, min_score=0.5, vault_root=vault
    )
    mtime_ns = db.stat().st_mtime_ns
    time.sleep(0.01)

    second = related_notes.build_index(
        npz, manifest, db, top_k=2, min_score=0.5, vault_root=vault
    )

    assert second == first
    assert db.stat().st_mtime_ns == mtime_ns


def test_build_index_fast_path_skips_full_input_hashes(tmp_path, monkeypatch):
    npz, manifest, vault = _make_inputs(tmp_path)
    db = tmp_path / "related.sqlite"
    first = related_notes.build_index(
        npz, manifest, db, top_k=2, min_score=0.5, vault_root=vault
    )
    related_notes.build_index(npz, manifest, db, top_k=2, min_score=0.5, vault_root=vault)

    def fail_file_signature(path):
        raise AssertionError(f"current relatedness fast path should not hash {path}")

    monkeypatch.setattr(related_notes, "_file_signature", fail_file_signature)

    second = related_notes.build_index(
        npz, manifest, db, top_k=2, min_score=0.5, vault_root=vault
    )

    assert second == first
