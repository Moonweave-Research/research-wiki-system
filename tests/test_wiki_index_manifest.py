import os
from pathlib import Path

from legacy.graphrag.wiki_index_manifest import (
    embedding_index_is_current,
    file_fingerprint,
    write_manifest,
)


def _write_embedding_artifacts(index_dir: Path) -> None:
    index_dir.mkdir()
    (index_dir / "facts.jsonl").write_text('{"fact_id":"fact:x:1"}\n', encoding="utf-8")
    (index_dir / "fact_embeddings.npz").write_bytes(b"fake-vectors")
    (index_dir / "fact_embeddings_manifest.jsonl").write_text(
        '{"fact_id":"fact:x:1"}\n',
        encoding="utf-8",
    )


def test_embedding_index_current_ignores_mtime_only_drift(tmp_path):
    index_dir = tmp_path / ".graph_index"
    _write_embedding_artifacts(index_dir)
    write_manifest(index_dir)

    facts_path = index_dir / "facts.jsonl"
    before = file_fingerprint(facts_path)
    os.utime(facts_path, (facts_path.stat().st_atime + 10, facts_path.stat().st_mtime + 10))
    after = file_fingerprint(facts_path)

    assert before["sha256"] == after["sha256"]
    assert before["mtime_ns"] != after["mtime_ns"]
    assert embedding_index_is_current(index_dir)


def test_embedding_index_stale_when_content_hash_changes(tmp_path):
    index_dir = tmp_path / ".graph_index"
    _write_embedding_artifacts(index_dir)
    write_manifest(index_dir)

    (index_dir / "facts.jsonl").write_text('{"fact_id":"fact:y:1"}\n', encoding="utf-8")

    assert not embedding_index_is_current(index_dir)
