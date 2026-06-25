from pathlib import Path

from scripts.ingest.zotero_watcher import SyncHandler


def _make_handler(tmp_path: Path, target: Path) -> SyncHandler:
    return SyncHandler(
        target_path=target,
        sync_args=["true"],
        vault_dir=tmp_path,
        debounce_seconds=0.05,
    )


class TestContentHashDedupe:
    """Hash-based skip blocks the BBT-rewrite trigger storm (5/21: 22 entries/day)."""

    def test_hash_changes_when_content_changes(self, tmp_path):
        target = tmp_path / "lib.json"
        target.write_bytes(b'{"items": [1]}')
        handler = _make_handler(tmp_path, target)
        h1 = handler._compute_hash()
        target.write_bytes(b'{"items": [1, 2]}')
        h2 = handler._compute_hash()
        assert h1 != h2

    def test_hash_stable_when_content_unchanged(self, tmp_path):
        target = tmp_path / "lib.json"
        target.write_bytes(b'{"items": [1]}')
        handler = _make_handler(tmp_path, target)
        assert handler._compute_hash() == handler._compute_hash()

    def test_hash_returns_none_when_file_missing(self, tmp_path):
        target = tmp_path / "missing.json"
        handler = _make_handler(tmp_path, target)
        assert handler._compute_hash() is None

    def test_schedule_skips_when_hash_matches_last_synced(self, tmp_path):
        target = tmp_path / "lib.json"
        target.write_bytes(b'{"items": [1]}')
        handler = _make_handler(tmp_path, target)
        handler._last_content_hash = handler._compute_hash()
        handler._schedule_sync(str(target))
        assert handler._timer is None, "duplicate-content event should not arm timer"

    def test_schedule_proceeds_when_hash_differs(self, tmp_path):
        target = tmp_path / "lib.json"
        target.write_bytes(b'{"items": [1]}')
        handler = _make_handler(tmp_path, target)
        handler._last_content_hash = "stale-hash"
        handler._schedule_sync(str(target))
        # Timer should be armed; cancel before the test exits so no sync runs.
        assert handler._timer is not None
        handler._timer.cancel()
