import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ingest import auto_ingest_queue  # noqa: E402


def _setup_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    (vault / "wiki" / "log.md").write_text("# log\n", encoding="utf-8")
    return vault


def test_happy_path(tmp_path, monkeypatch):
    vault = _setup_vault(tmp_path)
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps({"arxiv_ids": ["2401.04088", "2305.13245"], "options": {"merge": True}}),
        encoding="utf-8",
    )

    received_argv = []

    def fake_main(argv):
        received_argv.extend(argv)
        return 0

    monkeypatch.setattr(auto_ingest_queue.auto_ingest, "main", fake_main)

    rc = auto_ingest_queue.main(
        [
            "--queue",
            str(queue_path),
            "--vault",
            str(vault),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert rc == 0
    assert "--arxiv-ids" in received_argv
    idx = received_argv.index("--arxiv-ids")
    assert "2401.04088" in received_argv[idx + 1]
    assert "2305.13245" in received_argv[idx + 1]
    assert "--merge" in received_argv


def test_missing_queue_file(tmp_path):
    rc = auto_ingest_queue.main(
        [
            "--queue",
            str(tmp_path / "nonexistent.json"),
        ]
    )
    assert rc == 2
