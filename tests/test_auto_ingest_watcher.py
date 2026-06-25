import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ingest import auto_ingest_watcher  # noqa: E402


def _make_atom(ids: list[str]) -> bytes:
    entries = ""
    for arxiv_id in ids:
        entries += f"""
  <entry xmlns="http://www.w3.org/2005/Atom">
    <id>http://arxiv.org/abs/{arxiv_id}v1</id>
  </entry>"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
{entries}
</feed>""".encode()


def _fake_urlopen(atom_bytes: bytes):
    def _open(req, timeout=30):
        mock_resp = MagicMock()
        mock_resp.read.return_value = atom_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    return _open


def _ok_completed():
    return SimpleNamespace(stdout="ingested: 1\n", stderr="", returncode=0)


def test_state_file_filters_seen(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"seen_ids": ["2401.00001"]}), encoding="utf-8")

    atom_bytes = _make_atom(["2401.00001", "2401.00002"])
    monkeypatch.setattr(
        "scripts.ingest.auto_ingest_watcher.urllib.request.urlopen", _fake_urlopen(atom_bytes)
    )

    ingested_ids = []

    def fake_run(cmd, **kwargs):
        # capture --arxiv-ids value
        idx = cmd.index("--arxiv-ids")
        ingested_ids.append(cmd[idx + 1])
        return _ok_completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = SimpleNamespace(
        query="cat:cond-mat.soft",
        vault=vault,
        cache_dir=tmp_path / "cache",
        max_per_poll=10,
        state_file=state_file,
        merge=False,
        dry_run=False,
    )
    auto_ingest_watcher._poll_once(args, {"2401.00001"})

    assert ingested_ids == ["2401.00002"]


def test_once_mode_exits(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    (tmp_path / "cache").mkdir(parents=True)

    atom_bytes = _make_atom(["2401.00010"])
    monkeypatch.setattr(
        "scripts.ingest.auto_ingest_watcher.urllib.request.urlopen", _fake_urlopen(atom_bytes)
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_completed())

    rc = auto_ingest_watcher.main(
        [
            "--query",
            "cat:cond-mat.soft",
            "--vault",
            str(vault),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--state-file",
            str(tmp_path / "state.json"),
            "--once",
        ]
    )
    assert rc == 0


def test_max_per_poll_caps(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    state_file = tmp_path / "state.json"

    ids = [f"2401.0000{i}" for i in range(5)]
    atom_bytes = _make_atom(ids)
    monkeypatch.setattr(
        "scripts.ingest.auto_ingest_watcher.urllib.request.urlopen", _fake_urlopen(atom_bytes)
    )

    ingested_ids = []

    def fake_run(cmd, **kwargs):
        idx = cmd.index("--arxiv-ids")
        ingested_ids.append(cmd[idx + 1])
        return _ok_completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = SimpleNamespace(
        query="cat:cond-mat.soft",
        vault=vault,
        cache_dir=tmp_path / "cache",
        max_per_poll=1,
        state_file=state_file,
        merge=False,
        dry_run=False,
    )
    auto_ingest_watcher._poll_once(args, set())

    assert len(ingested_ids) == 1


def test_state_file_persists(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    state_file = tmp_path / "state.json"

    ids = ["2401.00020", "2401.00021"]
    atom_bytes = _make_atom(ids)
    monkeypatch.setattr(
        "scripts.ingest.auto_ingest_watcher.urllib.request.urlopen", _fake_urlopen(atom_bytes)
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_completed())

    args = SimpleNamespace(
        query="cat:cond-mat.soft",
        vault=vault,
        cache_dir=tmp_path / "cache",
        max_per_poll=10,
        state_file=state_file,
        merge=False,
        dry_run=False,
    )
    auto_ingest_watcher._poll_once(args, set())

    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert set(saved["seen_ids"]) == {"2401.00020", "2401.00021"}


def test_subprocess_runs_from_repo_root(tmp_path, monkeypatch):
    """Regression guard: subprocess MUST run from REPO_ROOT, not vault.
    Phase 4 fix to auto_ingest.py was not propagated to the watcher; if the
    user passes --vault pointing outside the repo, `python -m scripts.ingest.auto_ingest`
    would fail with No module named scripts.
    """
    vault = tmp_path / "external_vault"
    (vault / "wiki").mkdir(parents=True)
    state_file = tmp_path / "state.json"
    atom_bytes = _make_atom(["2401.99999"])
    monkeypatch.setattr(
        "scripts.ingest.auto_ingest_watcher.urllib.request.urlopen", _fake_urlopen(atom_bytes)
    )

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        return _ok_completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = SimpleNamespace(
        query="cat:x",
        vault=vault,
        cache_dir=tmp_path / "cache",
        max_per_poll=10,
        state_file=state_file,
        merge=False,
        dry_run=False,
    )
    auto_ingest_watcher._poll_once(args, set())

    expected_repo_root = Path(auto_ingest_watcher.__file__).resolve().parent.parent
    assert Path(captured["cwd"]).resolve() == expected_repo_root, (
        f"watcher subprocess ran in {captured['cwd']}, not REPO_ROOT={expected_repo_root}"
    )


def test_corrupt_state_file_moved_aside(tmp_path, monkeypatch, caplog):
    """Corrupt state JSON must be moved to .corrupt rather than silently
    treated as empty (which would re-ingest every ID in the arXiv window).
    """
    state_file = tmp_path / "state.json"
    state_file.write_text("{not valid json", encoding="utf-8")

    import logging

    caplog.set_level(logging.ERROR)
    result = auto_ingest_watcher._load_state(state_file)

    assert result == set()
    assert not state_file.exists()
    assert state_file.with_suffix(".json.corrupt").exists()


def test_malformed_atom_does_not_crash_poll(tmp_path, monkeypatch):
    """A malformed Atom response otherwise kills the daemon between polls."""
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(
        "scripts.ingest.auto_ingest_watcher.urllib.request.urlopen",
        _fake_urlopen(b"this is not xml at all <<<"),
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_completed())

    args = SimpleNamespace(
        query="cat:x",
        vault=vault,
        cache_dir=tmp_path / "cache",
        max_per_poll=10,
        state_file=state_file,
        merge=False,
        dry_run=False,
    )
    # Must not raise.
    result = auto_ingest_watcher._poll_once(args, set())
    assert result == set()
