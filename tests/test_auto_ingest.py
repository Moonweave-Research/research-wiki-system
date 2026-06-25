import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ingest import auto_ingest  # noqa: E402


def _setup_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    (vault / "wiki" / "log.md").write_text("# log\n", encoding="utf-8")
    return vault


def _ok_completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _ingest_json_line(arxiv_id: str, action: str = "created") -> str:
    return json.dumps(
        {
            "arxiv_id": arxiv_id,
            "action": action,
            "note_path": f"wiki/research/arxiv_{arxiv_id.replace('.', '_')}.md",
            "pdf_path": "",
            "error": "",
        }
    )


def test_chain_success(tmp_path, monkeypatch, capsys):
    vault = _setup_vault(tmp_path)
    graph_index = vault / ".graph_index"
    graph_index.mkdir()
    # Markdown manifest pretends both papers cached after text + Markdown stages.
    (graph_index / "pdf_markdown_manifest.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"paper_id": "paper:arxiv_2401_04088", "cached": True}),
                json.dumps({"paper_id": "paper:arxiv_2305_13245", "cached": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (graph_index / "pdf_facts.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"paper_id": "paper:arxiv_2401_04088", "fact": "x"}),
                json.dumps({"paper_id": "paper:arxiv_2401_04088", "fact": "y"}),
                json.dumps({"paper_id": "paper:arxiv_2305_13245", "fact": "z"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "arxiv_ingest" in cmd[2]:
            stdout = "\n".join([_ingest_json_line("2401.04088"), _ingest_json_line("2305.13245")])
            return _ok_completed(stdout=stdout)
        return _ok_completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = auto_ingest.main(
        [
            "--arxiv-ids",
            "2401.04088,2305.13245",
            "--vault",
            str(vault),
            "--graph-index",
            str(graph_index),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 0
    # arxiv_ingest + wiki_pdf_cache + wiki_pdf_markdown + wiki_pdf_fact_extractor
    assert len(calls) == 4
    assert "wiki_pdf_markdown" in calls[2][2]
    assert str(graph_index / "pdf_markdown_manifest.jsonl") in calls[3]
    out = capsys.readouterr().out
    assert "ingested: 2" in out
    assert "pdf_cached: 2" in out
    assert "facts_extracted: 3" in out


def test_chain_stage1_full_fail(tmp_path, monkeypatch):
    vault = _setup_vault(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "arxiv_ingest" in cmd[2]:
            stdout = _ingest_json_line("2401.04088", action="failed")
            return _ok_completed(stdout=stdout)
        return _ok_completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = auto_ingest.main(
        [
            "--arxiv-ids",
            "2401.04088",
            "--vault",
            str(vault),
            "--graph-index",
            str(vault / ".graph_index"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 2
    # Only stage 1 should have run.
    assert len(calls) == 1


def test_chain_stage1_partial(tmp_path, monkeypatch):
    vault = _setup_vault(tmp_path)
    graph_index = vault / ".graph_index"
    graph_index.mkdir()
    (graph_index / "pdf_markdown_manifest.jsonl").write_text(
        json.dumps({"paper_id": "paper:arxiv_2401_04088", "cached": True}) + "\n",
        encoding="utf-8",
    )
    (graph_index / "pdf_facts.jsonl").write_text("", encoding="utf-8")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "arxiv_ingest" in cmd[2]:
            stdout = "\n".join(
                [_ingest_json_line("2401.04088"), _ingest_json_line("9999.99999", action="failed")]
            )
            return _ok_completed(stdout=stdout)
        return _ok_completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = auto_ingest.main(
        [
            "--arxiv-ids",
            "2401.04088,9999.99999",
            "--vault",
            str(vault),
            "--graph-index",
            str(graph_index),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 0
    # Stage 2 and 3 still ran for the surviving ID.
    assert len(calls) == 4


def test_skip_facts(tmp_path, monkeypatch):
    vault = _setup_vault(tmp_path)
    graph_index = vault / ".graph_index"
    graph_index.mkdir()
    (graph_index / "pdf_markdown_manifest.jsonl").write_text(
        json.dumps({"paper_id": "paper:arxiv_2401_04088", "cached": True}) + "\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "arxiv_ingest" in cmd[2]:
            return _ok_completed(stdout=_ingest_json_line("2401.04088"))
        return _ok_completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = auto_ingest.main(
        [
            "--arxiv-ids",
            "2401.04088",
            "--vault",
            str(vault),
            "--graph-index",
            str(graph_index),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--skip-facts",
        ]
    )
    assert rc == 0
    # Stage 1 plus text + Markdown cache; no fact extractor.
    assert len(calls) == 3
    assert any("wiki_pdf_markdown" in part for cmd in calls for part in cmd)
    assert not any("wiki_pdf_fact_extractor" in part for cmd in calls for part in cmd)


def test_dry_run(tmp_path, monkeypatch):
    vault = _setup_vault(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "arxiv_ingest" in cmd[2]:
            return _ok_completed(stdout=_ingest_json_line("2401.04088"))
        return _ok_completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    log_path = vault / "wiki" / "log.md"
    log_before = log_path.read_text(encoding="utf-8")

    rc = auto_ingest.main(
        [
            "--arxiv-ids",
            "2401.04088",
            "--vault",
            str(vault),
            "--graph-index",
            str(vault / ".graph_index"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--dry-run",
        ]
    )
    assert rc == 0
    # Only stage 1 in dry-run mode.
    assert len(calls) == 1
    assert "--dry-run" in calls[0]
    # Log should be unchanged.
    assert log_path.read_text(encoding="utf-8") == log_before


def test_log_entry_format(tmp_path, monkeypatch):
    vault = _setup_vault(tmp_path)
    graph_index = vault / ".graph_index"
    graph_index.mkdir()
    (graph_index / "pdf_markdown_manifest.jsonl").write_text(
        json.dumps({"paper_id": "paper:arxiv_2401_04088", "cached": True}) + "\n",
        encoding="utf-8",
    )
    (graph_index / "pdf_facts.jsonl").write_text(
        json.dumps({"paper_id": "paper:arxiv_2401_04088", "fact": "x"}) + "\n",
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if "arxiv_ingest" in cmd[2]:
            return _ok_completed(stdout=_ingest_json_line("2401.04088"))
        return _ok_completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = auto_ingest.main(
        [
            "--arxiv-ids",
            "2401.04088",
            "--vault",
            str(vault),
            "--graph-index",
            str(graph_index),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 0
    log_text = (vault / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "auto_ingest |" in log_text
    assert "arxiv 1 ingested" in log_text
    assert "1 cached" in log_text
    assert "1 facts" in log_text
    assert "paper:arxiv_2401_04088" in log_text
    # Single new entry, not duplicated.
    assert log_text.count("auto_ingest |") == 1


def test_stage1_subprocess_crash_aborts_chain(tmp_path, monkeypatch):
    """If arxiv_ingest crashes after emitting only partial JSON, the
    orchestrator must NOT continue to Stage 2/3 with that partial data —
    it must report stage1_crashed and exit 2.
    """
    vault = _setup_vault(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "arxiv_ingest" in cmd[2]:
            # Two IDs requested, only one JSON line emitted, non-zero exit.
            return SimpleNamespace(
                stdout=_ingest_json_line("2401.04088"),
                stderr="Traceback (most recent call last)\nValueError: boom",
                returncode=1,
            )
        return _ok_completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = auto_ingest.main(
        [
            "--arxiv-ids",
            "2401.04088,2305.13245",
            "--vault",
            str(vault),
            "--graph-index",
            str(vault / ".graph_index"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 2
    # Only Stage 1 should have been invoked — chain aborts before cache/extractor.
    assert len(calls) == 1
