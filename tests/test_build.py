import sys

from scripts.core import build


class _FakeCompleted:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def test_build_dispatches_foundation_pipeline(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd):
        calls.append(cmd)
        return _FakeCompleted(0)

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    assert build.main([]) == 0
    # Each step runs as `python -m <module> <args>`. The embed step regressed
    # once to bare ["facts"] (CLI rejects with exit 2), so the exact sequence
    # is locked here.
    for cmd in calls:
        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
    dispatched = [(cmd[2], cmd[3:]) for cmd in calls]
    assert dispatched == [
        ("scripts.core.linkgraph", ["build"]),
        ("scripts.ingest.concept_backfill", ["--apply"]),
        ("scripts.ingest.concept_split", ["--apply"]),
        ("scripts.core.linkgraph", ["build"]),
        ("scripts.core.wiki", ["pdf", "cache"]),
        ("scripts.core.wiki", ["pdf", "markdown"]),
        (
            "scripts.core.wiki",
            ["pdf", "facts", "--manifest", ".graph_index/pdf_markdown_manifest.jsonl"],
        ),
        ("scripts.core.wiki", ["facts", "embed"]),
        ("scripts.search.related_notes", ["build"]),
        ("scripts.search.graph_communities", ["build"]),
    ]


def test_build_dry_run_executes_nothing(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(build.subprocess, "run", lambda cmd: calls.append(cmd))

    assert build.main(["--dry-run"]) == 0
    assert calls == []


def test_build_stops_on_first_failure(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd):
        calls.append(cmd)
        return _FakeCompleted(3)

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    assert build.main([]) == 3
    assert len(calls) == 1
