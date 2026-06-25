import json
import sys
import builtins
from types import SimpleNamespace

from scripts.core import build


def test_build_dry_run_keeps_plain_pdf_markdown_default(capsys) -> None:
    assert build.main(["--dry-run"]) == 0

    output = capsys.readouterr().out

    assert f"{sys.executable} -m scripts.core.wiki pdf markdown\n" in output
    assert "--provider hybrid" not in output


def test_build_high_quality_pdf_markdown_profile_uses_hybrid_provider(capsys) -> None:
    assert build.main(["--dry-run", "--pdf-markdown-profile", "high-quality"]) == 0

    output = capsys.readouterr().out

    assert f"{sys.executable} -m scripts.core.wiki pdf markdown --provider hybrid\n" in output


def test_build_pdf_markdown_incremental_reuses_cached_markdown(capsys) -> None:
    assert (
        build.main(
            [
                "--dry-run",
                "--pdf-markdown-profile",
                "high-quality",
                "--pdf-markdown-incremental",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out

    assert (
        f"{sys.executable} -m scripts.core.wiki pdf markdown "
        "--provider hybrid --incremental\n"
        in output
    )


def test_build_pdf_incremental_reuses_pdf_source_layers_and_facts(capsys) -> None:
    assert (
        build.main(
            [
                "--dry-run",
                "--pdf-markdown-profile",
                "high-quality",
                "--pdf-incremental",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out

    assert (
        f"{sys.executable} -m scripts.core.wiki pdf cache "
        "--incremental --linkgraph-meta .graph_index/linkgraph.sqlite.meta.json\n"
        in output
    )
    assert (
        f"{sys.executable} -m scripts.core.wiki pdf markdown "
        "--provider hybrid --incremental --pdf-cache-manifest .graph_index/pdf_text_manifest.jsonl "
        "--linkgraph-meta .graph_index/linkgraph.sqlite.meta.json\n"
        in output
    )
    assert (
        f"{sys.executable} -m scripts.core.wiki pdf facts "
        "--manifest .graph_index/pdf_markdown_manifest.jsonl --incremental\n"
        in output
    )


def test_build_dry_run_json_reports_step_commands(capsys) -> None:
    assert (
        build.main(
            [
                "--dry-run",
                "--json",
                "--pdf-markdown-profile",
                "high-quality",
                "--pdf-incremental",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is True
    pdf_markdown = next(step for step in payload["steps"] if step["label"] == "pdf markdown")
    assert pdf_markdown["cmd"] == [
        sys.executable,
        "-m",
        "scripts.core.wiki",
        "pdf",
        "markdown",
        "--provider",
        "hybrid",
        "--incremental",
        "--pdf-cache-manifest",
        ".graph_index/pdf_text_manifest.jsonl",
        "--linkgraph-meta",
        ".graph_index/linkgraph.sqlite.meta.json",
        "--json",
        "--summary-only",
    ]
    pdf_cache = next(step for step in payload["steps"] if step["label"] == "pdf cache")
    assert pdf_cache["cmd"] == [
        sys.executable,
        "-m",
        "scripts.core.wiki",
        "pdf",
        "cache",
        "--incremental",
        "--linkgraph-meta",
        ".graph_index/linkgraph.sqlite.meta.json",
        "--json",
        "--summary-only",
    ]


def test_build_json_reports_step_timings(monkeypatch, capsys) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        assert kwargs == {"text": True, "capture_output": True}
        return SimpleNamespace(returncode=0, stdout="done\n", stderr="")

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    assert build.main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["dry_run"] is False
    assert payload["elapsed_s"] >= 0
    assert len(payload["steps"]) == len(build.STEPS)
    assert len(calls) == len(build.STEPS)
    assert payload["steps"][0]["returncode"] == 0
    assert payload["steps"][0]["stdout"] == "done\n"
    assert payload["steps"][0]["stderr"] == ""
    assert payload["steps"][0]["elapsed_s"] >= 0


def test_build_json_parses_child_json_reports(monkeypatch, capsys) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        assert kwargs == {"text": True, "capture_output": True}
        if "cache" in cmd:
            stdout = json.dumps({"summary": {"cached": 2, "fast_path": True}, "rows": []})
        else:
            stdout = "plain output\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    assert build.main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    pdf_cache = next(step for step in payload["steps"] if step["label"] == "pdf cache")

    assert pdf_cache["report"] == {"summary": {"cached": 2, "fast_path": True}, "rows": []}


def test_build_non_json_step_logs_are_flushed(monkeypatch) -> None:
    print_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_print(*args: object, **kwargs: object) -> None:
        print_calls.append((args, kwargs))

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(builtins, "print", fake_print)
    monkeypatch.setattr(build.subprocess, "run", fake_run)

    assert build.main([]) == 0

    step_logs = [
        kwargs
        for args, kwargs in print_calls
        if args and isinstance(args[0], str) and args[0].startswith("[build]")
    ]
    assert step_logs
    assert all(kwargs.get("flush") is True for kwargs in step_logs)


def test_build_json_reports_failed_step(monkeypatch, capsys) -> None:
    calls = 0

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=7 if calls == 2 else 0,
            stdout="",
            stderr="failed\n" if calls == 2 else "",
        )

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    assert build.main(["--json"]) == 7

    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is False
    assert payload["steps"][-1]["returncode"] == 7
    assert payload["steps"][-1]["stderr"] == "failed\n"
    assert calls == 2
