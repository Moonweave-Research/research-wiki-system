import importlib
import json
from pathlib import Path

import pytest

from scripts.ingest import wiki_llm_fact_augmenter as mod
from scripts.ingest.wiki_llm_fact_augmenter import build_prompt

FIX = Path(__file__).parent / "fixtures" / "augmenter"


def test_module_imports():
    mod = importlib.import_module("scripts.ingest.wiki_llm_fact_augmenter")
    assert hasattr(mod, "augment_facts_for_paper")
    assert hasattr(mod, "DEFAULT_MODEL_FLAG")
    assert mod.DEFAULT_MODEL_FLAG == "sonnet"


def test_build_prompt_matches_golden():
    sample = json.loads((FIX / "sample_paper_input.json").read_text())
    expected = (FIX / "sample_expected_prompt.txt").read_text()
    out = build_prompt(
        paper_id=sample["paper_id"],
        existing_facts=sample["existing_facts"],
        paper_text=sample["paper_text"],
    )
    assert out == expected


def test_call_claude_subprocess_returns_facts(monkeypatch):
    inner = '[{"fact_type":"Metric","text":"tensile 12 MPa","evidence":"...","targets":["metric:tensile_strength"]}]'
    captured = {}

    def fake_provider_call(prompt, *, system=None, model=None, timeout_s=120):
        captured["prompt"] = prompt
        captured["system"] = system
        captured["model"] = model
        return inner

    monkeypatch.setattr(mod, "provider_call", fake_provider_call)

    out = mod._call_claude_subprocess(user_prompt="dummy", model_flag=mod.DEFAULT_MODEL_FLAG)
    assert isinstance(out, list)
    assert out[0]["fact_type"] == "Metric"
    assert captured["model"] == mod.DEFAULT_MODEL_FLAG
    assert captured["system"] == mod._EXTRACTOR_SYSTEM
    assert captured["prompt"] == "dummy"


def test_augment_facts_for_paper_dedupes_against_existing(monkeypatch):
    from scripts.ingest.wiki_llm_fact_augmenter import augment_facts_for_paper
    from legacy.graphrag.wiki_semantic_schema import SemanticFact

    existing = [
        SemanticFact(
            paper_id="paper:foo2024",
            fact_id="fact:foo2024:claim:0",
            fact_type="Metric",
            text="tensile strength 12 MPa",
            evidence="The composite achieved a tensile strength of 12 MPa.",
            confidence=0.9,
            targets=["metric:tensile_strength"],
        )
    ]

    def fake_call(user_prompt, model_flag="sonnet", timeout_s=120):
        return [
            {
                "fact_type": "Metric",
                "text": "tensile strength 12 MPa",
                "evidence": "The composite achieved a tensile strength of 12 MPa.",
                "targets": ["metric:tensile_strength"],
            },
            {
                "fact_type": "Material",
                "text": "PDMS elastomer matrix",
                "evidence": "PDMS as the elastomer matrix",
                "targets": ["material:pdms"],
            },
        ]

    monkeypatch.setattr("scripts.ingest.wiki_llm_fact_augmenter._call_claude_subprocess", fake_call)

    new_facts = augment_facts_for_paper(
        paper_id="paper:foo2024",
        existing_facts=existing,
        paper_text="dummy",
    )
    assert len(new_facts) == 1
    assert new_facts[0].fact_type == "Material"
    assert new_facts[0].fact_id.startswith("fact:foo2024:llm:")


def test_augment_facts_for_paper_raises_when_provider_fails(monkeypatch):
    from scripts.ingest.wiki_llm_fact_augmenter import ClaudeCallError, augment_facts_for_paper

    def fake_call(user_prompt, model_flag="sonnet", timeout_s=120):
        raise ClaudeCallError("provider unavailable")

    monkeypatch.setattr("scripts.ingest.wiki_llm_fact_augmenter._call_claude_subprocess", fake_call)

    with pytest.raises(ClaudeCallError, match="provider unavailable"):
        augment_facts_for_paper(
            paper_id="paper:foo2024",
            existing_facts=[],
            paper_text=(
                "The PDMS elastomer actuator reached 12 MPa stress and 30% strain. "
                "The device was demonstrated as a wearable strain sensor."
            ),
        )


def test_main_retries_provider_failures_before_success(monkeypatch, tmp_path, capsys):
    from scripts.ingest.wiki_llm_fact_augmenter import ClaudeCallError

    calls = 0

    def flaky_call(user_prompt, model_flag="sonnet", timeout_s=120):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ClaudeCallError("provider temporarily unavailable")
        return [
            {
                "fact_type": "Material",
                "text": "PDMS elastomer matrix",
                "evidence": "PDMS as the elastomer matrix",
                "targets": ["material:pdms"],
            }
        ]

    monkeypatch.setattr("scripts.ingest.wiki_llm_fact_augmenter._call_claude_subprocess", flaky_call)

    graph_index = tmp_path / ".graph_index"
    graph_index.mkdir()
    (graph_index / "facts.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "paper:foo2024",
                "fact_id": "fact:foo2024:c0",
                "fact_type": "Claim",
                "text": "x",
                "evidence": "PDMS as the elastomer matrix.",
                "confidence": 0.9,
                "targets": ["concept:Foo"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paper_ids_file = tmp_path / "papers.txt"
    paper_ids_file.write_text("paper:foo2024\n", encoding="utf-8")

    mod.main(
        [
            "--graph-index",
            str(graph_index),
            "--paper-ids-file",
            str(paper_ids_file),
            "--dry-run",
        ]
    )

    err = capsys.readouterr().err
    assert calls == 3
    assert "provider fallback -> heuristic" not in err
    assert "paper:foo2024: +1 facts" in err


def test_main_reports_provider_failures_without_heuristic_fallback(monkeypatch, tmp_path, capsys):
    from scripts.ingest.wiki_llm_fact_augmenter import ClaudeCallError

    def fake_call(user_prompt, model_flag="sonnet", timeout_s=120):
        raise ClaudeCallError("provider unavailable")

    monkeypatch.setattr("scripts.ingest.wiki_llm_fact_augmenter._call_claude_subprocess", fake_call)

    graph_index = tmp_path / ".graph_index"
    graph_index.mkdir()
    (graph_index / "facts.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "paper:foo2024",
                "fact_id": "fact:foo2024:c0",
                "fact_type": "Claim",
                "text": "PDMS actuator",
                "evidence": "PDMS actuator measured 12 MPa stress for a wearable sensor.",
                "confidence": 0.9,
                "targets": ["concept:Foo"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paper_ids_file = tmp_path / "papers.txt"
    paper_ids_file.write_text("paper:foo2024\n", encoding="utf-8")

    mod.main(
        [
            "--graph-index",
            str(graph_index),
            "--paper-ids-file",
            str(paper_ids_file),
            "--dry-run",
        ]
    )

    err = capsys.readouterr().err
    assert "paper:foo2024: ERROR ClaudeCallError: provider unavailable" in err
    assert "provider fallback -> heuristic" not in err
    assert "provider fallbacks" not in err


def test_augment_facts_for_paper_can_use_heuristic_mode_without_provider(monkeypatch):
    def fail_if_called(user_prompt, model_flag="sonnet", timeout_s=120):
        raise AssertionError("provider should not be called")

    monkeypatch.setattr("scripts.ingest.wiki_llm_fact_augmenter._call_claude_subprocess", fail_if_called)

    new_facts = mod.augment_facts_for_paper(
        paper_id="paper:foo2024",
        existing_facts=[],
        paper_text="PDMS actuator measured 12 MPa stress for a wearable sensor.",
        model_flag="heuristic",
    )

    assert {fact.fact_type for fact in new_facts} == {"Metric", "Material", "Application"}


def test_main_dry_run_with_mocked_subprocess(monkeypatch, tmp_path):
    """End-to-end main() smoke with mocked claude CLI — no real subprocess calls."""
    import json
    from scripts.ingest import wiki_llm_fact_augmenter as mod

    # Stub _call_claude_subprocess so no real CLI runs.
    def fake_call(user_prompt, model_flag="sonnet", timeout_s=120):
        return [
            {
                "fact_type": "Material",
                "text": "PDMS elastomer matrix",
                "evidence": "PDMS as the elastomer matrix",
                "targets": ["material:pdms"],
            }
        ]

    monkeypatch.setattr("scripts.ingest.wiki_llm_fact_augmenter._call_claude_subprocess", fake_call)

    # Build a tiny facts.jsonl in tmp_path/.graph_index/
    graph_index = tmp_path / ".graph_index"
    graph_index.mkdir()
    facts_path = graph_index / "facts.jsonl"
    facts_path.write_text(
        json.dumps(
            {
                "paper_id": "paper:foo2024",
                "fact_id": "fact:foo2024:c0",
                "fact_type": "Claim",
                "text": "x",
                "evidence": "y",
                "confidence": 0.9,
                "targets": ["concept:Foo"],
            },
            sort_keys=True,
        )
        + "\n"
    )

    paper_ids_file = tmp_path / "papers.txt"
    paper_ids_file.write_text("paper:foo2024\n")

    # Dry-run path — should NOT write the output file
    mod.main(
        [
            "--graph-index",
            str(graph_index),
            "--paper-ids-file",
            str(paper_ids_file),
            "--dry-run",
        ]
    )
    assert not (graph_index / "facts_llm_augment.jsonl").exists()

    # Non-dry-run path — should write 1 fact
    mod.main(
        [
            "--graph-index",
            str(graph_index),
            "--paper-ids-file",
            str(paper_ids_file),
        ]
    )
    out_path = graph_index / "facts_llm_augment.jsonl"
    assert out_path.exists()
    written = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    assert len(written) == 1
    assert written[0]["fact_type"] == "Material"
    assert written[0]["paper_id"] == "paper:foo2024"
    assert written[0]["fact_id"].startswith("fact:foo2024:llm:")


def test_load_paper_text_reads_from_pdf_cache(tmp_path):
    from scripts.ingest.wiki_llm_fact_augmenter import _load_paper_text

    graph_index = tmp_path / ".graph_index"
    pdf_text_dir = graph_index / "pdf_text"
    pdf_text_dir.mkdir(parents=True)

    # File present — should return content (truncated to 8000 chars)
    body = "x" * 9000
    (pdf_text_dir / "foo2024.txt").write_text(body, encoding="utf-8")
    out = _load_paper_text("paper:foo2024", graph_index)
    assert len(out) == 8000
    assert out == "x" * 8000

    # File missing — should return empty string
    out_missing = _load_paper_text("paper:bar2024", graph_index)
    assert out_missing == ""


class TestParseJsonArray:
    """_parse_json_array tolerates LLM response noise that broke 77 papers on 5/25."""

    def test_direct_array(self):
        from scripts.ingest.wiki_llm_fact_augmenter import _parse_json_array

        assert _parse_json_array('[{"a": 1}]', context="t") == [{"a": 1}]

    def test_empty_array(self):
        from scripts.ingest.wiki_llm_fact_augmenter import _parse_json_array

        assert _parse_json_array("[]", context="t") == []

    def test_fenced(self):
        from scripts.ingest.wiki_llm_fact_augmenter import _parse_json_array

        text = '```json\n[{"a": 1}]\n```'
        assert _parse_json_array(text, context="t") == [{"a": 1}]

    def test_double_array_concatenated(self):
        # Real failure mode from 5/25 (bayatLargeHybridHigh, bukowczanMolecular...).
        from scripts.ingest.wiki_llm_fact_augmenter import _parse_json_array

        assert _parse_json_array("[]\n[]", context="t") == []

    def test_leading_commentary(self):
        from scripts.ingest.wiki_llm_fact_augmenter import _parse_json_array

        text = 'Here is the result:\n[{"fact_type": "Material", "text": "PDMS"}]'
        assert _parse_json_array(text, context="t") == [{"fact_type": "Material", "text": "PDMS"}]

    def test_bracket_inside_string(self):
        from scripts.ingest.wiki_llm_fact_augmenter import _parse_json_array

        text = '[{"text": "array [1, 2] in evidence"}]'
        assert _parse_json_array(text, context="t") == [{"text": "array [1, 2] in evidence"}]

    def test_unterminated_array_raises(self):
        from scripts.ingest.wiki_llm_fact_augmenter import _parse_json_array, ClaudeCallError

        with pytest.raises(ClaudeCallError, match="no JSON array found|bracket-scoped"):
            _parse_json_array('[{"a": 1', context="t")

    def test_non_array_raises(self):
        from scripts.ingest.wiki_llm_fact_augmenter import _parse_json_array, ClaudeCallError

        with pytest.raises(ClaudeCallError, match="expected JSON array"):
            _parse_json_array('{"not": "array"}', context="t")
