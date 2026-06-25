import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ingest.wiki_llm_fact_augmenter import extract_triples_for_paper  # noqa: E402
from legacy.graphrag.wiki_semantic_schema import SemanticTriple, _stable_triple_id  # noqa: E402


def _mock_provider(payload):
    """Patch provider_call to return the given payload string."""
    return patch("scripts.ingest.wiki_llm_fact_augmenter.provider_call", return_value=payload)


def test_extract_triples_parses_response():
    payload = json.dumps(
        [
            {
                "subject": "PEDOT:PSS film",
                "predicate": "exhibits",
                "object": "1000 S/cm conductivity",
                "evidence": "The PEDOT:PSS film exhibits 1000 S/cm.",
                "confidence": 0.9,
            },
            {
                "subject": "azobenzene",
                "predicate": "isomerizes under",
                "object": "365 nm UV",
                "evidence": "Azobenzene isomerizes under 365 nm UV.",
                "confidence": 0.95,
            },
        ]
    )
    with _mock_provider(payload):
        triples = extract_triples_for_paper(
            paper_id="paper:test",
            paper_text="some paper text",
            model_flag="sonnet",
        )
    assert len(triples) == 2
    assert all(isinstance(t, SemanticTriple) for t in triples)
    assert triples[0].subject == "PEDOT:PSS film"
    assert triples[1].predicate == "isomerizes under"


def test_extract_triples_empty_array():
    with _mock_provider("[]"):
        triples = extract_triples_for_paper(paper_id="paper:test", paper_text="text")
    assert triples == []


def test_extract_triples_strips_markdown_fences():
    payload = (
        "```json\n"
        '[{"subject": "PDMS", "predicate": "is", "object": "elastomer", '
        '"evidence": "PDMS is an elastomer.", "confidence": 0.8}]\n'
        "```"
    )
    with _mock_provider(payload):
        triples = extract_triples_for_paper(paper_id="paper:test", paper_text="text")
    assert len(triples) == 1
    assert triples[0].subject == "PDMS"


def test_triples_dedupe_by_lowercase_spo():
    # Existing triple in corpus
    existing = [
        SemanticTriple(
            paper_id="paper:other",
            triple_id=_stable_triple_id("paper:other", "PEDOT:PSS", "exhibits", "1000 S/cm"),
            subject="PEDOT:PSS",
            predicate="exhibits",
            object="1000 S/cm",
            evidence="Some other paper.",
            confidence=0.85,
        )
    ]
    # New LLM response with same SPO in different case → must be filtered
    payload = json.dumps(
        [
            {
                "subject": "pedot:pss",
                "predicate": "EXHIBITS",
                "object": "1000 s/cm",
                "evidence": "Same triple lowercased.",
                "confidence": 0.9,
            },
            {
                "subject": "graphene",
                "predicate": "transports",
                "object": "charge carriers",
                "evidence": "Graphene transports charge carriers.",
                "confidence": 0.9,
            },
        ]
    )
    with _mock_provider(payload):
        new = extract_triples_for_paper(
            paper_id="paper:test", paper_text="text", existing_triples=existing
        )
    # PEDOT:PSS filtered, graphene kept
    assert len(new) == 1
    assert new[0].subject == "graphene"


def test_extract_triples_drops_malformed_items():
    payload = json.dumps(
        [
            {
                "subject": "ok",
                "predicate": "is",
                "object": "fine",
                "evidence": "ev",
                "confidence": 0.7,
            },
            {"subject": "missing_evidence", "predicate": "x", "object": "y"},
            "not a dict",
            {"subject": "", "predicate": "p", "object": "o", "evidence": "e", "confidence": 0.5},
        ]
    )
    with _mock_provider(payload):
        triples = extract_triples_for_paper(paper_id="paper:test", paper_text="text")
    # Only the first item is valid; rest dropped silently.
    assert len(triples) == 1
    assert triples[0].subject == "ok"


def test_extract_triples_default_confidence_when_missing():
    payload = json.dumps([{"subject": "s", "predicate": "p", "object": "o", "evidence": "ev"}])
    with _mock_provider(payload):
        triples = extract_triples_for_paper(paper_id="paper:test", paper_text="text")
    assert len(triples) == 1
    assert triples[0].confidence == 0.7


def test_main_triples_resume_extracts_when_facts_done_but_triples_empty(tmp_path, monkeypatch):
    """Regression for the resume bug: when facts have already been extracted
    (all papers in done_facts) but --triples was never run before, the script
    must still call the triple extractor for each paper. Previously
    done_papers from facts file would short-circuit the entire run.
    """
    from scripts.ingest.wiki_llm_fact_augmenter import main as augmenter_main
    from legacy.graphrag.wiki_semantic_schema import read_triples_jsonl

    graph_index = tmp_path / ".graph_index"
    graph_index.mkdir()
    # facts.jsonl: corpus of facts for 2 papers
    facts_path = graph_index / "facts.jsonl"
    facts_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "paper:p1",
                        "fact_id": "fact:p1:1",
                        "fact_type": "Claim",
                        "text": "claim 1",
                        "evidence": "ev",
                        "confidence": 0.7,
                        "targets": ["concept:Polymer"],
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "paper_id": "paper:p2",
                        "fact_id": "fact:p2:1",
                        "fact_type": "Claim",
                        "text": "claim 2",
                        "evidence": "ev",
                        "confidence": 0.7,
                        "targets": ["concept:Polymer"],
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # Pretend fact augmentation already finished for both papers.
    (graph_index / "facts_llm_augment.jsonl").write_text(
        facts_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    # No triples file yet — first time --triples is enabled.

    triple_call_count = {"n": 0}

    def fake_provider(prompt, system=None, model=None, timeout_s=None):
        if "subject" in (system or "").lower():
            # Triple extractor call
            triple_call_count["n"] += 1
            return json.dumps(
                [
                    {
                        "subject": f"entity{triple_call_count['n']}",
                        "predicate": "is",
                        "object": "thing",
                        "evidence": "ev",
                        "confidence": 0.8,
                    }
                ]
            )
        # Facts extractor — should NOT be called since done_facts is full
        return "[]"

    monkeypatch.setattr("scripts.ingest.wiki_llm_fact_augmenter.provider_call", fake_provider)

    augmenter_main(
        [
            "--graph-index",
            str(graph_index),
            "--triples",
            "--yes-cost-confirm",
        ]
    )

    triples_out = graph_index / "triples_llm_augment.jsonl"
    assert triples_out.exists()
    triples = read_triples_jsonl(triples_out)
    # Both papers got triples even though facts were "done".
    assert {t.paper_id for t in triples} == {"paper:p1", "paper:p2"}
    assert triple_call_count["n"] == 2


def test_cost_guard_blocks_large_triples_run_without_confirm(tmp_path, monkeypatch, capsys):
    """--triples on > 50 papers must require --yes-cost-confirm."""
    from scripts.ingest.wiki_llm_fact_augmenter import main as augmenter_main

    graph_index = tmp_path / ".graph_index"
    graph_index.mkdir()
    # Build a facts.jsonl with 60 papers (above the 50-paper threshold).
    facts_rows = []
    for i in range(60):
        facts_rows.append(
            json.dumps(
                {
                    "paper_id": f"paper:p{i}",
                    "fact_id": f"fact:p{i}:1",
                    "fact_type": "Claim",
                    "text": "claim",
                    "evidence": "ev",
                    "confidence": 0.7,
                    "targets": ["concept:Polymer"],
                },
                sort_keys=True,
            )
        )
    (graph_index / "facts.jsonl").write_text("\n".join(facts_rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.ingest.wiki_llm_fact_augmenter.provider_call",
        lambda *a, **k: pytest.fail("provider_call should not be invoked when cost guard fires"),
    )

    augmenter_main(["--graph-index", str(graph_index), "--triples"])

    err = capsys.readouterr().err
    assert "--yes-cost-confirm" in err
    assert not (graph_index / "triples_llm_augment.jsonl").exists()


import pytest  # noqa: E402  (used by test_cost_guard_blocks_large_triples_run_without_confirm)
