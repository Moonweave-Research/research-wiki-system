import json
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture
def gold_md(tmp_path: Path) -> Path:
    body = dedent("""
        # baseline

        ## 20 candidate papers (≥10 facts...)

        - `paper:foo2024`
        - `paper:bar2024`
        - `paper:bar2024a`

        ## Hand-labeling worksheet

        | # | paper | fact_type_now | text | gold_label | new_target_id | source |
        |---|-------|---------------|------|------------|---------------|--------|
        | 1 | `foo2024` | Claim | tensile strength of the composite reached 12 MPa under standard testing | **Metric** | `metric:tensile_strength` | agreed |
        | 2 | `foo2024` | Claim | unrelated mention | **Stay-Claim** | `—` | agreed |
        | 3 | `bar2024` | Claim | PDMS used as the elastomer matrix substrate | **Material** | `material:pdms` | agreed |
    """).strip()
    p = tmp_path / "gold.md"
    p.write_text(body, encoding="utf-8")
    return p


def test_parse_candidate_ids(gold_md: Path) -> None:
    from eval.wiki_eval_augment import parse_candidate_ids

    out = parse_candidate_ids(gold_md.read_text())
    assert out == ["foo2024", "bar2024", "bar2024a"]


def test_parse_gold_rows(gold_md: Path) -> None:
    from eval.wiki_eval_augment import parse_gold_rows

    rows = parse_gold_rows(gold_md.read_text())
    assert len(rows) == 3
    assert rows[0]["gold_label"] == "Metric"
    assert rows[0]["new_target_id"] == "metric:tensile_strength"
    assert rows[1]["new_target_id"] is None
    assert rows[1]["gold_label"] == "Stay-Claim"


def test_expand_truncated_handles_ambiguity() -> None:
    from eval.wiki_eval_augment import expand_truncated

    out = expand_truncated("bar2024", ["foo2024", "bar2024", "bar2024a"])
    assert sorted(out) == ["paper:bar2024", "paper:bar2024a"]


def test_evaluate_recall_via_target_token_match(gold_md: Path, tmp_path: Path) -> None:
    """Augment emits Metric and Material facts whose targets share tokens with gold's
    new_target_id (e.g. gold `metric:tensile_strength` matches augment
    `metric:tensile_strength_12_mpa` because both contain 'tensile' + 'strength')."""
    from eval.wiki_eval_augment import evaluate

    aug = tmp_path / "facts_llm_augment.jsonl"
    aug.write_text(
        json.dumps(
            {
                "paper_id": "paper:foo2024",
                "fact_id": "fact:foo2024:llm:1",
                "fact_type": "Metric",
                "text": "tensile strength 12 MPa",
                "evidence": "irrelevant — eval is target-token based",
                "confidence": 0.7,
                "targets": ["metric:tensile_strength_12_mpa"],
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "paper_id": "paper:bar2024",
                "fact_id": "fact:bar2024:llm:1",
                "fact_type": "Material",
                "text": "PDMS elastomer matrix",
                "evidence": "irrelevant — eval is target-token based",
                "confidence": 0.7,
                "targets": ["material:pdms_elastomer"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    out = evaluate(gold_md, aug)
    assert out["metric"]["tp"] == 1
    assert out["metric"]["fn"] == 0
    assert out["metric"]["recall"] == 1.0
    assert out["material"]["tp"] == 1
    assert out["material"]["recall"] == 1.0
    assert out["application"]["tp"] == 0
    assert "precision" not in out["metric"]  # precision intentionally not auto-computed


def test_target_tokens_helper() -> None:
    from eval.wiki_eval_augment import _target_tokens

    assert _target_tokens("metric:tensile_strength") == {"tensile", "strength"}
    assert _target_tokens("material:pdms") == {"pdms"}
    assert _target_tokens("no_colon") == set()


def test_find_matching_fact_handles_ambiguous_prefix(tmp_path: Path) -> None:
    """When a gold paper_trunc expands to multiple full IDs (e.g. bar2024 → bar2024 +
    bar2024a), accept a hit on EITHER paper."""
    from eval.wiki_eval_augment import find_matching_fact_by_target
    from legacy.graphrag.wiki_semantic_schema import SemanticFact

    augment_by_paper = {
        "paper:bar2024a": [
            SemanticFact(
                paper_id="paper:bar2024a",
                fact_id="fact:bar2024a:llm:1",
                fact_type="Material",
                text="PDMS elastomer matrix",
                evidence="irrelevant",
                confidence=0.7,
                targets=("material:pdms_elastomer",),
            )
        ]
    }
    match = find_matching_fact_by_target(
        gold_target_id="material:pdms",
        gold_label="Material",
        paper_ids=["paper:bar2024", "paper:bar2024a"],
        augment_by_paper=augment_by_paper,
    )
    assert match is not None
    assert match.paper_id == "paper:bar2024a"
