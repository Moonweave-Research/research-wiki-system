import json

import pytest

from legacy.graphrag.wiki_semantic_schema import (
    SemanticFact,
    fact_from_dict,
    fact_to_edge_rows,
    facts_to_dicts,
    read_facts_jsonl,
    write_facts_jsonl,
)


def test_semantic_fact_serializes_to_json_dict():
    fact = SemanticFact(
        paper_id="paper:feng2024",
        fact_id="fact:feng2024:claim:1",
        fact_type="Claim",
        text="Fluorinated elastomers achieved high strain at moderate field.",
        evidence="253% strain at 46 MV/m",
        confidence=0.82,
        targets=["concept:Dielectric_Elastomer_Actuator", "metric:strain"],
    )

    payload = fact.to_dict()

    assert payload["paper_id"] == "paper:feng2024"
    assert payload["fact_type"] == "Claim"
    assert payload["confidence"] == 0.82
    assert payload["targets"] == ["concept:Dielectric_Elastomer_Actuator", "metric:strain"]


def test_fact_to_edge_rows_links_paper_fact_and_targets():
    fact = SemanticFact(
        paper_id="paper:feng2024",
        fact_id="fact:feng2024:claim:1",
        fact_type="Claim",
        text="High strain result.",
        evidence="253% strain",
        confidence=0.9,
        targets=["metric:strain"],
    )

    edges = fact_to_edge_rows(fact)

    assert {
        "source": "paper:feng2024",
        "type": "HAS_FACT",
        "target": "fact:feng2024:claim:1",
    } in edges
    assert {"source": "fact:feng2024:claim:1", "type": "ABOUT", "target": "metric:strain"} in edges


def test_semantic_fact_rejects_invalid_fact_type():
    with pytest.raises(ValueError, match="Unsupported fact_type"):
        SemanticFact(
            paper_id="paper:feng2024",
            fact_id="fact:feng2024:claim:1",
            fact_type="Unsupported",
            text="High strain result.",
            evidence="253% strain",
            confidence=0.9,
            targets=["metric:strain"],
        )


@pytest.mark.parametrize("confidence", [1.1, -0.1])
def test_semantic_fact_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
        SemanticFact(
            paper_id="paper:feng2024",
            fact_id="fact:feng2024:claim:1",
            fact_type="Claim",
            text="High strain result.",
            evidence="253% strain",
            confidence=confidence,
            targets=["metric:strain"],
        )


def test_semantic_fact_stores_targets_immutably_but_serializes_list():
    fact = SemanticFact(
        paper_id="paper:feng2024",
        fact_id="fact:feng2024:claim:1",
        fact_type="Claim",
        text="High strain result.",
        evidence="253% strain",
        confidence=0.9,
        targets=["metric:strain"],
    )

    assert fact.targets == ("metric:strain",)
    with pytest.raises(AttributeError):
        fact.targets.append("metric:field")
    assert fact.to_dict()["targets"] == ["metric:strain"]


@pytest.mark.parametrize("targets", ["metric:strain", [object()]])
def test_semantic_fact_rejects_malformed_targets(targets):
    with pytest.raises(ValueError, match="targets must be a list or tuple of strings"):
        SemanticFact(
            paper_id="paper:feng2024",
            fact_id="fact:feng2024:claim:1",
            fact_type="Claim",
            text="High strain result.",
            evidence="253% strain",
            confidence=0.9,
            targets=targets,
        )


@pytest.mark.parametrize("targets", [[""], ["junk"], ["metric:"], ["metric:two words"]])
def test_semantic_fact_rejects_invalid_target_ids(targets):
    with pytest.raises(ValueError, match="targets must be non-empty prefixed node ids"):
        SemanticFact(
            paper_id="paper:feng2024",
            fact_id="fact:feng2024:claim:1",
            fact_type="Claim",
            text="High strain result.",
            evidence="253% strain",
            confidence=0.9,
            targets=targets,
        )


def test_semantic_fact_rejects_invalid_paper_id_prefix():
    with pytest.raises(ValueError, match="paper_id must start with 'paper:'"):
        SemanticFact(
            paper_id="feng2024",
            fact_id="fact:feng2024:claim:1",
            fact_type="Claim",
            text="High strain result.",
            evidence="253% strain",
            confidence=0.9,
            targets=["metric:strain"],
        )


def test_semantic_fact_rejects_invalid_fact_id_prefix():
    with pytest.raises(ValueError, match="fact_id must start with 'fact:'"):
        SemanticFact(
            paper_id="paper:feng2024",
            fact_id="feng2024:claim:1",
            fact_type="Claim",
            text="High strain result.",
            evidence="253% strain",
            confidence=0.9,
            targets=["metric:strain"],
        )


def test_fact_from_dict_defaults_missing_targets_to_empty_tuple():
    fact = fact_from_dict(
        {
            "paper_id": "paper:feng2024",
            "fact_id": "fact:feng2024:claim:1",
            "fact_type": "Claim",
            "text": "High strain result.",
            "evidence": "253% strain",
            "confidence": 0.9,
        }
    )

    assert fact.targets == ()


def test_fact_from_dict_defaults_empty_targets_to_empty_tuple():
    fact = fact_from_dict(
        {
            "paper_id": "paper:feng2024",
            "fact_id": "fact:feng2024:claim:1",
            "fact_type": "Claim",
            "text": "High strain result.",
            "evidence": "253% strain",
            "confidence": 0.9,
            "targets": [],
        }
    )

    assert fact.targets == ()


def test_fact_from_dict_raises_key_error_for_missing_required_field():
    with pytest.raises(KeyError):
        fact_from_dict(
            {
                "paper_id": "paper:feng2024",
                "fact_type": "Claim",
                "text": "High strain result.",
                "evidence": "253% strain",
                "confidence": 0.9,
            }
        )


def test_fact_from_dict_preserves_existing_validation_errors():
    with pytest.raises(ValueError, match="Unsupported fact_type"):
        fact_from_dict(
            {
                "paper_id": "paper:feng2024",
                "fact_id": "fact:feng2024:claim:1",
                "fact_type": "Unsupported",
                "text": "High strain result.",
                "evidence": "253% strain",
                "confidence": 0.9,
            }
        )


def test_facts_to_dicts_serializes_multiple_facts():
    facts = [
        SemanticFact(
            paper_id="paper:feng2024",
            fact_id="fact:feng2024:claim:1",
            fact_type="Claim",
            text="High strain result.",
            evidence="253% strain",
            confidence=0.9,
            targets=["metric:strain"],
        ),
        SemanticFact(
            paper_id="paper:kim2023",
            fact_id="fact:kim2023:method:2",
            fact_type="Method",
            text="Used solvent casting.",
            evidence="Methods section",
            confidence=0.7,
        ),
    ]

    payload = facts_to_dicts(facts)

    assert payload == [fact.to_dict() for fact in facts]


def test_write_and_read_facts_jsonl_round_trip(tmp_path):
    path = tmp_path / "facts.jsonl"
    facts = [
        SemanticFact(
            paper_id="paper:feng2024",
            fact_id="fact:feng2024:claim:1",
            fact_type="Claim",
            text="High strain result.",
            evidence="253% strain",
            confidence=0.9,
            targets=["metric:strain", "concept:Dielectric_Elastomer_Actuator"],
        ),
        SemanticFact(
            paper_id="paper:kim2023",
            fact_id="fact:kim2023:method:2",
            fact_type="Method",
            text="Used solvent casting.",
            evidence="Methods section",
            confidence=0.7,
        ),
    ]

    write_facts_jsonl(path, facts)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [json.dumps(fact.to_dict(), sort_keys=True) for fact in facts]
    assert read_facts_jsonl(path) == facts
