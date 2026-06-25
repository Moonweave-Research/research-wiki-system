import json
import sys

from scripts.ingest.wiki_fact_extractor import (
    extract_facts_from_note,
    extract_section_map,
    infer_targets,
    main,
    split_fact_units,
)
from legacy.graphrag.wiki_semantic_schema import read_facts_jsonl


def make_note(path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""---
aliases: [Pilot Paper]
domain: research
category: materials
tags: []
sources: []
abstract: Pilot abstract
---
# Pilot Paper

{body}
""",
        encoding="utf-8",
    )


def test_extract_section_map_reads_markdown_sections():
    body = """## Abstract
First sentence.

## 핵심 메커니즘
- Charge trapping increases hysteresis.

## 방법론
EHD printing was used.
"""

    sections = extract_section_map(body)

    assert sections["Abstract"] == "First sentence."
    assert sections["핵심 메커니즘"] == "- Charge trapping increases hysteresis."
    assert sections["방법론"] == "EHD printing was used."


def test_split_fact_units_prefers_bullets_and_splits_sentences():
    bullet_text = "- First bullet.\n- **한계**: Second bullet."
    sentence_text = "First sentence. Second sentence with 3V."

    assert split_fact_units(bullet_text) == ["First bullet.", "한계: Second bullet."]
    assert split_fact_units(sentence_text) == ["First sentence.", "Second sentence with 3V."]


def test_infer_targets_maps_keywords_to_prefixed_node_ids():
    text = "Low-voltage EHD printed soft actuator reached 3V with high force."

    assert infer_targets(text) == [
        "metric:voltage",
        "metric:force",
        "concept:Soft_Actuator",
        "method:electrohydrodynamic_printing",
    ]


def test_infer_targets_avoids_substring_false_positives():
    text = "The idea was evaluated under vacuum with viscoelastic vibration damping."

    assert infer_targets(text) == []


def test_extract_facts_from_note_maps_sections_to_fact_types(tmp_path):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "cao2019.md"
    make_note(
        note,
        """## Abstract
The actuator achieved about 3V operation and 1.0 cm-1 curvature.

## 핵심 메커니즘
- Thermal expansion mismatch drives bending.

## 주요 기여
- Low-voltage soft actuator performance was demonstrated.

## 방법론
- EHD printing was used for the heater.
""",
    )

    facts = extract_facts_from_note(note, wiki, limit_per_paper=6)

    assert [fact.fact_type for fact in facts] == ["Claim", "Mechanism", "Claim", "Method"]
    assert facts[0].fact_id.startswith("fact:cao2019:claim:")
    assert facts[0].paper_id == "paper:cao2019"
    assert "metric:voltage" in facts[0].targets
    assert "metric:curvature" in facts[0].targets
    assert "mechanism:electrothermal_actuation" in facts[1].targets
    assert "method:electrohydrodynamic_printing" in facts[3].targets


def test_fact_ids_are_stable_when_sentence_order_changes(tmp_path):
    wiki = tmp_path / "wiki"
    first = wiki / "research" / "first.md"
    second = wiki / "research" / "second.md"
    make_note(
        first,
        """## Abstract
The actuator achieved low-voltage operation at 3V.
The actuator produced 3 mN force.
""",
    )
    make_note(
        second,
        """## Abstract
The actuator produced 3 mN force.
The actuator achieved low-voltage operation at 3V.
""",
    )

    first_by_text = {fact.text: fact.fact_id for fact in extract_facts_from_note(first, wiki)}
    second_by_text = {
        fact.text.replace("first", "second"): fact.fact_id.replace(":second:", ":first:")
        for fact in extract_facts_from_note(second, wiki)
    }

    assert first_by_text == second_by_text


def test_duplicate_text_in_different_sections_gets_distinct_fact_ids(tmp_path):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "repeated.md"
    make_note(
        note,
        """## Abstract
Repeated claim sentence.

## 주요 기여
- Repeated claim sentence.
""",
    )

    facts = extract_facts_from_note(note, wiki)

    assert len(facts) == 2
    assert facts[0].text == facts[1].text
    assert facts[0].fact_id != facts[1].fact_id


def test_duplicate_text_in_same_section_gets_distinct_fact_ids(tmp_path):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "repeated.md"
    make_note(
        note,
        """## Abstract
Repeated claim sentence.
Repeated claim sentence.
""",
    )

    facts = extract_facts_from_note(note, wiki)

    assert len(facts) == 2
    assert facts[0].text == facts[1].text
    assert facts[0].fact_id != facts[1].fact_id


def test_extract_facts_prioritizes_targeted_units_with_small_limit(tmp_path):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "cao2019.md"
    make_note(
        note,
        """## Abstract
Soft actuators are widely studied.
This paper introduces a heater.
The actuator achieved low-voltage operation at 3V.
The actuator produced 3 mN force.
""",
    )

    facts = extract_facts_from_note(note, wiki, limit_per_paper=2)

    assert [fact.targets for fact in facts] == [
        ("metric:voltage",),
        ("metric:force",),
    ]


def test_extract_facts_from_sparse_note_returns_empty(tmp_path):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "sparse2024.md"
    make_note(note, "## Abstract\n\n")

    assert extract_facts_from_note(note, wiki) == []


def test_cli_writes_facts_jsonl_without_modifying_source(monkeypatch, tmp_path):
    wiki = tmp_path / "wiki"
    note = wiki / "research" / "cao2019.md"
    output = tmp_path / "facts.jsonl"
    make_note(note, "## Abstract\nLow-voltage actuation reached 3V.")
    before = note.read_text(encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki_fact_extractor.py",
            "--wiki-dir",
            str(wiki),
            "--output",
            str(output),
            "--paper",
            "paper:cao2019",
        ],
    )

    main()

    assert note.read_text(encoding="utf-8") == before
    facts = read_facts_jsonl(output)
    assert len(facts) == 1
    assert facts[0].paper_id == "paper:cao2019"
    assert json.loads(output.read_text(encoding="utf-8"))["fact_type"] == "Claim"


def test_build_concept_patterns_loads_aliases_and_stem(tmp_path):
    from scripts.ingest.wiki_fact_extractor import build_concept_patterns

    wiki = tmp_path / "wiki"
    concepts = wiki / "concepts"
    concepts.mkdir(parents=True)
    (concepts / "Charge_Trapping.md").write_text(
        "---\naliases:\n  - Charge Trap\n  - 전하 트래핑\n  - ab\ndomain: research\n---\n# Charge Trapping\n",
        encoding="utf-8",
    )

    patterns = build_concept_patterns(wiki)

    assert len(patterns) == 1
    target_id, regexes = patterns[0]
    assert target_id == "concept:Charge_Trapping"
    joined = " | ".join(regexes)
    assert "Charge\\ Trapping" in joined or "Charge\\ Trap" in joined
    assert "전하\\ 트래핑" in joined
    assert "ab" not in joined


def test_load_active_patterns_augments_target_patterns(tmp_path):
    import scripts.ingest.wiki_fact_extractor as fact_extractor

    wiki = tmp_path / "wiki"
    concepts = wiki / "concepts"
    concepts.mkdir(parents=True)
    (concepts / "Liquid_Crystal_Elastomer.md").write_text(
        "---\naliases:\n  - LCE\ndomain: research\n---\n# LCE\n",
        encoding="utf-8",
    )

    fact_extractor._ACTIVE_PATTERNS = None
    active = fact_extractor.load_active_patterns(wiki)
    target_ids = [tid for tid, _ in active]

    assert "concept:Liquid_Crystal_Elastomer" in target_ids
    assert "concept:Charge_Trapping" in target_ids
    assert "metric:voltage" in target_ids
    fact_extractor._ACTIVE_PATTERNS = None


def test_infer_targets_uses_active_patterns(tmp_path):
    import scripts.ingest.wiki_fact_extractor as fact_extractor
    from scripts.ingest.wiki_fact_extractor import infer_targets, load_active_patterns

    wiki = tmp_path / "wiki"
    concepts = wiki / "concepts"
    concepts.mkdir(parents=True)
    (concepts / "Liquid_Crystal_Elastomer.md").write_text(
        "---\naliases:\n  - LCE\n  - 액정 엘라스토머\ndomain: research\n---\n",
        encoding="utf-8",
    )

    fact_extractor._ACTIVE_PATTERNS = None
    load_active_patterns(wiki)
    try:
        assert "concept:Liquid_Crystal_Elastomer" in infer_targets("the LCE film bent at 80 °C")
        assert "concept:Liquid_Crystal_Elastomer" in infer_targets(
            "이 연구는 액정 엘라스토머 기반 actuator를 사용한다"
        )
    finally:
        fact_extractor._ACTIVE_PATTERNS = None
