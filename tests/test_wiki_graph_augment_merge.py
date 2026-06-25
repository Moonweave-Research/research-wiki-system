import json


def test_graph_build_merges_augment_jsonl(tmp_path):
    from legacy.graphrag import wiki_graph

    facts = tmp_path / "facts.jsonl"
    aug = tmp_path / "facts_llm_augment.jsonl"
    facts.write_text(
        json.dumps(
            {
                "paper_id": "paper:test",
                "fact_id": "fact:test:c0",
                "fact_type": "Claim",
                "text": "x",
                "evidence": "y",
                "confidence": 0.9,
                "targets": ["concept:Foo"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    aug.write_text(
        json.dumps(
            {
                "paper_id": "paper:test",
                "fact_id": "fact:test:llm:abc",
                "fact_type": "Mechanism",
                "text": "m",
                "evidence": "e",
                "confidence": 0.7,
                "targets": ["mechanism:foo_bar"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    merged = wiki_graph._load_all_facts(tmp_path)
    assert len(merged) == 2
    types = {f.fact_type for f in merged}
    assert types == {"Claim", "Mechanism"}


def test_graph_build_handles_missing_augment_jsonl(tmp_path):
    """Backwards-compat: when facts_llm_augment.jsonl is absent, returns just facts.jsonl content."""
    from legacy.graphrag import wiki_graph

    facts = tmp_path / "facts.jsonl"
    facts.write_text(
        json.dumps(
            {
                "paper_id": "paper:test",
                "fact_id": "fact:test:c0",
                "fact_type": "Claim",
                "text": "x",
                "evidence": "y",
                "confidence": 0.9,
                "targets": ["concept:Foo"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    merged = wiki_graph._load_all_facts(tmp_path)
    assert len(merged) == 1
    assert merged[0].fact_type == "Claim"
