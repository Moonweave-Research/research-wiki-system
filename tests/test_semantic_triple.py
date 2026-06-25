import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legacy.graphrag.wiki_semantic_schema import (  # noqa: E402
    SemanticTriple,
    _stable_triple_id,
    read_triples_jsonl,
    triple_from_dict,
    write_triples_jsonl,
)


def _make(**overrides):
    base = dict(
        paper_id="paper:abc2023",
        triple_id=_stable_triple_id("paper:abc2023", "PEDOT:PSS", "exhibits", "1000 S/cm"),
        subject="PEDOT:PSS",
        predicate="exhibits",
        object="1000 S/cm",
        evidence="The film exhibits 1000 S/cm conductivity.",
        confidence=0.9,
    )
    base.update(overrides)
    return SemanticTriple(**base)


def test_triple_construction_valid():
    t = _make()
    assert t.subject == "PEDOT:PSS"
    assert t.predicate == "exhibits"
    assert t.object == "1000 S/cm"
    assert 0.0 <= t.confidence <= 1.0
    assert t.triple_id.startswith("triple:")


def test_triple_rejects_invalid_paper_id():
    with pytest.raises(ValueError, match="paper:"):
        _make(paper_id="abc2023")


def test_triple_rejects_invalid_triple_id():
    with pytest.raises(ValueError, match="triple:"):
        _make(triple_id="not-prefixed-id")


def test_triple_rejects_bad_confidence():
    with pytest.raises(ValueError, match="confidence"):
        _make(confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        _make(confidence=-0.1)


def test_triple_rejects_empty_field():
    with pytest.raises(ValueError, match="subject"):
        _make(subject="")
    with pytest.raises(ValueError, match="evidence"):
        _make(evidence="")


def test_write_then_read_roundtrip(tmp_path):
    triples = [
        _make(),
        _make(
            triple_id=_stable_triple_id("paper:abc2023", "azobenzene", "isomerizes", "UV"),
            subject="azobenzene",
            predicate="isomerizes",
            object="UV",
            evidence="Azobenzene isomerizes under UV light.",
        ),
    ]
    path = tmp_path / "triples.jsonl"
    write_triples_jsonl(path, triples)

    loaded = read_triples_jsonl(path)
    assert len(loaded) == 2
    assert {t.subject for t in loaded} == {"PEDOT:PSS", "azobenzene"}
    # Every loaded item round-trips back to identical dict.
    for original, restored in zip(triples, loaded):
        assert restored.to_dict() == original.to_dict()


def test_triple_from_dict_recovers_object():
    t = _make()
    row = json.loads(json.dumps(t.to_dict()))
    restored = triple_from_dict(row)
    assert restored == t


def test_stable_triple_id_is_deterministic():
    a = _stable_triple_id("paper:p1", "subj", "pred", "obj")
    b = _stable_triple_id("paper:p1", "subj", "pred", "obj")
    c = _stable_triple_id("paper:p1", "subj", "pred", "OBJ")
    assert a == b
    assert a != c
