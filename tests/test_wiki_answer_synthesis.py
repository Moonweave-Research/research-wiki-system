from legacy.graphrag import wiki_answer


_BUNDLE = {
    "query": "What tensile strength has been reported for PDMS actuators?",
    "match_mode": "hybrid",
    "context_items": [
        {
            "citation_id": "F1",
            "paper_id": "paper:wang2021",
            "paper_title": "Wang 2021 PDMS Composite",
            "paper_path": "research/wang2021.md",
            "fact_id": "fact:wang2021:m0",
            "text": "Tensile strength of PDMS composite measured at 12 MPa.",
        },
        {
            "citation_id": "F2",
            "paper_id": "paper:lee2023",
            "paper_title": "Lee 2023 Soft Actuator",
            "paper_path": "research/lee2023.md",
            "fact_id": "fact:lee2023:m0",
            "text": "PDMS actuator showed 15 MPa tensile strength.",
        },
    ],
}


def test_build_synthesis_prompt_contains_query_and_citations():
    prompt = wiki_answer._build_synthesis_prompt(
        _BUNDLE["query"],
        wiki_answer.answer_from_context_bundle(_BUNDLE),
    )
    assert "tensile strength" in prompt
    assert "[F1]" in prompt and "[F2]" in prompt
    assert "12 MPa" in prompt and "15 MPa" in prompt
    assert "aggregate" in prompt.lower() or "average" in prompt.lower()


def test_synthesize_answer_returns_llm_output(monkeypatch):
    fake = (
        "PDMS actuators have been reported with tensile strength of 12 MPa [F1] "
        "and 15 MPa [F2].\n\nCitations:\n[F1] Wang 2021 PDMS Composite\n[F2] Lee 2023 Soft Actuator"
    )
    monkeypatch.setattr(wiki_answer, "_call_synthesizer", lambda *a, **k: fake)
    out = wiki_answer.synthesize_answer(_BUNDLE)
    assert "12 MPa [F1]" in out
    assert "15 MPa [F2]" in out


def test_synthesize_answer_falls_back_when_cli_fails(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("claude CLI not available")

    monkeypatch.setattr(wiki_answer, "_call_synthesizer", boom)
    out = wiki_answer.synthesize_answer(_BUNDLE)
    assert "Local evidence answer:" in out
    assert "[F1]" in out and "[F2]" in out


def test_synthesize_answer_with_empty_context_skips_llm(monkeypatch):
    called = {"n": 0}

    def spy(*_a, **_k):
        called["n"] += 1
        return "should not be called"

    monkeypatch.setattr(wiki_answer, "_call_synthesizer", spy)
    out = wiki_answer.synthesize_answer(
        {"query": "no facts here", "match_mode": "hybrid", "context_items": []}
    )
    assert called["n"] == 0
    assert "No local evidence found" in out
