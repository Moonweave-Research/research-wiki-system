def test_module_imports():
    from eval import wiki_eval_query_synth

    assert hasattr(wiki_eval_query_synth, "synthesize_query")
    assert hasattr(wiki_eval_query_synth, "DEFAULT_SYNTH_MODEL")


def test_synthesize_query_calls_claude(monkeypatch):
    from eval import wiki_eval_query_synth as mod

    fake_query = "What is the tensile strength of the composite?"
    monkeypatch.setattr(mod, "_call_claude", lambda prompt, model, **kw: fake_query)

    seed = mod.SeedFact(
        fact_id="fact:foo:m0",
        paper_id="paper:foo2024",
        fact_type="Metric",
        text="tensile strength 12 MPa",
        targets=["metric:tensile_strength"],
    )
    q = mod.synthesize_query(seed, language="en")
    assert q.query == fake_query
    assert q.language == "en"
    assert q.expected_paper == "paper:foo2024"
    assert q.expected_fact == "fact:foo:m0"
    assert q.bucket == "Metric"
