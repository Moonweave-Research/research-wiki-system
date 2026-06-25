def test_build_faithfulness_prompt_contains_inputs():
    from eval.wiki_eval_quality import build_faithfulness_prompt

    p = build_faithfulness_prompt(
        question="What is the tensile strength?",
        contexts=["Tensile strength was 12 MPa."],
        answer="12 MPa.",
    )
    assert "12 MPa" in p
    assert "What is the tensile strength?" in p
    assert "JSON" in p


def test_score_faithfulness_with_mocked_judge(monkeypatch):
    from eval import wiki_eval_quality as mod

    monkeypatch.setattr(mod, "_call_judge", lambda *a, **k: {"score": 0.8, "rationale": "ok"})
    s, r = mod.score_faithfulness("q", ["c"], "a")
    assert s == 0.8 and r == "ok"


def test_score_precision_with_mocked_judge(monkeypatch):
    from eval import wiki_eval_quality as mod

    monkeypatch.setattr(
        mod,
        "_call_judge",
        lambda *a, **k: {
            "items": [
                {"index": 0, "useful": True, "rationale": "relevant"},
                {"index": 1, "useful": False, "rationale": "off-topic"},
            ]
        },
    )
    prec, items = mod.score_precision("q", ["ctx1", "ctx2"])
    assert prec == 0.5
    assert len(items) == 2


class TestCallJudgeRetry:
    """One retry survives the transient ProviderError seen during 5/25 starvation."""

    def test_succeeds_on_first_attempt(self, monkeypatch):
        from eval import wiki_eval_quality as mod

        calls = {"n": 0}

        def fake_provider_call(*_a, **_kw):
            calls["n"] += 1
            return '{"score": 0.9, "rationale": "ok"}'

        monkeypatch.setattr(mod, "provider_call", fake_provider_call)
        result = mod._call_judge("p", "haiku", timeout_s=10, retries=1)
        assert result == {"score": 0.9, "rationale": "ok"}
        assert calls["n"] == 1

    def test_retries_then_succeeds(self, monkeypatch):
        from eval import wiki_eval_quality as mod
        from scripts.core.wiki_llm_provider import ProviderError

        attempts = {"n": 0}

        def flaky(*_a, **_kw):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ProviderError("transient quota timeout")
            return '{"score": 0.7, "rationale": "second-try"}'

        monkeypatch.setattr(mod, "provider_call", flaky)
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)  # skip 5s backoff
        result = mod._call_judge("p", "haiku", timeout_s=10, retries=1)
        assert result["rationale"] == "second-try"
        assert attempts["n"] == 2

    def test_exhausted_retries_raises(self, monkeypatch):
        from eval import wiki_eval_quality as mod
        from scripts.core.wiki_llm_provider import ProviderError

        def always_fail(*_a, **_kw):
            raise ProviderError("persistent failure")

        monkeypatch.setattr(mod, "provider_call", always_fail)
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
        import pytest

        with pytest.raises(RuntimeError, match="after 2 attempt"):
            mod._call_judge("p", "haiku", timeout_s=10, retries=1)


class TestParseJudgeResponse:
    """The 5/25 failure modes the parser must absorb."""

    def test_plain_object(self):
        from eval.wiki_eval_quality import _parse_judge_response

        assert _parse_judge_response('{"score": 1.0}') == {"score": 1.0}

    def test_fenced(self):
        from eval.wiki_eval_quality import _parse_judge_response

        assert _parse_judge_response('```json\n{"score": 0.5}\n```') == {"score": 0.5}

    def test_concatenated_objects_returns_first(self):
        from eval.wiki_eval_quality import _parse_judge_response

        # claude-CLI list envelope joined with "\n" produces this shape.
        text = '{"score": 0.5}\n{"score": 0.9}'
        assert _parse_judge_response(text) == {"score": 0.5}

    def test_leading_commentary_stripped(self):
        from eval.wiki_eval_quality import _parse_judge_response

        assert _parse_judge_response('Sure! {"score": 0.4}') == {"score": 0.4}


class TestEvaluateQuerySerialFlag:
    """judge_serial=True bypasses the ThreadPoolExecutor branch."""

    def test_serial_calls_three_judges_in_order(self, monkeypatch):
        from pathlib import Path

        from eval import wiki_eval_quality as mod

        call_order: list[str] = []

        def fake_query_answer_context(**_kw):
            return {"context_items": [{"evidence": "ctx-a"}], "query": _kw["query"]}

        def fake_answer(_bundle):
            return "answer-text"

        def fake_faith(*_a, **_kw):
            call_order.append("faith")
            return 0.9, "f-r"

        def fake_rel(*_a, **_kw):
            call_order.append("rel")
            return 0.8, "r-r"

        def fake_prec(*_a, **_kw):
            call_order.append("prec")
            return 0.7, [{"index": 0, "useful": True}]

        # Patch within the loadable wiki_answer module so the local import resolves.
        import legacy.graphrag.wiki_answer as wa

        monkeypatch.setattr(wa, "query_answer_context", fake_query_answer_context)
        monkeypatch.setattr(wa, "answer_from_context_bundle", fake_answer)
        monkeypatch.setattr(mod, "score_faithfulness", fake_faith)
        monkeypatch.setattr(mod, "score_relevance", fake_rel)
        monkeypatch.setattr(mod, "score_precision", fake_prec)

        result = mod.evaluate_query(
            {"query_id": "q1", "query": "Q", "language": "en"},
            rerank=False,
            model="haiku",
            graph_index=Path("."),
            judge_serial=True,
        )
        assert call_order == ["faith", "rel", "prec"], "serial path must call in declared order"
        assert result.weighted == 0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.7
