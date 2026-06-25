import json
from pathlib import Path

import pytest


def _stub_eval_result(weighted: float):
    from eval.wiki_eval_quality import QualityResult

    return QualityResult(
        query_id="q",
        bucket="Metric",
        language="en",
        faithfulness=weighted,
        relevance=weighted,
        precision=weighted,
        weighted=weighted,
        rationales={},
    )


def _write_queries(path: Path, n: int) -> None:
    lines = [
        json.dumps({"query_id": f"q{i}", "query": f"Q{i}", "language": "en", "bucket": "Metric"})
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestVariantDefinitions:
    """Variant menu drives spec coverage — drift here means lever 누락."""

    def test_baseline_variants_present(self):
        from eval.wiki_eval_bakeoff_v2 import VARIANTS

        ids = {v.id for v in VARIANTS}
        assert {"V0a", "V0b"}.issubset(ids), "baseline pair required for noise floor"

    def test_candidates_exclude_baseline(self):
        from eval.wiki_eval_bakeoff_v2 import CANDIDATE_IDS

        assert not any(c.startswith("V0") for c in CANDIDATE_IDS)
        assert len(CANDIDATE_IDS) >= 4, "spec mandates ≥4 levers"


class TestRunVariant:
    def test_collects_rows_and_isolates_failures(self, tmp_path, monkeypatch):
        from eval import wiki_eval_bakeoff_v2 as mod

        queries = tmp_path / "queries.jsonl"
        _write_queries(queries, 3)

        call_count = {"n": 0}

        def fake_evaluate_query(query_obj, **_kwargs):
            call_count["n"] += 1
            if query_obj["query_id"] == "q1":
                raise RuntimeError("simulated judge timeout")
            return _stub_eval_result(weighted=0.8)

        monkeypatch.setattr(mod, "evaluate_query", fake_evaluate_query)

        variant = mod.Variant("V_test", "smoke", queries_path=str(queries))
        rows = mod.run_variant(variant, judge_model="haiku", judge_serial=True)
        assert call_count["n"] == 3
        assert len(rows) == 3
        assert sum(1 for r in rows if r["weighted"] is None) == 1, "failures recorded as None"
        assert sum(1 for r in rows if r["weighted"] == 0.8) == 2

    def test_missing_queries_file_raises(self, tmp_path):
        from eval.wiki_eval_bakeoff_v2 import Variant, run_variant

        v = Variant("V_test", "smoke", queries_path=str(tmp_path / "missing.jsonl"))
        with pytest.raises(FileNotFoundError):
            run_variant(v, judge_model="haiku", judge_serial=True)


class TestCheckpointRoundtrip:
    """variant checkpoint must survive a crash and resume verbatim."""

    def test_write_then_read_preserves_values_and_none(self, tmp_path):
        from eval.wiki_eval_bakeoff_v2 import read_variant_rows, write_variant_rows

        rows = [
            {
                "query_id": "q0",
                "bucket": "Metric",
                "language": "en",
                "faithfulness": 0.9,
                "relevance": 0.7,
                "precision": 0.5,
                "weighted": 0.76,
            },
            {
                "query_id": "q1",
                "bucket": "Material",
                "language": "ko",
                "faithfulness": None,
                "relevance": None,
                "precision": None,
                "weighted": None,
            },
        ]
        path = tmp_path / "variant_V0a.csv"
        write_variant_rows(path, rows)
        loaded = read_variant_rows(path)
        assert loaded[0]["weighted"] == 0.76
        assert loaded[1]["weighted"] is None
        assert loaded[1]["query_id"] == "q1"

    def test_main_resumes_existing_variant_without_rerunning(self, tmp_path, monkeypatch):
        from eval import wiki_eval_bakeoff_v2 as mod

        # Pre-seed a V0a checkpoint; run_variant must NOT be called for it.
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        mod.write_variant_rows(
            out_dir / "variant_V0a.csv",
            [
                {
                    "query_id": "q0",
                    "bucket": "Metric",
                    "language": "en",
                    "faithfulness": 0.9,
                    "relevance": 0.7,
                    "precision": 0.5,
                    "weighted": 0.76,
                }
            ],
        )

        ran: list[str] = []

        def fake_run_variant(variant, *_a, **_kw):
            ran.append(variant.id)
            return [
                {
                    "query_id": "q0",
                    "bucket": "Metric",
                    "language": "en",
                    "faithfulness": 0.8,
                    "relevance": 0.6,
                    "precision": 0.4,
                    "weighted": 0.66,
                }
            ]

        monkeypatch.setattr(mod, "run_variant", fake_run_variant)
        mod.main(["--out-dir", str(out_dir), "--variants", "V0a,V0b", "--limit-q", "1"])
        assert "V0a" not in ran, "V0a had a checkpoint — must resume, not re-run"
        assert "V0b" in ran, "V0b had no checkpoint — must run"


class TestAggregateAndWinner:
    def test_aggregate_ignores_nones(self):
        from eval.wiki_eval_bakeoff_v2 import aggregate

        rows = [
            {"weighted": 0.8, "faithfulness": 0.9, "relevance": 0.7, "precision": 0.6},
            {"weighted": None, "faithfulness": None, "relevance": None, "precision": None},
            {"weighted": 0.6, "faithfulness": 0.7, "relevance": 0.5, "precision": 0.4},
        ]
        s = aggregate(rows)
        assert s["n"] == 2
        assert s["weighted_mean"] == pytest.approx(0.7)

    def test_winner_passes_threshold(self, tmp_path):
        from eval.wiki_eval_bakeoff_v2 import write_winner

        summaries = {
            "V0a": {
                "n": 90,
                "weighted_mean": 0.750,
                "faith_mean": 0.9,
                "rel_mean": 0.6,
                "prec_mean": 0.55,
                "weighted_std": 0.1,
            },
            "V0b": {
                "n": 90,
                "weighted_mean": 0.748,
                "faith_mean": 0.9,
                "rel_mean": 0.6,
                "prec_mean": 0.55,
                "weighted_std": 0.1,
            },
            # Δ = +0.060 vs V0a, noise floor = max(0.05, 0.002) = 0.05 → winner.
            "V_pprA0.6": {
                "n": 90,
                "weighted_mean": 0.810,
                "faith_mean": 0.92,
                "rel_mean": 0.68,
                "prec_mean": 0.60,
                "weighted_std": 0.09,
            },
        }
        out = tmp_path / "winner.md"
        write_winner(summaries, out)
        body = out.read_text()
        assert "Winner: **V_pprA0.6**" in body
        assert "Δ vs V0a = +0.060" in body

    def test_winner_no_candidate_above_noise(self, tmp_path):
        from eval.wiki_eval_bakeoff_v2 import write_winner

        summaries = {
            "V0a": {
                "n": 90,
                "weighted_mean": 0.750,
                "faith_mean": 0.9,
                "rel_mean": 0.6,
                "prec_mean": 0.55,
                "weighted_std": 0.1,
            },
            "V0b": {
                "n": 90,
                "weighted_mean": 0.748,
                "faith_mean": 0.9,
                "rel_mean": 0.6,
                "prec_mean": 0.55,
                "weighted_std": 0.1,
            },
            "V_depth2": {
                "n": 90,
                "weighted_mean": 0.770,  # within noise of 0.800 threshold
                "faith_mean": 0.91,
                "rel_mean": 0.61,
                "prec_mean": 0.56,
                "weighted_std": 0.1,
            },
        }
        out = tmp_path / "winner.md"
        write_winner(summaries, out)
        assert "No winner" in out.read_text()
