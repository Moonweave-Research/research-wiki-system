from pathlib import Path

from eval.wiki_fact_eval import load_benchmark_cases


def test_semantic_fact_paraphrase_benchmark_is_loadable_and_well_formed():
    benchmark = Path("benchmarks/semantic_facts_paraphrase.json")

    cases = load_benchmark_cases(benchmark)

    assert len(cases) >= 20
    case_ids = [case.case_id for case in cases]
    assert len(set(case_ids)) == len(case_ids)
    targets = {case.target for case in cases}
    assert {
        "concept:Charge_Trapping",
        "concept:Dielectric_Elastomer_Actuator",
        "mechanism:electrothermal_actuation",
        "method:electrohydrodynamic_printing",
        "metric:strain",
        "metric:voltage",
    }.issubset(targets)
    for case in cases:
        assert case.expected_papers
        assert case.query
        assert case.target
        assert not (
            set(case.expected_papers or []) & set(case.forbidden_papers or [])
        )
        assert all(paper.startswith("paper:") for paper in case.expected_papers)
