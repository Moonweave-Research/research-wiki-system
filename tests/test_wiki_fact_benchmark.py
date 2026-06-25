from pathlib import Path

from eval.wiki_fact_eval import load_benchmark_cases


def test_semantic_fact_seed_benchmark_is_loadable_and_well_formed():
    benchmark = Path("benchmarks/semantic_facts_seed.json")

    cases = load_benchmark_cases(benchmark)

    assert len(cases) >= 20
    case_by_id = {case.case_id: case for case in cases}
    assert len(case_by_id) == len(cases)
    assert case_by_id["cao-3v-curvature"].expected_papers == [
        "paper:caoHighperformanceLowvoltageSoft2019"
    ]
    assert case_by_id["choi-microliquid-resistive-heating"].target == (
        "mechanism:electrothermal_actuation"
    )
    assert case_by_id["bernat-dea-transducer-review"].target == (
        "concept:Dielectric_Elastomer_Actuator"
    )
    for case in cases:
        assert case.expected_papers
        assert case.query or case.target
        assert all(paper.startswith("paper:") for paper in case.expected_papers)
