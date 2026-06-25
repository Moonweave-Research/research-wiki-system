# Bake-off Winner Report
_Generated: 2026-05-06 20:39:00 KST_

- **Baseline V0a weighted_mean**: 0.756
- **V0a–V0b variance**: 0.005
- **Noise floor (max(0.05, variance))**: 0.050
- **Winning threshold (V0a + noise)**: 0.806

## No winner
All candidate variants within noise floor of V0a.

Next step: revisit lever menu — maybe combine, or dig into Phase 0 failure modes.

## All variants
| Variant | n | faith | rel | prec | weighted | std | description |
|---------|---|-------|-----|------|----------|-----|-------------|
| V0a | 93 | 0.942 | 0.582 | 0.555 | 0.756 | 0.152 | baseline mpnet rerank=True (variance run 1) |
| V0b | 94 | 0.942 | 0.560 | 0.562 | 0.751 | 0.143 | baseline mpnet rerank=True (variance run 2) |
| V1_e5large | 83 | 0.943 | 0.621 | 0.593 | 0.777 | 0.130 | embedding swap → multilingual-e5-large |
| V2_filter | 92 | 0.958 | 0.572 | 0.557 | 0.762 | 0.126 | hard-negative cosine filter (min_score=0.5) |
| V3_expand | 91 | 0.945 | 0.493 | 0.673 | 0.755 | 0.123 | LLM query expansion (precomputed) |

