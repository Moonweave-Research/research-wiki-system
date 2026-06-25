#!/usr/bin/env bash
# smoke-check.sh — fast pre-build sanity (< 60 sec).
# Run before refresh-graph.sh to catch broken imports, missing deps, and
# floor regressions without committing to the full 90-min pipeline.
# Exit code 0 = all green. Non-zero = something is broken.
set -euo pipefail

GRAPH_INDEX="${GRAPH_INDEX:-.graph_index}"
PASS=0
FAIL=0

ok()   { echo "  [OK]  $*"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $*" >&2; FAIL=$((FAIL + 1)); }

echo "=== smoke-check: pipeline script imports ==="
for mod in \
    scripts.ingest.wiki_pdf_cache \
    scripts.ingest.wiki_pdf_fact_extractor \
    scripts.ingest.wiki_fact_extractor \
    legacy.graphrag.wiki_fact_overlay \
    scripts.ingest.wiki_llm_fact_augmenter \
    legacy.graphrag.wiki_graph \
    scripts.search.wiki_fact_embeddings \
    legacy.graphrag.wiki_communities \
    legacy.graphrag.wiki_index_manifest \
    legacy.graphrag.wiki_community_summaries \
    legacy.graphrag.wiki_verify_index \
    legacy.graphrag.wiki_answer \
    legacy.graphrag.wiki_query \
    legacy.graphrag.wiki_rerank \
    eval.wiki_eval_quality \
    scripts.core.wiki_llm_provider \
    scripts.core.wiki \
    scripts.core.build \
    scripts.search.wiki_embedding_defaults; do
    if uv run python -c "import $mod" 2>/dev/null; then
        ok "$mod"
    else
        fail "$mod import failed"
    fi
done

echo ""
echo "=== smoke-check: lean foundation build wiring ==="
if uv run python -m scripts.core.build --dry-run >/dev/null 2>&1; then
    ok "scripts.core.build --dry-run"
else
    fail "scripts.core.build --dry-run failed"
fi

echo ""
echo "=== smoke-check: backend unit tests ==="
if PYTHONPATH=apps/vault-ui uv run python -m pytest apps/vault-ui/backend/tests -x -q --tb=no 2>&1 | tail -3; then
    ok "backend tests"
else
    fail "backend tests failed"
fi

echo ""
echo "=== smoke-check: graph index ==="
if [ ! -d "$GRAPH_INDEX" ]; then
    fail "$GRAPH_INDEX missing — run refresh-graph.sh first"
else
    if uv run python -m legacy.graphrag.wiki_verify_index --graph-index "$GRAPH_INDEX" 2>/dev/null; then
        ok "wiki_verify_index passed"
    else
        fail "wiki_verify_index failed"
    fi

    META="$GRAPH_INDEX/fact_embeddings_meta.json"
    if [ -f "$META" ]; then
        MODEL=$(uv run python -c "import json; print(json.load(open('$META'))['model'])" 2>/dev/null || echo "unknown")
        ok "embedding model: $MODEL"
    else
        fail "fact_embeddings_meta.json missing"
    fi
fi

echo ""
echo "=== smoke-check: summary ==="
echo "  passed: $PASS  failed: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "  SMOKE FAILED — fix the above before running refresh-graph.sh"
    exit 1
fi
echo "  SMOKE OK — safe to run refresh-graph.sh"
