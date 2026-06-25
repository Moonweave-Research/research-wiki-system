import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legacy.graphrag.wiki_traversal import _prune_neighbor_contributions  # noqa: E402


def _row(via, hop, raw, orig):
    return (via, hop, raw, orig)


def test_prune_below_score_drops_low_cosine_neighbors():
    contribs = {
        "fact:strong": _row("via1", 1, 0.8, "seed1"),
        "fact:medium": _row("via1", 1, 0.4, "seed1"),
        "fact:weak": _row("via1", 1, 0.15, "seed1"),
        "fact:noise": _row("via1", 1, 0.05, "seed1"),
    }
    pruned = _prune_neighbor_contributions(
        contribs,
        prune_below_score=0.3,
        max_neighbors_per_seed=0,
    )
    assert set(pruned.keys()) == {"fact:strong", "fact:medium"}


def test_max_neighbors_per_seed_caps_fan_out():
    # Seed A has 5 neighbors, seed B has 3. Cap = 2.
    contribs = {
        "fact:a1": _row("via", 1, 0.9, "seedA"),
        "fact:a2": _row("via", 1, 0.8, "seedA"),
        "fact:a3": _row("via", 1, 0.7, "seedA"),
        "fact:a4": _row("via", 1, 0.6, "seedA"),
        "fact:a5": _row("via", 1, 0.5, "seedA"),
        "fact:b1": _row("via", 1, 0.95, "seedB"),
        "fact:b2": _row("via", 1, 0.85, "seedB"),
        "fact:b3": _row("via", 1, 0.75, "seedB"),
    }
    pruned = _prune_neighbor_contributions(
        contribs,
        prune_below_score=0.0,
        max_neighbors_per_seed=2,
    )
    # Per seed, top-2 by raw kept.
    by_seed: dict[str, set[str]] = {}
    for fid, (_v, _h, _r, orig) in pruned.items():
        by_seed.setdefault(orig, set()).add(fid)
    assert by_seed["seedA"] == {"fact:a1", "fact:a2"}
    assert by_seed["seedB"] == {"fact:b1", "fact:b2"}


def test_both_filters_combined():
    contribs = {
        "fact:keep_high": _row("via", 1, 0.9, "seedA"),
        "fact:keep_mid": _row("via", 1, 0.5, "seedA"),
        "fact:drop_low": _row("via", 1, 0.2, "seedA"),  # below threshold
        "fact:drop_cap": _row("via", 1, 0.4, "seedA"),  # above threshold but capped
    }
    pruned = _prune_neighbor_contributions(
        contribs,
        prune_below_score=0.3,
        max_neighbors_per_seed=2,
    )
    # After threshold: {keep_high, keep_mid, drop_cap} survive
    # After cap=2 by raw: {keep_high (0.9), keep_mid (0.5)} survive
    assert set(pruned.keys()) == {"fact:keep_high", "fact:keep_mid"}


def test_no_pruning_when_both_args_default():
    contribs = {
        "fact:a": _row("v", 1, 0.05, "seed"),
        "fact:b": _row("v", 1, 0.95, "seed"),
    }
    # Both defaults → identity.
    out = _prune_neighbor_contributions(contribs, prune_below_score=0.0, max_neighbors_per_seed=0)
    assert out == contribs


def test_pruning_does_not_mutate_input():
    contribs = {"fact:a": _row("v", 1, 0.05, "seed")}
    original = dict(contribs)
    _prune_neighbor_contributions(contribs, prune_below_score=0.5, max_neighbors_per_seed=0)
    assert contribs == original


def test_community_context_bucket_bypasses_per_seed_cap():
    # Community-context neighbors have orig_seed="" (no real parent). The cap
    # must NOT lump them into one bucket — that would silently censor
    # across communities. Score threshold still applies.
    contribs = {
        "fact:c1": _row("community", 1, 0.6, ""),
        "fact:c2": _row("community", 1, 0.4, ""),
        "fact:c3": _row("community", 1, 0.2, ""),
    }
    pruned = _prune_neighbor_contributions(
        contribs, prune_below_score=0.0, max_neighbors_per_seed=2
    )
    # All three survive — cap=2 only constrains real seeds.
    assert set(pruned.keys()) == {"fact:c1", "fact:c2", "fact:c3"}


def test_community_context_still_obeys_score_threshold():
    # Score gate is independent of orig_seed; community items below threshold
    # are dropped just like any other neighbor.
    contribs = {
        "fact:c1": _row("community", 1, 0.6, ""),
        "fact:c2": _row("community", 1, 0.05, ""),  # below threshold
        "fact:s1": _row("via", 1, 0.7, "seedA"),
    }
    pruned = _prune_neighbor_contributions(
        contribs, prune_below_score=0.3, max_neighbors_per_seed=5
    )
    assert set(pruned.keys()) == {"fact:c1", "fact:s1"}
