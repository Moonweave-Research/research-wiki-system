#!/usr/bin/env python3
"""Route broad concept auto links to curated, more specific concept notes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.ingest import concept_routing
from scripts.ingest.concept_routing import DEFAULT_DB, DEFAULT_VAULT, ConceptPlan


def build_split_plan(
    vault_dir: Path,
    tag_papers: list[concept_routing.TagPapers],
    *,
    min_papers: int | None = None,
    backfill_min_papers: int = 20,
) -> tuple[list[ConceptPlan], concept_routing.RoutingResult]:
    routing = concept_routing.route_papers_to_concepts(
        vault_dir,
        tag_papers,
        min_papers=min_papers,
    )
    parent_plans = concept_routing.parent_backfill_plans(
        vault_dir,
        tag_papers,
        routing.suppressed_papers_by_parent,
        min_papers=backfill_min_papers,
    )
    plans_by_rel_path = {plan.rel_path: plan for plan in parent_plans}
    plans_by_rel_path.update({plan.rel_path: plan for plan in routing.specific_plans})
    return sorted(plans_by_rel_path.values(), key=lambda plan: plan.rel_path), routing


def apply_plan(plans: list[ConceptPlan], *, apply: bool) -> int:
    return concept_routing.apply_plan(plans, apply=apply, label="concept split")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--min-papers",
        type=int,
        default=None,
        help="Override the per-rule minimum qualifying paper count.",
    )
    parser.add_argument(
        "--backfill-min-papers",
        type=int,
        default=20,
        help="Minimum parent tag frequency to mirror concept_backfill auto sections.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print planned changes without writing")
    mode.add_argument("--apply", action="store_true", help="Write concept notes and auto sections")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_papers is not None and args.min_papers < 1:
        print("--min-papers must be >= 1", file=sys.stderr)
        return 2
    if args.backfill_min_papers < 1:
        print("--backfill-min-papers must be >= 1", file=sys.stderr)
        return 2
    if not args.db.exists():
        print(f"linkgraph database not found: {args.db}", file=sys.stderr)
        return 2
    if not args.vault.exists():
        print(f"vault directory not found: {args.vault}", file=sys.stderr)
        return 2

    tag_papers = concept_routing.load_tag_papers(args.db)
    plans, routing = build_split_plan(
        args.vault,
        tag_papers,
        min_papers=args.min_papers,
        backfill_min_papers=args.backfill_min_papers,
    )
    for slug in routing.skipped_rules:
        print(f"SKIP {slug} (specific concept unavailable)")
    try:
        apply_plan(plans, apply=bool(args.apply))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    routed = sum(len(plan.paper_stems) for plan in routing.specific_plans)
    suppressed = sum(len(papers) for papers in routing.suppressed_papers_by_parent.values())
    print(f"concept split routed: {routed} specific links, {suppressed} parent links suppressed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
