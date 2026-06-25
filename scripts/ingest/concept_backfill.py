#!/usr/bin/env python3
"""Backfill concept notes from high-frequency research-paper tags.

The script is intentionally vault-mutating only with ``--apply``. By default it
prints the concept notes or auto sections it would create/update.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.ingest import concept_routing
from scripts.ingest.concept_routing import (
    AUTO_END,
    AUTO_HEADING,
    AUTO_NOTE,
    AUTO_START,
    DEFAULT_DB,
    DEFAULT_VAULT,
    DEFINITIONS,
    SKIPPED_TAGS,
    ConceptPlan,
    ExistingConcept,
    TagPapers,
    build_plan,
    canonical_key,
    definition_for,
    find_concept,
    load_concept_index,
    load_qualifying_tags,
    normalized_variants,
    render_auto_section,
    render_new_concept,
    replace_auto_section,
    singular_display_title,
    slugify_tag,
)


def apply_plan(plans: list[ConceptPlan], *, apply: bool) -> int:
    return concept_routing.apply_plan(plans, apply=apply, label="concept backfill")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--min-papers", type=int, default=20)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print planned changes without writing")
    mode.add_argument("--apply", action="store_true", help="Write concept notes and auto sections")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_papers < 1:
        print("--min-papers must be >= 1", file=sys.stderr)
        return 2
    if not args.db.exists():
        print(f"linkgraph database not found: {args.db}", file=sys.stderr)
        return 2
    if not args.vault.exists():
        print(f"vault directory not found: {args.vault}", file=sys.stderr)
        return 2

    all_tag_papers = concept_routing.load_tag_papers(args.db)
    routing = concept_routing.route_papers_to_concepts(args.vault, all_tag_papers)
    qualifying_tags = [item for item in all_tag_papers if len(item.papers) >= args.min_papers]
    result = concept_routing.plan_tag_backfill(
        args.vault,
        qualifying_tags,
        suppressed_papers_by_parent=routing.suppressed_papers_by_parent,
        skip_rel_paths={plan.rel_path for plan in routing.specific_plans},
        skip_missing_definitions=True,
    )
    for tag in result.skipped_tags:
        print(f"SKIP {tag} (missing curated definition)")
    try:
        apply_plan(result.plans, apply=bool(args.apply))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


__all__ = [
    "AUTO_END",
    "AUTO_HEADING",
    "AUTO_NOTE",
    "AUTO_START",
    "DEFAULT_DB",
    "DEFAULT_VAULT",
    "DEFINITIONS",
    "SKIPPED_TAGS",
    "ConceptPlan",
    "ExistingConcept",
    "TagPapers",
    "apply_plan",
    "build_plan",
    "canonical_key",
    "definition_for",
    "find_concept",
    "load_concept_index",
    "load_qualifying_tags",
    "main",
    "normalized_variants",
    "parse_args",
    "render_auto_section",
    "render_new_concept",
    "replace_auto_section",
    "singular_display_title",
    "slugify_tag",
]


if __name__ == "__main__":
    raise SystemExit(main())
