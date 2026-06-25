#!/usr/bin/env python3
"""Read-only concept graph audit report.

The report is the canonical measurement surface for the paper<->concept graph:
connectivity, concept overlap, community modularity, split-rule effectiveness,
and remaining overlap candidates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import igraph as ig
import leidenalg

from scripts.ingest import concept_routing

DEFAULT_LINKGRAPH_DB = Path(".graph_index/linkgraph.sqlite")
DEFAULT_COMMUNITIES_DB = Path(".graph_index/communities.sqlite")
DEFAULT_RELATED_DB = Path(".graph_index/related.sqlite")
DEFAULT_VAULT = Path("wiki")
GRAPH_FOLDERS = ("research", "concepts")
SIMULATION_ISOLATION_LIMIT_PERCENT = 12


def _connect(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _folder(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else ""


def _round(value: float) -> float:
    return round(value, 2)


def _avg(values: list[int]) -> float:
    return _round(sum(values) / len(values)) if values else 0.0


def _percent(part: int, whole: int) -> int:
    return (100 * part // whole) if whole else 0


def _load_notes(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row["path"]): str(row["title"] or row["path"])
        for row in conn.execute("SELECT path, title FROM notes")
        if _folder(str(row["path"])) in GRAPH_FOLDERS
    }


def _load_resolved_links(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    return [
        (str(row["src_path"]), str(row["dst_target"]))
        for row in conn.execute(
            "SELECT src_path, dst_target FROM links WHERE dst_resolved = 1"
        )
    ]


def _concept_link_maps(
    links: list[tuple[str, str]], nodes: set[str]
) -> tuple[Counter[str], dict[str, set[str]], list[tuple[str, str]]]:
    rows: Counter[str] = Counter()
    sets: dict[str, set[str]] = defaultdict(set)
    pairs: list[tuple[str, str]] = []
    for src, dst in links:
        if src not in nodes or dst not in nodes or src == dst:
            continue
        if src.startswith("concepts/") and dst.startswith("research/"):
            rows[dst] += 1
            sets[dst].add(src)
            pairs.append((src, dst))
    return rows, sets, pairs


def _isolation(nodes: set[str], links: list[tuple[str, str]]) -> set[str]:
    connected: set[str] = set()
    for src, dst in links:
        if src in nodes and dst in nodes and src != dst:
            connected.add(src)
            connected.add(dst)
    return nodes - connected


def _undirected_pairs(
    links: list[tuple[str, str]],
    nodes: set[str],
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for src, dst in links:
        if src in nodes and dst in nodes and src != dst:
            pairs.add((src, dst) if src < dst else (dst, src))
    return pairs


def _sorted_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _isolation_from_pairs(nodes: set[str], pairs: set[tuple[str, str]]) -> set[str]:
    connected: set[str] = set()
    for left, right in pairs:
        if left in nodes and right in nodes and left != right:
            connected.add(left)
            connected.add(right)
    return nodes - connected


def _related_pairs(related_db: Path, nodes: set[str]) -> set[tuple[str, str]]:
    conn = _connect(related_db)
    if conn is None:
        return set()
    try:
        rows = conn.execute("SELECT src_path, dst_path FROM related").fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        src, dst = str(row["src_path"]), str(row["dst_path"])
        if src in nodes and dst in nodes and src != dst:
            pairs.add((src, dst) if src < dst else (dst, src))
    return pairs


def _modularity(nodes: list[str], pairs: set[tuple[str, str]]) -> dict[str, int | float]:
    if not nodes:
        return {"communities": 0, "modularity": 0.0}
    if not pairs:
        return {"communities": len(nodes), "modularity": 0.0}
    index = {path: i for i, path in enumerate(nodes)}
    edges = [(index[src], index[dst]) for src, dst in pairs if src in index and dst in index]
    if not edges:
        return {"communities": len(nodes), "modularity": 0.0}
    graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    part = leidenalg.find_partition(
        graph,
        leidenalg.ModularityVertexPartition,
        seed=42,
    )
    return {"communities": len(part), "modularity": round(float(part.modularity), 4)}


def _community_db_stats(communities_db: Path) -> dict[str, int]:
    conn = _connect(communities_db)
    if conn is None:
        return {"communities_db_count": 0, "largest_community": 0}
    try:
        rows = conn.execute("SELECT community FROM community").fetchall()
    except sqlite3.OperationalError:
        return {"communities_db_count": 0, "largest_community": 0}
    finally:
        conn.close()
    counts = Counter(int(row["community"]) for row in rows)
    return {
        "communities_db_count": len(counts),
        "largest_community": max(counts.values()) if counts else 0,
    }


def _tag_papers(conn: sqlite3.Connection) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    try:
        rows = conn.execute(
            "SELECT tag, note_path FROM tags WHERE note_path LIKE 'research/%'"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    for row in rows:
        tag = str(row["tag"])
        if tag.casefold() in concept_routing.SKIPPED_TAGS:
            continue
        grouped[tag].add(str(row["note_path"]))
    return dict(grouped)


def _top_concepts(concept_sets: dict[str, set[str]], limit: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for concepts in concept_sets.values():
        counts.update(concepts)
    return [
        {
            "concept": concept,
            "title": Path(concept).stem.replace("_", " "),
            "papers": count,
        }
        for concept, count in counts.most_common(limit)
    ]


def _top_papers(
    concept_rows: Counter[str],
    concept_sets: dict[str, set[str]],
    titles: dict[str, str],
    limit: int,
) -> list[dict[str, Any]]:
    rows = sorted(
        (
            (len(concepts), concept_rows.get(path, 0), path, concepts)
            for path, concepts in concept_sets.items()
        ),
        key=lambda item: (-item[0], -item[1], item[2]),
    )
    return [
        {
            "path": path,
            "title": titles.get(path, path),
            "unique_concepts": unique_count,
            "row_concepts": row_count,
            "concepts": sorted(Path(concept).stem for concept in concepts),
        }
        for unique_count, row_count, path, concepts in rows[:limit]
    ]


def _normalized_label_keys(labels: tuple[str, ...]) -> set[str]:
    return {
        variant
        for label in labels
        for variant in concept_routing.normalized_variants(label)
    }


def _rule_required_key_groups(rule: concept_routing.SplitRule) -> list[set[str]]:
    groups: list[set[str]] = []
    for index, tag in enumerate(rule.required_tags):
        aliases = rule.required_tag_aliases[index] if index < len(rule.required_tag_aliases) else ()
        groups.append(_normalized_label_keys((tag, *aliases)))
    return groups


def _rule_parent_keys(rule: concept_routing.SplitRule) -> set[str]:
    required = tuple(label for group in _rule_required_key_groups(rule) for label in group)
    suppressed = _normalized_label_keys(rule.suppressed_parent_slugs)
    return set(required) | suppressed


def _rule_target_keys(rule: concept_routing.SplitRule) -> set[str]:
    return _normalized_label_keys((rule.target_title, rule.target_slug, *rule.aliases))


def make_simulated_split_rule(
    target_title: str,
    *,
    required_tags: tuple[str, ...],
    suppressed_parent_slugs: tuple[str, ...] | None = None,
    aliases: tuple[str, ...] = (),
    min_papers: int = 1,
) -> concept_routing.SplitRule:
    """Construct an in-memory split rule for audit simulation only."""
    return concept_routing.SplitRule(
        target_title=target_title,
        target_slug=concept_routing.slugify_tag(target_title),
        required_tags=required_tags,
        suppressed_parent_slugs=suppressed_parent_slugs or required_tags,
        aliases=aliases,
        definition=f"Simulated split rule for {target_title}.",
        min_papers=min_papers,
    )


def _covered_rule_pairs() -> set[frozenset[str]]:
    covered: set[frozenset[str]] = set()
    for rule in concept_routing.SPLIT_RULES:
        required_groups = _rule_required_key_groups(rule)
        for left, right in itertools.combinations(required_groups, 2):
            for pair in itertools.product(left, right):
                covered.add(frozenset(pair))
        for target_key, parent_key in itertools.product(_rule_target_keys(rule), _rule_parent_keys(rule)):
            if target_key != parent_key:
                covered.add(frozenset((target_key, parent_key)))
    return covered


def _candidate_pairs(
    tags_by_paper: dict[str, set[str]],
    titles: dict[str, str],
    *,
    limit: int,
    sample_limit: int,
    min_count: int,
) -> list[dict[str, Any]]:
    covered = _covered_rule_pairs()
    pair_papers: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_labels: dict[tuple[str, str], tuple[str, str]] = {}

    for paper, tags in tags_by_paper.items():
        labels = sorted(tags, key=str.casefold)
        for left, right in itertools.combinations(labels, 2):
            left_key = concept_routing.canonical_key(left)
            right_key = concept_routing.canonical_key(right)
            if not left_key or not right_key or left_key == right_key:
                continue
            key_pair = tuple(sorted((left_key, right_key)))
            if frozenset(key_pair) in covered:
                continue
            pair_papers[key_pair].add(paper)
            pair_labels.setdefault(key_pair, tuple(sorted((left, right), key=str.casefold)))

    ranked = sorted(
        (
            (len(papers), key_pair, pair_labels[key_pair], sorted(papers))
            for key_pair, papers in pair_papers.items()
            if len(papers) >= min_count
        ),
        key=lambda item: (-item[0], item[2][0].casefold(), item[2][1].casefold()),
    )
    candidates: list[dict[str, Any]] = []
    for count, key_pair, labels, papers in ranked[:limit]:
        samples = [
            {"path": path, "title": titles.get(path, path)}
            for path in papers[:sample_limit]
        ]
        candidates.append(
            {
                "tags": list(labels),
                "normalized_tags": list(key_pair),
                "count": count,
                "samples": samples,
            }
        )
    return candidates


def _tags_by_paper(tag_papers: dict[str, set[str]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for tag, papers in tag_papers.items():
        for paper in papers:
            grouped[paper].add(tag)
    return dict(grouped)


def _concept_rel_path(label: str, concept_index: dict[str, concept_routing.ExistingConcept]) -> str:
    matched = concept_routing.find_concept(label, concept_index)
    if matched is not None:
        return matched.rel_path
    return f"concepts/{concept_routing.slugify_tag(label)}.md"


def _papers_matching_rule(
    tags_by_paper: dict[str, set[str]],
    rule: concept_routing.SplitRule,
) -> set[str]:
    required_groups = _rule_required_key_groups(rule)
    matching: set[str] = set()
    for paper, tags in tags_by_paper.items():
        paper_keys = _normalized_label_keys(tuple(tags))
        if all(paper_keys.intersection(group) for group in required_groups):
            matching.add(paper)
    return matching


def _simulation_snapshot(
    *,
    nodes: set[str],
    research: set[str],
    concept_sets: dict[str, set[str]],
    link_pairs: set[tuple[str, str]],
    related_pairs: set[tuple[str, str]],
) -> dict[str, int | float]:
    unique_values = [len(concepts) for concepts in concept_sets.values() if concepts]
    isolated = _isolation_from_pairs(nodes, link_pairs)
    wikilink_modularity = _modularity(sorted(nodes), link_pairs)
    overlay_modularity = _modularity(sorted(nodes), link_pairs | related_pairs)
    return {
        "nodes": len(nodes),
        "research": len(research),
        "concept_linked": sum(1 for paper in research if concept_sets.get(paper)),
        "zero_concept": sum(1 for paper in research if not concept_sets.get(paper)),
        "isolated": len(isolated),
        "isolation_percent": _percent(len(isolated), len(nodes)),
        "unique_avg_concepts_per_paper": _avg(unique_values),
        "unique_multi_concept_percent": _percent(
            sum(1 for value in unique_values if value >= 2),
            len(unique_values),
        ),
        "wikilink_modularity": wikilink_modularity["modularity"],
        "overlay_modularity": overlay_modularity["modularity"],
    }


def _simulation_delta(
    before: dict[str, int | float],
    after: dict[str, int | float],
) -> dict[str, int | float]:
    delta: dict[str, int | float] = {}
    for key, before_value in before.items():
        after_value = after[key]
        if isinstance(before_value, int) and isinstance(after_value, int):
            delta[key] = after_value - before_value
        else:
            precision = 4 if key.endswith("modularity") else 2
            delta[key] = round(float(after_value) - float(before_value), precision)
    return delta


def _simulation_decision(row: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    blockers: list[str] = []
    delta = row["delta"]
    after = row["after"]

    if row["routed_paper_count"] <= 0:
        blockers.append("no_routed_papers")
    if float(delta["unique_avg_concepts_per_paper"]) <= 0:
        reasons.append("avg_concepts_non_increasing")
    else:
        blockers.append("avg_concepts_increased")
    if int(after["isolation_percent"]) < SIMULATION_ISOLATION_LIMIT_PERCENT:
        reasons.append("isolation_below_limit")
    else:
        blockers.append("isolation_at_or_above_limit")
    if (
        float(delta["wikilink_modularity"]) >= 0
        and float(delta["overlay_modularity"]) >= 0
    ):
        reasons.append("modularity_not_worse")
    else:
        blockers.append("modularity_worse")
    if row["suppressed_parent_count"] > 0:
        reasons.append("parents_suppressed")
    else:
        blockers.append("no_parent_suppression")

    if blockers:
        return "hold_candidate", blockers
    return "promote_candidate", reasons


def _simulate_split_rules(
    *,
    vault_dir: Path,
    rules: list[concept_routing.SplitRule] | tuple[concept_routing.SplitRule, ...],
    nodes: set[str],
    research: set[str],
    links: list[tuple[str, str]],
    concept_sets: dict[str, set[str]],
    tags_by_paper: dict[str, set[str]],
    related_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    if not rules:
        return []

    concept_index = concept_routing.load_concept_index(vault_dir)
    base_pairs = _undirected_pairs(links, nodes)
    before = _simulation_snapshot(
        nodes=nodes,
        research=research,
        concept_sets=concept_sets,
        link_pairs=base_pairs,
        related_pairs=related_pairs,
    )
    rows: list[dict[str, Any]] = []
    for rule in rules:
        qualifying = _papers_matching_rule(tags_by_paper, rule)
        routed = qualifying if len(qualifying) >= rule.min_papers else set()
        matched = concept_routing.find_concept(rule.target_slug, concept_index)
        target = matched.rel_path if matched else f"concepts/{rule.target_slug}.md"
        simulated_nodes = set(nodes)
        simulated_sets = {paper: set(concepts) for paper, concepts in concept_sets.items()}
        simulated_pairs = set(base_pairs)
        suppressed_counts: Counter[str] = Counter()

        if routed:
            simulated_nodes.add(target)
            for paper in routed:
                if paper not in research:
                    continue
                current = simulated_sets.setdefault(paper, set())
                for parent_slug in rule.suppressed_parent_slugs:
                    parent = _concept_rel_path(parent_slug, concept_index)
                    if parent in current:
                        current.remove(parent)
                        suppressed_counts[parent] += 1
                    simulated_pairs.discard(_sorted_pair(parent, paper))
                current.add(target)
                simulated_pairs.add(_sorted_pair(target, paper))

        simulated_sets = {
            paper: concepts for paper, concepts in simulated_sets.items() if concepts
        }
        after = _simulation_snapshot(
            nodes=simulated_nodes,
            research=research,
            concept_sets=simulated_sets,
            link_pairs=simulated_pairs,
            related_pairs=related_pairs,
        )
        row = {
            "target_slug": rule.target_slug,
            "target": target,
            "target_reused_or_created": "reused" if matched else "would_create",
            "source_tags": list(rule.required_tags),
            "required_tags": list(rule.required_tags),
            "suppressed_parent_slugs": list(rule.suppressed_parent_slugs),
            "qualifying_count": len(qualifying),
            "routed_paper_count": len(routed),
            "suppressed_parent_count": sum(suppressed_counts.values()),
            "suppressed_parent_counts": dict(sorted(suppressed_counts.items())),
            "before": before,
            "after": after,
            "delta": _simulation_delta(before, after),
        }
        row["decision"], row["decision_reasons"] = _simulation_decision(row)
        rows.append(row)
    return rows


def _candidate_rule(candidate: dict[str, Any]) -> concept_routing.SplitRule:
    tags = tuple(str(tag) for tag in candidate["tags"])
    return make_simulated_split_rule(
        " ".join(tags),
        required_tags=tags,
        suppressed_parent_slugs=tags,
        min_papers=int(candidate["count"]),
    )


def _simulate_candidate_pairs(
    *,
    vault_dir: Path,
    candidates: list[dict[str, Any]],
    nodes: set[str],
    research: set[str],
    links: list[tuple[str, str]],
    concept_sets: dict[str, set[str]],
    tags_by_paper: dict[str, set[str]],
    related_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    simulations = _simulate_split_rules(
        vault_dir=vault_dir,
        rules=[_candidate_rule(candidate) for candidate in candidates],
        nodes=nodes,
        research=research,
        links=links,
        concept_sets=concept_sets,
        tags_by_paper=tags_by_paper,
        related_pairs=related_pairs,
    )
    rows: list[dict[str, Any]] = []
    for candidate, simulation in zip(candidates, simulations, strict=True):
        rows.append(
            {
                **simulation,
                "source_tags": list(candidate["tags"]),
                "normalized_tags": list(candidate["normalized_tags"]),
                "candidate_count": candidate["count"],
                "samples": candidate["samples"],
            }
        )
    return rows


def _rule_effectiveness(
    vault_dir: Path,
    tag_papers: list[concept_routing.TagPapers],
) -> list[dict[str, Any]]:
    concept_index = concept_routing.load_concept_index(vault_dir)
    rows: list[dict[str, Any]] = []
    for rule in concept_routing.SPLIT_RULES:
        routing = concept_routing.route_papers_to_concepts(
            vault_dir,
            tag_papers,
            rules=[rule],
        )
        plan = routing.specific_plans[0] if routing.specific_plans else None
        if plan is None:
            status = "skipped" if rule.target_slug in routing.skipped_rules else "not_qualified"
            routed = 0
        else:
            status = "created" if plan.create else "reused"
            routed = len(plan.paper_stems)
        matched = concept_routing.find_concept(rule.target_slug, concept_index)
        rel_target = matched.rel_path if matched else f"concepts/{rule.target_slug}.md"
        rows.append(
            {
                "target_slug": rule.target_slug,
                "target": rel_target,
                "required_tags": list(rule.required_tags),
                "suppressed_parent_slugs": list(rule.suppressed_parent_slugs),
                "qualifying_count": routing.qualifying_counts.get(rule.target_slug, 0),
                "routed_count": routed,
                "target_reused_or_created": status,
                "suppressed_parent_counts": {
                    parent: len(papers)
                    for parent, papers in sorted(routing.suppressed_papers_by_parent.items())
                },
            }
        )
    return rows


def build_audit_report(
    *,
    vault_dir: Path = DEFAULT_VAULT,
    linkgraph_db: Path = DEFAULT_LINKGRAPH_DB,
    communities_db: Path = DEFAULT_COMMUNITIES_DB,
    related_db: Path = DEFAULT_RELATED_DB,
    candidate_limit: int = 12,
    sample_limit: int = 5,
    candidate_min_count: int = 8,
    simulated_rules: list[concept_routing.SplitRule]
    | tuple[concept_routing.SplitRule, ...]
    | None = None,
    simulate_candidates: bool = False,
) -> dict[str, Any]:
    conn = _connect(linkgraph_db)
    if conn is None:
        raise FileNotFoundError(f"linkgraph database not found: {linkgraph_db}")
    try:
        titles = _load_notes(conn)
        links = _load_resolved_links(conn)
        tag_sets = _tag_papers(conn)
    finally:
        conn.close()

    nodes = set(titles)
    research = {path for path in nodes if path.startswith("research/")}
    concept_rows, concept_sets, concept_pairs = _concept_link_maps(links, nodes)
    zero_concept = research - set(concept_sets)
    isolated = _isolation(nodes, links)
    row_values = list(concept_rows.values())
    unique_values = [len(concepts) for concepts in concept_sets.values()]
    duplicate_rows = len(concept_pairs) - len(set(concept_pairs))

    link_pairs = _undirected_pairs(links, nodes)
    related_pairs = _related_pairs(related_db, nodes)
    wikilink_modularity = _modularity(sorted(nodes), link_pairs)
    overlay_modularity = _modularity(sorted(nodes), link_pairs | related_pairs)
    community_db = _community_db_stats(communities_db)

    tag_papers = [
        concept_routing.TagPapers(tag=tag, papers=set(papers))
        for tag, papers in sorted(tag_sets.items(), key=lambda item: item[0].casefold())
    ]
    tags_by_paper = _tags_by_paper(tag_sets)
    uncovered_pairs = _candidate_pairs(
        tags_by_paper,
        titles,
        limit=candidate_limit,
        sample_limit=sample_limit,
        min_count=candidate_min_count,
    )
    candidate_simulations = (
        _simulate_candidate_pairs(
            vault_dir=vault_dir,
            candidates=uncovered_pairs,
            nodes=nodes,
            research=research,
            links=links,
            concept_sets=concept_sets,
            tags_by_paper=tags_by_paper,
            related_pairs=related_pairs,
        )
        if simulate_candidates
        else []
    )
    if candidate_simulations:
        uncovered_pairs = [
            {
                **candidate,
                "simulation_decision": simulation["decision"],
                "simulation_reasons": simulation["decision_reasons"],
            }
            for candidate, simulation in zip(
                uncovered_pairs,
                candidate_simulations,
                strict=True,
            )
        ]

    return {
        "summary": {
            "nodes": len(nodes),
            "research": len(research),
            "concept_linked": len(concept_sets),
            "zero_concept": len(zero_concept),
            "isolated": len(isolated),
            "isolation_percent": _percent(len(isolated), len(nodes)),
        },
        "overlap": {
            "concept_paper_rows": len(concept_pairs),
            "unique_edges": len(set(concept_pairs)),
            "duplicate_rows": duplicate_rows,
            "row_avg_concepts_per_paper": _avg(row_values),
            "unique_avg_concepts_per_paper": _avg(unique_values),
            "row_multi_concept_percent": _percent(sum(1 for value in row_values if value >= 2), len(row_values)),
            "unique_multi_concept_percent": _percent(
                sum(1 for value in unique_values if value >= 2),
                len(unique_values),
            ),
        },
        "communities": {
            "wikilink_communities": wikilink_modularity["communities"],
            "wikilink_modularity": wikilink_modularity["modularity"],
            "overlay_communities": overlay_modularity["communities"],
            "overlay_modularity": overlay_modularity["modularity"],
            **community_db,
        },
        "rules": _rule_effectiveness(vault_dir, tag_papers),
        "simulations": _simulate_split_rules(
            vault_dir=vault_dir,
            rules=simulated_rules or [],
            nodes=nodes,
            research=research,
            links=links,
            concept_sets=concept_sets,
            tags_by_paper=tags_by_paper,
            related_pairs=related_pairs,
        ),
        "candidate_simulations": candidate_simulations,
        "hotspots": {
            "top_concepts": _top_concepts(concept_sets, candidate_limit),
            "top_overlapping_papers": _top_papers(
                concept_rows,
                concept_sets,
                titles,
                candidate_limit,
            ),
            "uncovered_tag_pairs": uncovered_pairs,
        },
    }


def _format_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    overlap = report["overlap"]
    communities = report["communities"]
    return "\n".join(
        [
            (
                f"nodes={summary['nodes']} isolated={summary['isolated']} "
                f"({summary['isolation_percent']}%)"
            ),
            (
                f"research={summary['research']} concept_linked={summary['concept_linked']} "
                f"zero_concept={summary['zero_concept']}"
            ),
            (
                f"rows={overlap['concept_paper_rows']} unique_edges={overlap['unique_edges']} "
                f"duplicate_rows={overlap['duplicate_rows']}"
            ),
            (
                f"row_avg={overlap['row_avg_concepts_per_paper']:.2f} "
                f"unique_avg={overlap['unique_avg_concepts_per_paper']:.2f}"
            ),
            (
                f"wikilink_modularity={communities['wikilink_modularity']:.4f} "
                f"overlay_modularity={communities['overlay_modularity']:.4f}"
            ),
        ]
    )


def _format_markdown(report: dict[str, Any]) -> str:
    lines = ["# Concept Graph Audit", "", "## Summary", "", "```text", _format_text(report), "```"]
    if report.get("simulations"):
        lines.extend(["", "## Split Simulations", ""])
        for item in report["simulations"]:
            delta = item["delta"]
            lines.append(
                "- "
                f"{item['target_slug']}: routed={item['routed_paper_count']} "
                f"suppressed={item['suppressed_parent_count']} "
                f"avg_delta={delta['unique_avg_concepts_per_paper']} "
                f"isolation_delta={delta['isolation_percent']} "
                f"modularity_delta={delta['wikilink_modularity']}"
            )
    if report.get("candidate_simulations"):
        lines.extend(["", "## Candidate Split Simulations", ""])
        for item in report["candidate_simulations"]:
            delta = item["delta"]
            tags = " + ".join(item["source_tags"])
            reasons = ", ".join(item["decision_reasons"])
            lines.append(
                "- "
                f"{tags}: {item['decision']} "
                f"routed={item['routed_paper_count']} "
                f"suppressed={item['suppressed_parent_count']} "
                f"avg_delta={delta['unique_avg_concepts_per_paper']} "
                f"isolation_delta={delta['isolation_percent']} "
                f"modularity_delta={delta['wikilink_modularity']} "
                f"reasons={reasons}"
            )
    lines.extend(["", "## Top Overlapping Papers", ""])
    for item in report["hotspots"]["top_overlapping_papers"][:10]:
        lines.append(
            f"- `{item['path']}`: {item['unique_concepts']} concepts — {item['title']}"
        )
    lines.extend(["", "## Uncovered Tag Pairs", ""])
    for item in report["hotspots"]["uncovered_tag_pairs"][:10]:
        tags = " + ".join(item["tags"])
        lines.append(f"- {tags}: {item['count']} papers")
    return "\n".join(lines) + "\n"


def _split_rule_labels(value: str) -> tuple[str, ...]:
    labels = tuple(label.strip() for label in value.split("+") if label.strip())
    if not labels:
        raise argparse.ArgumentTypeError("split rule labels cannot be empty")
    return labels


def parse_simulated_split_rule(value: str) -> concept_routing.SplitRule:
    try:
        target, rule_spec = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected TARGET=TAG+TAG or TARGET=TAG+TAG:PARENT+PARENT"
        ) from exc
    target = target.strip()
    if not target:
        raise argparse.ArgumentTypeError("split rule target cannot be empty")
    required_spec, _, parent_spec = rule_spec.partition(":")
    required_tags = _split_rule_labels(required_spec)
    suppressed = _split_rule_labels(parent_spec) if parent_spec.strip() else required_tags
    return make_simulated_split_rule(
        target,
        required_tags=required_tags,
        suppressed_parent_slugs=suppressed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--linkgraph-db", type=Path, default=DEFAULT_LINKGRAPH_DB)
    parser.add_argument("--communities-db", type=Path, default=DEFAULT_COMMUNITIES_DB)
    parser.add_argument("--related-db", type=Path, default=DEFAULT_RELATED_DB)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--candidate-min-count", type=int, default=8)
    parser.add_argument(
        "--simulate-split",
        action="append",
        default=[],
        type=parse_simulated_split_rule,
        metavar="TARGET=TAG+TAG[:PARENT+PARENT]",
        help="Evaluate a temporary split rule without changing routing rules.",
    )
    parser.add_argument(
        "--simulate-candidates",
        action="store_true",
        help="Evaluate uncovered tag-pair candidates through the split simulator.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_audit_report(
        vault_dir=args.vault,
        linkgraph_db=args.linkgraph_db,
        communities_db=args.communities_db,
        related_db=args.related_db,
        candidate_limit=args.candidate_limit,
        sample_limit=args.sample_limit,
        candidate_min_count=args.candidate_min_count,
        simulated_rules=args.simulate_split,
        simulate_candidates=args.simulate_candidates,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.markdown:
        print(_format_markdown(report), end="")
    else:
        print(_format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
