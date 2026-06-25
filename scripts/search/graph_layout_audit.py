#!/usr/bin/env python3
"""Read-only baseline for the vault UI graph layout inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_LINKGRAPH_DB = Path(".graph_index/linkgraph.sqlite")
DEFAULT_COMMUNITIES_DB = Path(".graph_index/communities.sqlite")
DEFAULT_RELATED_DB = Path(".graph_index/related.sqlite")
GRAPH_FOLDERS = frozenset({"research", "concepts"})
DEFAULT_QUALITY_THRESHOLDS = {
    "max_zero_degree_with_similar_percent": 5,
    "max_largest_community_percent": 35,
    "max_top_12_coverage_percent": 96,
}


def _connect(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _folder(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else ""


def _percent(part: int, whole: int) -> int:
    return (100 * part // whole) if whole else 0


def _sorted_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _unit_hash(value: str, *, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _failure(
    *,
    metric: str,
    actual: int,
    limit: int,
    reason: str,
) -> dict[str, int | str]:
    return {
        "metric": metric,
        "actual": actual,
        "limit": limit,
        "reason": reason,
    }


def _load_graph_nodes(linkgraph_db: Path) -> dict[str, str]:
    conn = _connect(linkgraph_db)
    if conn is None:
        return {}
    try:
        rows = conn.execute("SELECT path, title FROM notes").fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    return {
        str(row["path"]): str(row["title"] or row["path"])
        for row in rows
        if _folder(str(row["path"])) in GRAPH_FOLDERS
    }


def _explicit_pairs(linkgraph_db: Path, nodes: set[str]) -> set[tuple[str, str]]:
    conn = _connect(linkgraph_db)
    if conn is None:
        return set()
    try:
        rows = conn.execute(
            "SELECT src_path, dst_target FROM links WHERE dst_resolved = 1"
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()
    return {
        _sorted_pair(str(row["src_path"]), str(row["dst_target"]))
        for row in rows
        if str(row["src_path"]) in nodes
        and str(row["dst_target"]) in nodes
        and str(row["src_path"]) != str(row["dst_target"])
    }


def _similar_pairs(
    related_db: Path,
    nodes: set[str],
    *,
    top_k: int,
) -> set[tuple[str, str]]:
    conn = _connect(related_db)
    if conn is None:
        return set()
    try:
        rows = conn.execute("SELECT src_path, dst_path FROM related ORDER BY score DESC").fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()
    kept_per_src: dict[str, int] = {}
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        src, dst = str(row["src_path"]), str(row["dst_path"])
        if src not in nodes or dst not in nodes or src == dst:
            continue
        if kept_per_src.get(src, 0) >= top_k:
            continue
        kept_per_src[src] = kept_per_src.get(src, 0) + 1
        pairs.add(_sorted_pair(src, dst))
    return pairs


def _degree(nodes: set[str], pairs: set[tuple[str, str]]) -> dict[str, int]:
    degree = {node: 0 for node in nodes}
    for src, dst in pairs:
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1
    return degree


def _community_stats(communities_db: Path, nodes: set[str]) -> dict[str, int]:
    conn = _connect(communities_db)
    if conn is None:
        return {
            "count": 0,
            "largest": 0,
            "top_12_nodes": 0,
            "top_12_coverage_percent": 0,
            "long_tail_nodes": len(nodes),
        }
    try:
        rows = conn.execute("SELECT path, community FROM community").fetchall()
    except sqlite3.OperationalError:
        return {
            "count": 0,
            "largest": 0,
            "top_12_nodes": 0,
            "top_12_coverage_percent": 0,
            "long_tail_nodes": len(nodes),
        }
    finally:
        conn.close()
    counts = Counter(int(row["community"]) for row in rows if str(row["path"]) in nodes)
    top_12_nodes = sum(count for _, count in counts.most_common(12))
    return {
        "count": len(counts),
        "largest": max(counts.values()) if counts else 0,
        "top_12_nodes": top_12_nodes,
        "top_12_coverage_percent": _percent(top_12_nodes, len(nodes)),
        "long_tail_nodes": len(nodes) - top_12_nodes,
    }


def _community_assignments(communities_db: Path, nodes: set[str]) -> dict[str, int]:
    conn = _connect(communities_db)
    if conn is None:
        return {}
    try:
        rows = conn.execute("SELECT path, community FROM community").fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    return {str(row["path"]): int(row["community"]) for row in rows if str(row["path"]) in nodes}


def evaluate_quality_gate(
    report: dict[str, Any],
    *,
    thresholds: dict[str, int] | None = None,
) -> dict[str, Any]:
    active_thresholds = dict(DEFAULT_QUALITY_THRESHOLDS)
    if thresholds:
        active_thresholds.update(thresholds)

    summary = report["summary"]
    communities = report["communities"]
    nodes = int(summary["nodes"])
    zero_with_similar_percent = _percent(
        int(summary["zero_degree_with_similar"]),
        nodes,
    )
    largest_community_percent = _percent(int(communities["largest"]), nodes)
    top_12_coverage_percent = int(communities["top_12_coverage_percent"])
    failures: list[dict[str, int | str]] = []

    zero_limit = active_thresholds["max_zero_degree_with_similar_percent"]
    if zero_with_similar_percent > zero_limit:
        failures.append(
            _failure(
                metric="zero_degree_with_similar_percent",
                actual=zero_with_similar_percent,
                limit=zero_limit,
                reason="too_many_drawn_isolates",
            )
        )

    largest_limit = active_thresholds["max_largest_community_percent"]
    if largest_community_percent > largest_limit:
        failures.append(
            _failure(
                metric="largest_community_percent",
                actual=largest_community_percent,
                limit=largest_limit,
                reason="largest_community_too_dominant",
            )
        )

    top_12_limit = active_thresholds["max_top_12_coverage_percent"]
    if top_12_coverage_percent > top_12_limit:
        failures.append(
            _failure(
                metric="top_12_coverage_percent",
                actual=top_12_coverage_percent,
                limit=top_12_limit,
                reason="community_anchors_absorb_too_much_graph",
            )
        )

    return {
        "passed": not failures,
        "thresholds": active_thresholds,
        "measurements": {
            "zero_degree_with_similar_percent": zero_with_similar_percent,
            "largest_community_percent": largest_community_percent,
            "top_12_coverage_percent": top_12_coverage_percent,
        },
        "failures": failures,
    }


def _community_centers(
    paths: set[str],
    communities: dict[str, int],
) -> dict[int | None, tuple[float, float]]:
    community_sizes: Counter[int | None] = Counter(communities.get(path) for path in paths)
    ranked = sorted(
        community_sizes,
        key=lambda community: (-community_sizes[community], -1 if community is None else community),
    )
    graph_radius = 24 * math.sqrt(len(paths))
    centers: dict[int | None, tuple[float, float]] = {}
    for index, community in enumerate(ranked):
        angle = (index * math.pi * (3 - math.sqrt(5))) % (2 * math.pi)
        centers[community] = (
            math.cos(angle) * graph_radius,
            math.sin(angle) * graph_radius,
        )
    return centers


def _seeded_community_positions(
    paths: set[str],
    communities: dict[str, int],
) -> dict[str, tuple[float, float]]:
    if not paths:
        return {}
    community_sizes: Counter[int | None] = Counter(communities.get(path) for path in paths)
    centers = _community_centers(paths, communities)
    positions: dict[str, tuple[float, float]] = {}
    for path in sorted(paths):
        community = communities.get(path)
        center_x, center_y = centers[community]
        spread = 8 + 5 * math.sqrt(community_sizes[community])
        angle = _unit_hash(path, salt="layout-benchmark-angle") * 2 * math.pi
        radius = spread * math.sqrt(_unit_hash(path, salt="layout-benchmark-radius"))
        positions[path] = (
            center_x + math.cos(angle) * radius,
            center_y + math.sin(angle) * radius,
        )
    return positions


def _force_refined_positions(
    paths: set[str],
    pairs: set[tuple[str, str]],
    communities: dict[str, int],
    *,
    iterations: int = 60,
) -> dict[str, tuple[float, float]]:
    positions = _seeded_community_positions(paths, communities)
    centers = _community_centers(paths, communities)
    ordered_paths = sorted(paths)
    filtered_pairs = [(src, dst) for src, dst in sorted(pairs) if src in paths and dst in paths]
    if not positions:
        return {}

    for step in range(iterations):
        displacements = {path: [0.0, 0.0] for path in ordered_paths}

        for left_index, left in enumerate(ordered_paths):
            left_x, left_y = positions[left]
            for right in ordered_paths[left_index + 1 :]:
                right_x, right_y = positions[right]
                delta_x = left_x - right_x
                delta_y = left_y - right_y
                distance_sq = max(delta_x * delta_x + delta_y * delta_y, 0.01)
                distance = math.sqrt(distance_sq)
                force = 120.0 / distance_sq
                unit_x = delta_x / distance
                unit_y = delta_y / distance
                displacements[left][0] += unit_x * force
                displacements[left][1] += unit_y * force
                displacements[right][0] -= unit_x * force
                displacements[right][1] -= unit_y * force

        for src, dst in filtered_pairs:
            src_x, src_y = positions[src]
            dst_x, dst_y = positions[dst]
            delta_x = dst_x - src_x
            delta_y = dst_y - src_y
            distance = max(math.hypot(delta_x, delta_y), 0.1)
            same_community = communities.get(src) == communities.get(dst)
            desired = 38.0 if same_community else 68.0
            force = (distance - desired) * 0.018
            unit_x = delta_x / distance
            unit_y = delta_y / distance
            displacements[src][0] += unit_x * force
            displacements[src][1] += unit_y * force
            displacements[dst][0] -= unit_x * force
            displacements[dst][1] -= unit_y * force

        for path in ordered_paths:
            x, y = positions[path]
            center_x, center_y = centers[communities.get(path)]
            displacements[path][0] += (center_x - x) * 0.012
            displacements[path][1] += (center_y - y) * 0.012
            displacements[path][0] += -x * 0.001
            displacements[path][1] += -y * 0.001

        temperature = max(1.0, 7.0 * (1 - step / iterations))
        for path in ordered_paths:
            move_x, move_y = displacements[path]
            move = max(math.hypot(move_x, move_y), 0.001)
            scale = min(move, temperature) / move
            x, y = positions[path]
            positions[path] = (x + move_x * scale, y + move_y * scale)

    return positions


def _layout_metrics(
    positions: dict[str, tuple[float, float]],
    pairs: set[tuple[str, str]],
    communities: dict[str, int],
) -> dict[str, float | int]:
    paths = set(positions)
    filtered_pairs = [(src, dst) for src, dst in sorted(pairs) if src in paths and dst in paths]
    edge_lengths = [_distance(positions[src], positions[dst]) for src, dst in filtered_pairs]

    grouped: dict[int | None, list[tuple[float, float]]] = {}
    for path, point in positions.items():
        grouped.setdefault(communities.get(path), []).append(point)
    community_centroids: dict[int | None, tuple[float, float]] = {}
    within_radii: list[float] = []
    for community, points in grouped.items():
        center = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        community_centroids[community] = center
        within_radii.extend(_distance(point, center) for point in points)

    center_distances: list[float] = []
    centers = list(community_centroids.values())
    for index, left in enumerate(centers):
        for right in centers[index + 1 :]:
            center_distances.append(_distance(left, right))

    close_pairs = 0
    total_pairs = 0
    ordered_paths = sorted(paths)
    for index, left in enumerate(ordered_paths):
        for right in ordered_paths[index + 1 :]:
            total_pairs += 1
            if _distance(positions[left], positions[right]) < 10.0:
                close_pairs += 1

    same_community_edges = sum(
        1
        for src, dst in filtered_pairs
        if communities.get(src) is not None and communities.get(src) == communities.get(dst)
    )
    mean_within_radius = sum(within_radii) / len(within_radii) if within_radii else 0.0
    mean_center_distance = (
        sum(center_distances) / len(center_distances) if center_distances else 0.0
    )
    return {
        "node_count": len(paths),
        "edge_count": len(filtered_pairs),
        "community_count": len(grouped),
        "median_edge_length": round(_median(edge_lengths), 2),
        "same_community_edge_percent": round(
            100 * same_community_edges / len(filtered_pairs),
            2,
        )
        if filtered_pairs
        else 0.0,
        "mean_within_community_radius": round(mean_within_radius, 2),
        "mean_community_center_distance": round(mean_center_distance, 2),
        "community_separation_ratio": round(mean_center_distance / mean_within_radius, 2)
        if mean_within_radius
        else 0.0,
        "overlap_risk_percent": round(100 * close_pairs / total_pairs, 2)
        if total_pairs
        else 0.0,
    }


def evaluate_layout_candidates(
    nodes: set[str],
    pairs: set[tuple[str, str]],
    communities: dict[str, int],
) -> dict[str, Any]:
    baseline_positions = _seeded_community_positions(nodes, communities)
    semantic_positions = _force_refined_positions(nodes, pairs, communities)
    baseline_metrics = _layout_metrics(baseline_positions, pairs, communities)
    semantic_metrics = _layout_metrics(semantic_positions, pairs, communities)
    delta = {
        "median_edge_length_delta": round(
            float(semantic_metrics["median_edge_length"])
            - float(baseline_metrics["median_edge_length"]),
            2,
        ),
        "community_separation_ratio_delta": round(
            float(semantic_metrics["community_separation_ratio"])
            - float(baseline_metrics["community_separation_ratio"]),
            2,
        ),
        "overlap_risk_percent_delta": round(
            float(semantic_metrics["overlap_risk_percent"])
            - float(baseline_metrics["overlap_risk_percent"]),
            2,
        ),
    }
    edge_compactness_improved = delta["median_edge_length_delta"] <= 0
    overlap_not_worse = delta["overlap_risk_percent_delta"] <= 0.5
    separation_preserved = (
        float(semantic_metrics["community_separation_ratio"])
        >= float(baseline_metrics["community_separation_ratio"]) * 0.85
    )
    if edge_compactness_improved and overlap_not_worse and separation_preserved:
        adoption = "eligible_for_browser_smoke"
        reason = "candidate_improves_edge_compactness_without_new_overlap_risk"
    else:
        adoption = "keep_current"
        reason = "candidate_does_not_clear_layout_quality_margin"
    return {
        "baseline": {
            "algorithm": "community-ring-v1",
            "status": "current",
            "metrics": baseline_metrics,
        },
        "semantic_force": {
            "algorithm": "semantic-force-v1",
            "status": "benchmark_only",
            "basis": [
                "force_directed_edges",
                "community_gravity",
                "deterministic_hash_seed",
                "overlap_risk_metric",
            ],
            "metrics": semantic_metrics,
            "delta_vs_baseline": delta,
        },
        "recommendation": {
            "ui_adoption": adoption,
            "reason": reason,
        },
    }


def build_layout_report(
    *,
    linkgraph_db: Path = DEFAULT_LINKGRAPH_DB,
    communities_db: Path = DEFAULT_COMMUNITIES_DB,
    related_db: Path = DEFAULT_RELATED_DB,
    similar_top_k: int = 2,
    include_candidates: bool = False,
) -> dict[str, Any]:
    titles = _load_graph_nodes(linkgraph_db)
    nodes = set(titles)
    explicit = _explicit_pairs(linkgraph_db, nodes)
    similar = _similar_pairs(related_db, nodes, top_k=similar_top_k)
    explicit_degree = _degree(nodes, explicit)
    drawn_degree = _degree(nodes, explicit | similar)
    folder_counts = Counter(_folder(path) for path in nodes)
    report = {
        "summary": {
            "nodes": len(nodes),
            "research_nodes": folder_counts.get("research", 0),
            "concept_nodes": folder_counts.get("concepts", 0),
            "explicit_edges": len(explicit),
            "similar_edges_top_k": len(similar),
            "zero_degree_explicit": sum(1 for value in explicit_degree.values() if value == 0),
            "zero_degree_with_similar": sum(1 for value in drawn_degree.values() if value == 0),
        },
        "communities": _community_stats(communities_db, nodes),
        "layout_parameters": {
            "global_layout_algorithm": "community-ring-v1",
            "local_layout_algorithm": "local-radial-v1",
            "global_link_distance": 36,
            "local_link_distance": 52,
            "global_gravity": 0.05,
            "local_gravity": 0.09,
            "global_charge": -160,
            "local_charge": -180,
            "cluster_anchor_count": 12,
            "cluster_strength": 0.25,
            "cluster_radius_formula": "24 * sqrt(nodes)",
            "similar_top_k": similar_top_k,
        },
        "risks": [
            "ring_anchor_geometry_is_not_semantic_coordinates",
            "top_12_communities_are_anchored_long_tail_floats",
        ],
    }
    report["quality_gate"] = evaluate_quality_gate(report)
    if include_candidates:
        report["layout_candidates"] = evaluate_layout_candidates(
            nodes,
            explicit | similar,
            _community_assignments(communities_db, nodes),
        )
    return report


def format_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    communities = report["communities"]
    quality_gate = report.get("quality_gate", {})
    lines = [
        "# Graph Layout Baseline Audit",
        "",
        "## Global Graph Inputs",
        "",
        f"- nodes: {summary['nodes']}",
        f"- research_nodes: {summary['research_nodes']}",
        f"- concept_nodes: {summary['concept_nodes']}",
        f"- explicit_edges: {summary['explicit_edges']}",
        f"- similar_edges_top_k: {summary['similar_edges_top_k']}",
        f"- zero_degree_explicit: {summary['zero_degree_explicit']}",
        f"- zero_degree_with_similar: {summary['zero_degree_with_similar']}",
        "",
        "## Community Anchor Coverage",
        "",
        f"- communities: {communities['count']}",
        f"- largest: {communities['largest']}",
        f"- top_12_nodes: {communities['top_12_nodes']}",
        f"- top_12_coverage_percent: {communities['top_12_coverage_percent']}",
        f"- long_tail_nodes: {communities['long_tail_nodes']}",
        "",
        "## Current Layout Parameters",
        "",
    ]
    for key, value in report["layout_parameters"].items():
        lines.append(f"- {key}: {value}")
    if quality_gate:
        status = "pass" if quality_gate["passed"] else "fail"
        lines.extend(["", "## Layout Quality Gate", "", f"- status: {status}"])
        for key, value in quality_gate["measurements"].items():
            lines.append(f"- {key}: {value}")
        for failure in quality_gate["failures"]:
            lines.append(
                "- failure: "
                f"{failure['metric']}={failure['actual']} > {failure['limit']} "
                f"({failure['reason']})"
            )
    candidates = report.get("layout_candidates")
    if candidates:
        semantic = candidates["semantic_force"]
        recommendation = candidates["recommendation"]
        lines.extend(
            [
                "",
                "## Layout Candidate Benchmark",
                "",
                f"- baseline_algorithm: {candidates['baseline']['algorithm']}",
                f"- candidate_algorithm: {semantic['algorithm']}",
                f"- candidate_status: {semantic['status']}",
                f"- basis: {', '.join(semantic['basis'])}",
            ]
        )
        for key, value in semantic["metrics"].items():
            lines.append(f"- candidate_{key}: {value}")
        for key, value in semantic["delta_vs_baseline"].items():
            lines.append(f"- {key}: {value}")
        lines.extend(
            [
                f"- ui_adoption: {recommendation['ui_adoption']}",
                f"- recommendation_reason: {recommendation['reason']}",
            ]
        )
    lines.extend(["", "## Layout Risks", ""])
    for risk in report["risks"]:
        lines.append(f"- {risk}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit graph layout baseline inputs.")
    parser.add_argument("--linkgraph-db", default=str(DEFAULT_LINKGRAPH_DB))
    parser.add_argument("--communities-db", default=str(DEFAULT_COMMUNITIES_DB))
    parser.add_argument("--related-db", default=str(DEFAULT_RELATED_DB))
    parser.add_argument("--similar-top-k", type=int, default=2)
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Benchmark deterministic semantic layout candidates against the current baseline",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if the layout gate fails")
    parser.add_argument("--output", help="Write report to this path instead of stdout")
    args = parser.parse_args(argv)
    if args.similar_top_k <= 0:
        print("similar-top-k must be positive")
        raise SystemExit(2)
    report = build_layout_report(
        linkgraph_db=Path(args.linkgraph_db),
        communities_db=Path(args.communities_db),
        related_db=Path(args.related_db),
        similar_top_k=args.similar_top_k,
        include_candidates=args.include_candidates,
    )
    content = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.json
        else format_markdown(report)
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    if args.check and not report["quality_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
