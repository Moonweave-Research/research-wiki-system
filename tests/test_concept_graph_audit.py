import sqlite3
from pathlib import Path

import yaml

from scripts.search import concept_graph_audit


def _write_note(vault: Path, rel_path: str, tags: list[str]) -> None:
    path = vault / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = {
        "aliases": [Path(rel_path).stem],
        "domain": "research",
        "category": "materials" if rel_path.startswith("research/") else "concepts",
        "tags": tags,
        "sources": [],
    }
    path.write_text(
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
        + "---\n"
        + f"# {Path(rel_path).stem}\n",
        encoding="utf-8",
    )


def _minimal_linkgraph(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE notes (
            path TEXT PRIMARY KEY,
            title TEXT,
            aliases_json TEXT
        );
        CREATE TABLE links (
            src_path TEXT,
            dst_target TEXT,
            dst_resolved INTEGER
        );
        CREATE TABLE tags (
            tag TEXT,
            note_path TEXT
        );
        """
    )
    notes = [
        ("concepts/Concept_A.md", "Concept A", "[]"),
        ("concepts/Concept_B.md", "Concept B", "[]"),
        ("research/paperOne.md", "Paper One", "[]"),
        ("research/paperTwo.md", "Paper Two", "[]"),
        ("research/lonely.md", "Lonely Paper", "[]"),
    ]
    conn.executemany("INSERT INTO notes VALUES (?, ?, ?)", notes)
    links = [
        ("concepts/Concept_A.md", "research/paperOne.md", 1),
        ("concepts/Concept_A.md", "research/paperOne.md", 1),
        ("concepts/Concept_A.md", "research/paperTwo.md", 1),
        ("concepts/Concept_B.md", "research/paperOne.md", 1),
    ]
    conn.executemany("INSERT INTO links VALUES (?, ?, ?)", links)
    tags = [
        ("Photothermal", "research/paperOne.md"),
        ("Hydrogel", "research/paperOne.md"),
        ("Flexible", "research/paperTwo.md"),
        ("Actuator", "research/paperTwo.md"),
        ("Misc", "research/lonely.md"),
    ]
    conn.executemany("INSERT INTO tags VALUES (?, ?)", tags)
    conn.commit()
    conn.close()


def _communities(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE community (path TEXT PRIMARY KEY, community INTEGER)")
    rows = [
        ("concepts/Concept_A.md", 0),
        ("concepts/Concept_B.md", 0),
        ("research/paperOne.md", 0),
        ("research/paperTwo.md", 1),
        ("research/lonely.md", 2),
    ]
    conn.executemany("INSERT INTO community VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def _related(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE related (src_path TEXT, dst_path TEXT, score REAL)")
    conn.execute(
        "INSERT INTO related VALUES (?, ?, ?)",
        ("research/paperOne.md", "research/paperTwo.md", 0.93),
    )
    conn.commit()
    conn.close()


def test_audit_report_counts_overlap_zero_concept_and_duplicates(tmp_path: Path) -> None:
    vault = tmp_path / "wiki"
    _write_note(vault, "concepts/Concept_A.md", ["Concept A"])
    _write_note(vault, "concepts/Concept_B.md", ["Concept B"])
    _write_note(vault, "research/paperOne.md", ["Photothermal", "Hydrogel"])
    _write_note(vault, "research/paperTwo.md", ["Flexible", "Actuator"])
    _write_note(vault, "research/lonely.md", ["Misc"])
    linkgraph_db = tmp_path / "linkgraph.sqlite"
    communities_db = tmp_path / "communities.sqlite"
    related_db = tmp_path / "related.sqlite"
    _minimal_linkgraph(linkgraph_db)
    _communities(communities_db)
    _related(related_db)

    report = concept_graph_audit.build_audit_report(
        vault_dir=vault,
        linkgraph_db=linkgraph_db,
        communities_db=communities_db,
        related_db=related_db,
        candidate_min_count=1,
    )

    assert report["summary"]["nodes"] == 5
    assert report["summary"]["research"] == 3
    assert report["summary"]["concept_linked"] == 2
    assert report["summary"]["zero_concept"] == 1
    assert report["summary"]["isolated"] == 1
    assert report["overlap"]["concept_paper_rows"] == 4
    assert report["overlap"]["unique_edges"] == 3
    assert report["overlap"]["duplicate_rows"] == 1
    assert report["overlap"]["row_avg_concepts_per_paper"] == 2.0
    assert report["overlap"]["unique_avg_concepts_per_paper"] == 1.5
    assert report["communities"]["communities_db_count"] == 3
    assert report["communities"]["largest_community"] == 3
    assert report["hotspots"]["top_overlapping_papers"][0]["path"] == "research/paperOne.md"
    assert report["hotspots"]["top_overlapping_papers"][0]["unique_concepts"] == 2
    assert report["hotspots"]["uncovered_tag_pairs"][0]["count"] >= 1


def test_uncovered_tag_pairs_ignore_existing_split_alias_parent_pairs() -> None:
    candidates = concept_graph_audit._candidate_pairs(
        {"research/lce.md": {"Liquid Crystal", "Liquid Crystal Elastomers"}},
        {"research/lce.md": "LCE paper"},
        limit=10,
        sample_limit=5,
        min_count=1,
    )

    assert candidates == []


def test_uncovered_tag_pairs_keep_parent_parent_pairs_as_candidates() -> None:
    candidates = concept_graph_audit._candidate_pairs(
        {"research/lca.md": {"Liquid Crystal", "Actuator"}},
        {"research/lca.md": "Liquid crystal actuator paper"},
        limit=10,
        sample_limit=5,
        min_count=1,
    )

    assert candidates[0]["tags"] == ["Actuator", "Liquid Crystal"]


def test_audit_report_includes_split_rule_effectiveness(tmp_path: Path) -> None:
    vault = tmp_path / "wiki"
    for rel_path, tags in [
        ("concepts/Actuator.md", ["Actuator"]),
        ("concepts/Dielectric.md", ["Dielectric"]),
        ("concepts/Elastomer.md", ["Elastomer"]),
        ("concepts/Dielectric_Elastomer_Actuator.md", ["Dielectric Elastomer Actuator"]),
    ]:
        _write_note(vault, rel_path, tags)
    linkgraph_db = tmp_path / "linkgraph.sqlite"
    conn = sqlite3.connect(linkgraph_db)
    conn.executescript(
        """
        CREATE TABLE notes (path TEXT PRIMARY KEY, title TEXT, aliases_json TEXT);
        CREATE TABLE links (src_path TEXT, dst_target TEXT, dst_resolved INTEGER);
        CREATE TABLE tags (tag TEXT, note_path TEXT);
        """
    )
    notes = [
        ("concepts/Actuator.md", "Actuator", '["Actuator"]'),
        ("concepts/Dielectric.md", "Dielectric", '["Dielectric"]'),
        ("concepts/Elastomer.md", "Elastomer", '["Elastomer"]'),
        (
            "concepts/Dielectric_Elastomer_Actuator.md",
            "Dielectric Elastomer Actuator",
            '["DEA", "Dielectric Elastomer"]',
        ),
    ]
    for index in range(12):
        rel_path = f"research/dea{index:02d}.md"
        _write_note(vault, rel_path, ["Dielectric", "Elastomer", "Actuator"])
        notes.append((rel_path, f"DEA {index:02d}", "[]"))
        for tag in ["Dielectric", "Elastomer", "Actuator"]:
            conn.execute("INSERT INTO tags VALUES (?, ?)", (tag, rel_path))
    conn.executemany("INSERT INTO notes VALUES (?, ?, ?)", notes)
    conn.commit()
    conn.close()

    report = concept_graph_audit.build_audit_report(
        vault_dir=vault,
        linkgraph_db=linkgraph_db,
        communities_db=tmp_path / "missing-communities.sqlite",
        related_db=tmp_path / "missing-related.sqlite",
    )

    dea = next(
        rule
        for rule in report["rules"]
        if rule["target_slug"] == "Dielectric_Elastomer_Actuator"
    )
    assert dea["qualifying_count"] == 12
    assert dea["routed_count"] == 12
    assert dea["target_reused_or_created"] == "reused"
    assert dea["suppressed_parent_counts"]["concepts/Actuator.md"] == 12


def test_audit_report_simulates_candidate_split_without_registering_rule(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "wiki"
    for rel_path, tags in [
        ("concepts/Actuator.md", ["Actuator"]),
        ("concepts/Liquid_Crystal.md", ["Liquid Crystal"]),
        ("research/lca1.md", ["Liquid Crystal", "Actuator"]),
        ("research/lca2.md", ["Liquid Crystal", "Actuator"]),
    ]:
        _write_note(vault, rel_path, tags)

    linkgraph_db = tmp_path / "linkgraph.sqlite"
    conn = sqlite3.connect(linkgraph_db)
    conn.executescript(
        """
        CREATE TABLE notes (path TEXT PRIMARY KEY, title TEXT, aliases_json TEXT);
        CREATE TABLE links (src_path TEXT, dst_target TEXT, dst_resolved INTEGER);
        CREATE TABLE tags (tag TEXT, note_path TEXT);
        """
    )
    notes = [
        ("concepts/Actuator.md", "Actuator", '["Actuator"]'),
        ("concepts/Liquid_Crystal.md", "Liquid Crystal", '["Liquid Crystal"]'),
        ("research/lca1.md", "LCA 1", "[]"),
        ("research/lca2.md", "LCA 2", "[]"),
    ]
    conn.executemany("INSERT INTO notes VALUES (?, ?, ?)", notes)
    for paper in ("research/lca1.md", "research/lca2.md"):
        conn.execute("INSERT INTO links VALUES (?, ?, ?)", ("concepts/Actuator.md", paper, 1))
        conn.execute(
            "INSERT INTO links VALUES (?, ?, ?)",
            ("concepts/Liquid_Crystal.md", paper, 1),
        )
        conn.execute("INSERT INTO tags VALUES (?, ?)", ("Actuator", paper))
        conn.execute("INSERT INTO tags VALUES (?, ?)", ("Liquid Crystal", paper))
    conn.commit()
    conn.close()

    candidate = concept_graph_audit.make_simulated_split_rule(
        "Liquid Crystal Actuator",
        required_tags=("Liquid Crystal", "Actuator"),
        min_papers=1,
    )
    report = concept_graph_audit.build_audit_report(
        vault_dir=vault,
        linkgraph_db=linkgraph_db,
        communities_db=tmp_path / "missing-communities.sqlite",
        related_db=tmp_path / "missing-related.sqlite",
        simulated_rules=[candidate],
    )

    simulation = report["simulations"][0]
    assert simulation["target_slug"] == "Liquid_Crystal_Actuator"
    assert simulation["routed_paper_count"] == 2
    assert simulation["suppressed_parent_count"] == 4
    assert simulation["suppressed_parent_counts"] == {
        "concepts/Actuator.md": 2,
        "concepts/Liquid_Crystal.md": 2,
    }
    assert simulation["before"]["unique_avg_concepts_per_paper"] == 2.0
    assert simulation["after"]["unique_avg_concepts_per_paper"] == 1.0
    assert simulation["delta"]["unique_avg_concepts_per_paper"] == -1.0
    assert "Liquid_Crystal_Actuator" not in {
        rule.target_slug for rule in concept_graph_audit.concept_routing.SPLIT_RULES
    }


def test_audit_report_batch_simulates_uncovered_candidates_with_decisions(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "wiki"
    for rel_path, tags in [
        ("concepts/Actuator.md", ["Actuator"]),
        ("concepts/Hydrogel.md", ["Hydrogel"]),
        ("research/hydrogel_actuator1.md", ["Hydrogel", "Actuator"]),
        ("research/hydrogel_actuator2.md", ["Hydrogel", "Actuator"]),
        ("research/actuator_parent_anchor.md", ["Actuator"]),
        ("research/hydrogel_parent_anchor.md", ["Hydrogel"]),
    ]:
        _write_note(vault, rel_path, tags)

    linkgraph_db = tmp_path / "linkgraph.sqlite"
    conn = sqlite3.connect(linkgraph_db)
    conn.executescript(
        """
        CREATE TABLE notes (path TEXT PRIMARY KEY, title TEXT, aliases_json TEXT);
        CREATE TABLE links (src_path TEXT, dst_target TEXT, dst_resolved INTEGER);
        CREATE TABLE tags (tag TEXT, note_path TEXT);
        """
    )
    notes = [
        ("concepts/Actuator.md", "Actuator", '["Actuator"]'),
        ("concepts/Hydrogel.md", "Hydrogel", '["Hydrogel"]'),
        ("research/hydrogel_actuator1.md", "Hydrogel Actuator 1", "[]"),
        ("research/hydrogel_actuator2.md", "Hydrogel Actuator 2", "[]"),
        ("research/actuator_parent_anchor.md", "Actuator Parent Anchor", "[]"),
        ("research/hydrogel_parent_anchor.md", "Hydrogel Parent Anchor", "[]"),
    ]
    conn.executemany("INSERT INTO notes VALUES (?, ?, ?)", notes)
    for paper in ("research/hydrogel_actuator1.md", "research/hydrogel_actuator2.md"):
        conn.execute("INSERT INTO links VALUES (?, ?, ?)", ("concepts/Actuator.md", paper, 1))
        conn.execute("INSERT INTO links VALUES (?, ?, ?)", ("concepts/Hydrogel.md", paper, 1))
        conn.execute("INSERT INTO tags VALUES (?, ?)", ("Actuator", paper))
        conn.execute("INSERT INTO tags VALUES (?, ?)", ("Hydrogel", paper))
    conn.execute(
        "INSERT INTO links VALUES (?, ?, ?)",
        ("concepts/Actuator.md", "research/actuator_parent_anchor.md", 1),
    )
    conn.execute(
        "INSERT INTO links VALUES (?, ?, ?)",
        ("concepts/Hydrogel.md", "research/hydrogel_parent_anchor.md", 1),
    )
    conn.execute("INSERT INTO tags VALUES (?, ?)", ("Actuator", "research/actuator_parent_anchor.md"))
    conn.execute("INSERT INTO tags VALUES (?, ?)", ("Hydrogel", "research/hydrogel_parent_anchor.md"))
    conn.commit()
    conn.close()

    report = concept_graph_audit.build_audit_report(
        vault_dir=vault,
        linkgraph_db=linkgraph_db,
        communities_db=tmp_path / "missing-communities.sqlite",
        related_db=tmp_path / "missing-related.sqlite",
        candidate_min_count=1,
        simulate_candidates=True,
    )

    candidate = report["candidate_simulations"][0]
    assert candidate["source_tags"] == ["Actuator", "Hydrogel"]
    assert candidate["routed_paper_count"] == 2
    assert candidate["suppressed_parent_count"] == 4
    assert candidate["decision"] == "promote_candidate"
    assert candidate["decision_reasons"] == [
        "avg_concepts_non_increasing",
        "isolation_below_limit",
        "modularity_not_worse",
        "parents_suppressed",
    ]
    assert candidate["delta"]["unique_avg_concepts_per_paper"] == -0.5
    assert report["hotspots"]["uncovered_tag_pairs"][0]["simulation_decision"] == "promote_candidate"


def test_audit_markdown_includes_candidate_simulation_decisions() -> None:
    report = {
        "summary": {
            "nodes": 4,
            "isolated": 0,
            "isolation_percent": 0,
            "research": 2,
            "concept_linked": 2,
            "zero_concept": 0,
        },
        "overlap": {
            "concept_paper_rows": 4,
            "unique_edges": 4,
            "duplicate_rows": 0,
            "row_avg_concepts_per_paper": 2.0,
            "unique_avg_concepts_per_paper": 2.0,
        },
        "communities": {
            "wikilink_modularity": 0.5,
            "overlay_modularity": 0.5,
        },
        "simulations": [],
        "candidate_simulations": [
            {
                "target_slug": "Actuator_Hydrogel",
                "source_tags": ["Actuator", "Hydrogel"],
                "routed_paper_count": 2,
                "suppressed_parent_count": 4,
                "decision": "promote_candidate",
                "decision_reasons": ["avg_concepts_non_increasing"],
                "delta": {
                    "unique_avg_concepts_per_paper": -1.0,
                    "isolation_percent": 0,
                    "wikilink_modularity": 0.01,
                },
            }
        ],
        "hotspots": {
            "top_overlapping_papers": [],
            "uncovered_tag_pairs": [],
        },
    }

    markdown = concept_graph_audit._format_markdown(report)

    assert "## Candidate Split Simulations" in markdown
    assert "Actuator + Hydrogel" in markdown
    assert "promote_candidate" in markdown
