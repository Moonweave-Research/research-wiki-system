import sqlite3
from pathlib import Path

import yaml

from scripts.core import linkgraph
from scripts.ingest import concept_backfill


def _write_paper(
    vault: Path, stem: str, tags: list[str], *, status: str | None = None
) -> None:
    path = vault / "research" / f"{stem}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = {
        "aliases": [stem],
        "domain": "research",
        "category": "materials",
        "tags": tags,
        "sources": [],
    }
    if status is not None:
        frontmatter["status"] = status
    path.write_text(
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
        + "---\n"
        + f"# {stem}\n\nNo hand-authored wikilinks here.\n",
        encoding="utf-8",
    )


def _write_concept(vault: Path, stem: str, aliases: list[str], body: str) -> Path:
    path = vault / "concepts" / f"{stem}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = {
        "aliases": aliases,
        "domain": "research",
        "category": "concepts",
        "tags": aliases[:1],
        "sources": [],
    }
    path.write_text(
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
        + "---\n"
        + body,
        encoding="utf-8",
    )
    return path


def _build_db(vault: Path, db: Path) -> None:
    linkgraph.build_index(vault, db)


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]
    loaded = yaml.safe_load(frontmatter)
    assert isinstance(loaded, dict)
    return loaded


def test_apply_creates_missing_concept_note_from_qualifying_tag(tmp_path: Path) -> None:
    vault = tmp_path / "wiki"
    db = tmp_path / "linkgraph.sqlite"
    _write_paper(vault, "paperAlpha2024", ["Elastomer"])
    _write_paper(vault, "paperBeta2024", ["Elastomer"])
    _build_db(vault, db)

    result = concept_backfill.main(
        ["--vault", str(vault), "--db", str(db), "--min-papers", "2", "--apply"]
    )

    assert result == 0
    concept = vault / "concepts" / "Elastomer.md"
    assert concept.exists()
    assert _frontmatter(concept) == {
        "aliases": ["Elastomer"],
        "domain": "research",
        "category": "concepts",
        "tags": ["Elastomer"],
        "sources": [],
    }
    text = concept.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "# Elastomer\n" in text
    assert "## 정의\n" in text
    assert "## 관련 논문 (자동 생성)\n" in text
    assert "- [[paperAlpha2024]]\n" in text
    assert "- [[paperBeta2024]]\n" in text

    rebuilt_db = tmp_path / "rebuilt.sqlite"
    linkgraph.build_index(vault, rebuilt_db)
    conn = sqlite3.connect(rebuilt_db)
    try:
        rows = conn.execute(
            "SELECT dst_target, dst_resolved FROM links WHERE src_path=? ORDER BY dst_target",
            ("concepts/Elastomer.md",),
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("research/paperAlpha2024.md", 1), ("research/paperBeta2024.md", 1)]


def test_existing_concept_auto_section_is_replaced_without_touching_hand_body(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "wiki"
    db = tmp_path / "linkgraph.sqlite"
    _write_paper(vault, "paperGamma2024", ["Liquid Crystal Elastomers"])
    _write_paper(vault, "paperTheta2024", ["Liquid Crystal Elastomers"])
    concept = _write_concept(
        vault,
        "Liquid_Crystal_Elastomer",
        ["LCE", "Liquid Crystal Elastomers"],
        "# Liquid Crystal Elastomer\n\n## 정의\n\nHand-written definition.\n\n"
        "- Hand curated paper: [[paperGamma2024]]\n\n"
        "<!-- concept-backfill:start -->\n"
        "## 관련 논문 (자동 생성)\n\n"
        "- [[stalePaper]]\n"
        "<!-- concept-backfill:end -->\n",
    )
    _build_db(vault, db)

    concept_backfill.main(["--vault", str(vault), "--db", str(db), "--min-papers", "1", "--apply"])
    first_pass = concept.read_text(encoding="utf-8")
    concept_backfill.main(["--vault", str(vault), "--db", str(db), "--min-papers", "1", "--apply"])

    assert concept.read_text(encoding="utf-8") == first_pass
    assert "## 정의\n\nHand-written definition.\n\n" in first_pass
    assert first_pass.count("[[paperGamma2024]]") == 1
    assert "- [[paperTheta2024]]\n" in first_pass
    assert "stalePaper" not in first_pass


def test_plural_tags_are_folded_into_one_created_concept(tmp_path: Path) -> None:
    vault = tmp_path / "wiki"
    db = tmp_path / "linkgraph.sqlite"
    _write_paper(vault, "paperDelta2024", ["Actuator"])
    _write_paper(vault, "paperEpsilon2024", ["Actuators"])
    _build_db(vault, db)

    concept_backfill.main(["--vault", str(vault), "--db", str(db), "--min-papers", "1", "--apply"])

    actuator = vault / "concepts" / "Actuator.md"
    assert actuator.exists()
    assert not (vault / "concepts" / "Actuators.md").exists()
    assert _frontmatter(actuator)["aliases"] == ["Actuator", "Actuators"]
    text = actuator.read_text(encoding="utf-8")
    assert "- [[paperDelta2024]]\n" in text
    assert "- [[paperEpsilon2024]]\n" in text


def test_backfill_skips_irrelevant_papers(tmp_path: Path) -> None:
    vault = tmp_path / "wiki"
    db = tmp_path / "linkgraph.sqlite"
    _write_paper(vault, "paperRelevant2024", ["Actuator"])
    _write_paper(vault, "paperIrrelevant2024", ["Actuator"], status="irrelevant")
    _build_db(vault, db)

    concept_backfill.main(["--vault", str(vault), "--db", str(db), "--min-papers", "1", "--apply"])

    text = (vault / "concepts" / "Actuator.md").read_text(encoding="utf-8")
    assert "- [[paperRelevant2024]]\n" in text
    assert "paperIrrelevant2024" not in text


def test_dry_run_is_default_and_skips_system_tags(tmp_path: Path, capsys) -> None:
    vault = tmp_path / "wiki"
    db = tmp_path / "linkgraph.sqlite"
    _write_paper(vault, "paperZeta2024", ["zotero_import"])
    _write_paper(vault, "paperEta2024", ["Sensor"])
    _build_db(vault, db)

    result = concept_backfill.main(["--vault", str(vault), "--db", str(db), "--min-papers", "1"])

    assert result == 0
    assert not (vault / "concepts" / "Sensor.md").exists()
    assert not (vault / "concepts" / "zotero_import.md").exists()
    output = capsys.readouterr().out
    assert "CREATE concepts/Sensor.md" in output
    assert "zotero_import" not in output


def test_apply_skips_unknown_high_frequency_tags_without_failing(
    tmp_path: Path, capsys
) -> None:
    vault = tmp_path / "wiki"
    db = tmp_path / "linkgraph.sqlite"
    _write_paper(vault, "paperIota2024", ["Uncurated Topic"])
    _write_paper(vault, "paperKappa2024", ["Uncurated Topic"])
    _build_db(vault, db)

    result = concept_backfill.main(
        ["--vault", str(vault), "--db", str(db), "--min-papers", "2", "--apply"]
    )

    assert result == 0
    assert not (vault / "concepts" / "Uncurated_Topic.md").exists()
    output = capsys.readouterr().out
    assert "SKIP Uncurated Topic" in output


def test_backfill_respects_specific_routing_when_updating_broad_parent(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "wiki"
    db = tmp_path / "linkgraph.sqlite"
    for index in range(12):
        _write_paper(
            vault,
            f"paperSpecific{index:02d}2024",
            ["Dielectric", "Elastomer", "Actuator", "Polymer"],
        )
    _write_paper(vault, "paperBroad2024", ["Actuator"])
    _write_concept(vault, "Dielectric", ["Dielectric"], "# Dielectric\n\n## 정의\n\nHand body.\n")
    _write_concept(vault, "Elastomer", ["Elastomer"], "# Elastomer\n\n## 정의\n\nHand body.\n")
    polymer = _write_concept(vault, "Polymer", ["Polymer"], "# Polymer\n\n## 정의\n\nHand body.\n")
    actuator = _write_concept(
        vault,
        "Actuator",
        ["Actuator"],
        "# Actuator\n\n## 정의\n\nHand body.\n",
    )
    _write_concept(
        vault,
        "Dielectric_Elastomer_Actuator",
        ["DEA", "Dielectric Elastomer"],
        "# Dielectric Elastomer Actuator\n\n## 정의\n\nHand body.\n",
    )
    _build_db(vault, db)

    result = concept_backfill.main(
        ["--vault", str(vault), "--db", str(db), "--min-papers", "1", "--apply"]
    )

    assert result == 0
    text = actuator.read_text(encoding="utf-8")
    assert "- [[paperBroad2024]]\n" in text
    assert "- [[paperSpecific002024]]\n" not in text
    assert "- [[paperSpecific002024]]\n" not in polymer.read_text(encoding="utf-8")


def test_backfill_does_not_overwrite_split_owned_specific_concepts(tmp_path: Path) -> None:
    vault = tmp_path / "wiki"
    db = tmp_path / "linkgraph.sqlite"
    for index in range(12):
        _write_paper(
            vault,
            f"paperLce{index:02d}2024",
            ["Liquid Crystal Elastomer", "Liquid Crystal", "Elastomer"],
        )
    specific = _write_concept(
        vault,
        "Liquid_Crystal_Elastomer",
        ["LCE", "Liquid Crystal Elastomer"],
        "# Liquid Crystal Elastomer\n\n## 정의\n\nHand body.\n\n"
        "<!-- concept-backfill:start -->\n"
        "## 관련 논문 (자동 생성)\n\n"
        "<!-- Regenerated by scripts.ingest.concept_backfill. -->\n"
        "- [[existingSplitOwnerPaper]]\n"
        "<!-- concept-backfill:end -->\n",
    )
    before = specific.read_text(encoding="utf-8")
    _build_db(vault, db)

    result = concept_backfill.main(
        ["--vault", str(vault), "--db", str(db), "--min-papers", "1", "--apply"]
    )

    assert result == 0
    assert specific.read_text(encoding="utf-8") == before
