from pathlib import Path

from scripts.core.linkgraph import scan_vault
from scripts.core.wiki_lint import (
    _should_skip_lint_file,
    _should_skip_lint_path,
    find_duplicate_titles,
    lint_file,
)


def test_lint_file_valid(tmp_path):
    p = tmp_path / "valid.md"
    p.write_text(
        "---\r\ndomain: general\r\ncategory: general\r\ntags: []\r\nsources: []\r\n---\r\n# Content",
        encoding="utf-8",
    )
    is_valid, _ = lint_file(str(p))
    assert is_valid


def test_lint_file_invalid(tmp_path):
    p = tmp_path / "invalid.md"
    p.write_text("# No Frontmatter\n", encoding="utf-8")
    is_valid, _ = lint_file(str(p))
    assert not is_valid


def test_lint_allows_http_links(tmp_path):
    file_path = tmp_path / "test_http.md"
    content = """---
domain: research
category: general
tags:
  - test
sources:
  - test:123
---
# Title
This is an [external link](https://example.com) and this is an [internal link](file.md).
"""
    file_path.write_text(content, encoding="utf-8")
    is_valid, msg = lint_file(str(file_path))
    assert not is_valid
    assert "Standard Markdown links" in msg


def test_lint_passes_with_only_http_links(tmp_path):
    file_path = tmp_path / "test_http_only.md"
    content = """---
domain: research
category: general
tags:
  - test
sources:
  - test:123
---
# Title
This is an [external link](https://example.com).
"""
    file_path.write_text(content, encoding="utf-8")
    is_valid, msg = lint_file(str(file_path))
    assert is_valid


def test_lint_scope_skips_operational_and_generated_pages():
    assert _should_skip_lint_path("wiki/log.md")
    assert _should_skip_lint_path("wiki/index.md")
    assert _should_skip_lint_path("wiki/dashboard.md")
    assert _should_skip_lint_path("wiki/research/02_Research_Timeline.md")
    assert _should_skip_lint_path("wiki/dispatches/2026-05-01/example/draft.md")
    assert not _should_skip_lint_path("wiki/research/example.md")
    assert not _should_skip_lint_path("wiki/dispatches_backup/example.md")
    assert not _should_skip_lint_path("wiki/templates_old/example.md")
    assert not _should_skip_lint_path("wiki/metaphor/example.md")

def test_lint_scope_skips_external_vault_operational_pages(tmp_path):
    vault = tmp_path / "external-vault"
    assert _should_skip_lint_file(vault / "log.md", vault)
    assert _should_skip_lint_file(vault / "research" / "02_Research_Timeline.md", vault)
    assert _should_skip_lint_file(vault / "meta" / "Tags" / "Actuator.md", vault)
    assert not _should_skip_lint_file(vault / "research" / "paper.md", vault)


def test_lint_skips_material_completeness_for_irrelevant_notes(tmp_path):
    file_path = tmp_path / "irrelevant.md"
    file_path.write_text(
        """---
domain: research
category: materials
tags: []
sources: []
status: irrelevant
---
# Mismatched note

## Abstract
""",
        encoding="utf-8",
    )

    is_valid, msg = lint_file(str(file_path))
    assert is_valid, msg


def _write_note(directory, name, title):
    (directory / name).write_text(
        f"""---
aliases:
- {title}
domain: research
category: general
tags: []
sources: []
---
# Body
""",
        encoding="utf-8",
    )


def test_find_duplicate_titles_flags_same_title(tmp_path):
    _write_note(tmp_path, "guoReview2021.md", "Review of Dielectric Elastomer Actuators")
    _write_note(tmp_path, "guoReview2021a.md", "Review of Dielectric Elastomer Actuators")
    notes, _ = scan_vault(Path(tmp_path))
    groups = find_duplicate_titles(notes)
    assert len(groups) == 1
    (paths,) = groups.values()
    assert sorted(paths) == ["guoReview2021.md", "guoReview2021a.md"]


def test_find_duplicate_titles_none_for_distinct(tmp_path):
    _write_note(tmp_path, "alpha.md", "Dielectric Elastomer Actuators Review")
    _write_note(tmp_path, "beta.md", "Liquid Crystal Network Soft Robots")
    notes, _ = scan_vault(Path(tmp_path))
    assert find_duplicate_titles(notes) == {}
