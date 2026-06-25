from pathlib import Path

from scripts.core.export_system import has_private_marker, is_system_export_path


def test_system_export_path_allowlist_excludes_private_surfaces():
    assert is_system_export_path(Path("apps/vault-ui/backend/services.py"))
    assert is_system_export_path(Path("scripts/core/linkgraph.py"))
    assert is_system_export_path(Path("tests/fixtures/sample-vault/research/sample-paper.md"))
    assert is_system_export_path(Path("product_samples/research-wiki-skill-kit/index.html"))
    assert is_system_export_path(Path("pyproject.toml"))

    assert not is_system_export_path(Path("wiki/research/private-paper.md"))
    assert not is_system_export_path(Path(".obsidian/graph.json"))
    assert not is_system_export_path(Path(".graph_index/linkgraph.sqlite"))
    assert not is_system_export_path(Path("내 라이브러리/files/private.pdf"))
    assert not is_system_export_path(Path("product_samples/demo/.wrangler/cache/pages.json"))


def test_private_marker_scan_detects_local_paths(tmp_path):
    clean = tmp_path / "clean.md"
    clean.write_text("# clean\nNo private path here.\n", encoding="utf-8")
    assert has_private_marker(clean) is None

    private = tmp_path / "private.md"
    marker = "/" + "Users" + "/"
    private.write_text(f"pdf_path: {marker}example/private.pdf\n", encoding="utf-8")
    assert has_private_marker(private) == marker
