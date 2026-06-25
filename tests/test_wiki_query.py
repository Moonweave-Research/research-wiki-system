from legacy.graphrag.wiki_query import query_wiki
from scripts.core.wiki_compiler import generate_wiki_page


def test_query_wiki(tmp_path):
    base_dir = tmp_path / "wiki"
    generate_wiki_page(
        {"title": "A", "domain": "research", "category": "materials", "tags": ["x"]}, str(base_dir)
    )
    generate_wiki_page(
        {"title": "B", "domain": "general", "category": "tools", "tags": ["y"]}, str(base_dir)
    )

    results_domain = query_wiki(str(base_dir), domain="research")
    assert len(results_domain) == 1
    assert results_domain[0]["path"].endswith("a.md")

    results_tag = query_wiki(str(base_dir), tag="y")
    assert len(results_tag) == 1
    assert results_tag[0]["path"].endswith("b.md")


def test_query_wiki_tag_case_insensitive(tmp_path):
    base_dir = tmp_path / "wiki"
    generate_wiki_page(
        {"title": "C", "domain": "research", "category": "materials", "tags": ["Polymer"]},
        str(base_dir),
    )

    # Query with lowercase must match a Title-Case stored tag.
    results = query_wiki(str(base_dir), tag="polymer")
    assert len(results) == 1
    assert results[0]["path"].endswith("c.md")

    # And uppercase query against the same tag.
    results_upper = query_wiki(str(base_dir), tag="POLYMER")
    assert len(results_upper) == 1
