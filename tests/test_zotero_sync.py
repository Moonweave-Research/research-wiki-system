import yaml

from scripts.ingest.zotero_sync import (
    _norm_title,
    extract_item_data,
    generate_markdown,
    generate_merged_markdown,
    index_titles,
)

ZOTERO_SELECT_PREFIX = "zotero://" + "select/library/items/"

_LONG_ABSTRACT = (
    "This is a deliberately long abstract that exceeds the default PyYAML dump width "
    "of eighty characters so that re-serializing it would reflow the lines and change "
    "the bytes on disk even though the content is identical."
)


def _existing_note(tmp_path):
    note = tmp_path / "smith2023.md"
    note.write_text(
        "---\n"
        "aliases:\n- A Paper Title\n"
        "domain: research\n"
        "category: materials\n"
        "tags:\n- Graphene\n"
        f"sources:\n- {ZOTERO_SELECT_PREFIX}ABC\n"
        "status: reading\n"
        f"abstract: {_LONG_ABSTRACT}\n"
        "---\n## Notes\nSome body text.\n",
        encoding="utf-8",
    )
    return note


def test_generate_merged_markdown_skips_write_when_unchanged(tmp_path):
    note = _existing_note(tmp_path)
    before = note.read_text(encoding="utf-8")
    data = {
        "title": "A Paper Title",
        "tags_list": ["Graphene"],
        "abstract": _LONG_ABSTRACT,
        "pdf_path": "",
        "citation_key": "smith2023",
        "zotero_link": f"{ZOTERO_SELECT_PREFIX}ABC",
    }
    generate_merged_markdown(data, "research", "materials", note)
    # Nothing changed semantically -> the file is left byte-identical (no reflow churn).
    assert note.read_text(encoding="utf-8") == before


def test_generate_merged_markdown_writes_when_abstract_filled(tmp_path):
    note = tmp_path / "smith2023.md"
    note.write_text(
        "---\naliases:\n- A Paper Title\ndomain: research\ncategory: materials\n"
        "tags:\n- Graphene\nsources:\n- zotero:smith2023\nstatus: unread\n"
        "abstract: ''\n---\n## Abstract\n\n",
        encoding="utf-8",
    )
    data = {
        "title": "A Paper Title",
        "tags_list": ["Graphene"],
        "abstract": "A freshly fetched abstract.",
        "pdf_path": "",
        "citation_key": "smith2023",
        "zotero_link": "",
    }
    generate_merged_markdown(data, "research", "materials", note)
    assert "A freshly fetched abstract." in note.read_text(encoding="utf-8")


def test_extract_item_data():
    mock_item = {
        "id": "smith2023",
        "title": "A Great Paper",
        "author": [{"family": "Smith", "given": "John"}, {"family": "Doe", "given": "Jane"}],
        "issued": {"date-parts": [["2023", 5]]},
        "abstract": "This is the abstract.",
        "URL": "https://example.com/paper",
    }

    result = extract_item_data(mock_item)

    assert result["citation_key"] == "smith2023"
    assert result["title"] == "A Great Paper"
    assert result["authors"] == "John Smith, Jane Doe"
    assert result["year"] == "2023"
    assert result["abstract"] == "This is the abstract."
    assert result["url"] == "https://example.com/paper"


def test_generate_markdown(tmp_path):
    data = {
        "citation_key": "smith2023",
        "title": "A Great Paper",
        "authors": "John Smith",
        "year": "2023",
        "abstract": "Abstract text.",
        "url": "http://example.com",
    }

    output_dir = tmp_path / "wiki" / "research"
    output_dir.mkdir(parents=True)

    # Attempt to generate
    file_path = generate_markdown(data, "research", "papers", output_dir)

    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")

    # Check frontmatter boundaries
    parts = content.split("---")
    assert len(parts) >= 3

    # Parse YAML frontmatter
    frontmatter = yaml.safe_load(parts[1])
    assert frontmatter["domain"] == "research"
    assert frontmatter["category"] == "papers"
    assert "zotero_import" in frontmatter["tags"]
    assert frontmatter["sources"] == ["zotero:smith2023"]

    # Check body
    body = parts[2]
    assert "# A Great Paper" in body
    assert "**Authors:** John Smith" in body


def test_index_titles_detects_cross_citekey_duplicate(tmp_path):
    output_dir = tmp_path / "wiki" / "research"
    output_dir.mkdir(parents=True)
    generate_markdown(
        {
            "citation_key": "smith2023",
            "title": "A Great Long Paper Title On Actuators",
            "authors": "John Smith",
            "year": "2023",
            "abstract": "Abstract text.",
            "url": "http://example.com",
        },
        "research",
        "papers",
        output_dir,
    )
    idx = index_titles(output_dir)
    norm = _norm_title("A Great Long Paper Title On Actuators")
    # the existing note is indexed by its normalized title
    assert idx.get(norm, "").endswith("smith2023.md")
    # a re-ingest under a BBT 'a' citekey shares the title -> caught as a duplicate
    dup_path = output_dir / "smith2023a.md"
    existing = idx.get(norm)
    assert existing is not None and existing != str(dup_path)
