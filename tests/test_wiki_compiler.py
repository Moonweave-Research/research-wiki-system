import os
from scripts.core.wiki_compiler import generate_wiki_page, slugify

def test_slugify():
    assert slugify("Test Concept!") == "test_concept"
    assert slugify("Invalid/Path:Name") == "invalidpathname"
    assert slugify("   Spaces  Here   ") == "spaces_here"

def test_generate_wiki_page(tmp_path):
    output_dir = tmp_path / "wiki"
    
    mock_llm_data = {
        "title": "Test Concept?",
        "domain": "research",
        "category": "concepts",
        "tags": ["test"],
        "sources": ["test:1"],
        "content": "This is a test concept."
    }
    
    filepath = generate_wiki_page(mock_llm_data, str(output_dir))
    assert os.path.exists(filepath)
    assert filepath.endswith("test_concept.md")
    
    # Test collision
    filepath_2 = generate_wiki_page(mock_llm_data, str(output_dir))
    assert os.path.exists(filepath_2)
    assert filepath_2.endswith("test_concept_1.md")
