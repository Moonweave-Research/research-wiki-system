from scripts.core.wiki_core import validate_frontmatter

def test_validate_frontmatter_valid():
    valid_data = {
        "domain": "research",
        "category": "materials",
        "tags": ["optics"],
        "sources": ["zotero:123"]
    }
    is_valid, msg = validate_frontmatter(valid_data)
    assert is_valid

def test_validate_frontmatter_invalid_domain():
    invalid_data = {
        "domain": "personal",
        "category": "materials",
        "tags": [],
        "sources": []
    }
    is_valid, msg = validate_frontmatter(invalid_data)
    assert not is_valid
    assert "Invalid domain" in msg

def test_validate_frontmatter_missing_keys():
    invalid_data = {
        "domain": "general"
    }
    is_valid, msg = validate_frontmatter(invalid_data)
    assert not is_valid
    assert "Missing required keys" in msg
    
def test_validate_frontmatter_invalid_category_type():
    invalid_data = {
        "domain": "research",
        "category": 123,
        "tags": [],
        "sources": []
    }
    is_valid, msg = validate_frontmatter(invalid_data)
    assert not is_valid
    assert "Invalid category type" in msg
