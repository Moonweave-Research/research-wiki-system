import warnings


ALLOWED_CATEGORIES = {"materials", "concepts", "general"}


def validate_frontmatter(data: dict) -> tuple[bool, str]:
    required_keys = {"domain", "category", "tags", "sources"}
    if not isinstance(data, dict):
        return False, "Frontmatter is not a dictionary"
    if not required_keys.issubset(data.keys()):
        missing = required_keys - data.keys()
        return False, f"Missing required keys: {', '.join(missing)}"
    if data["domain"] not in ["research", "general"]:
        return False, f"Invalid domain: {data['domain']}. Must be 'research' or 'general'"
    if not isinstance(data["category"], str):
        return False, f"Invalid category type: {type(data['category']).__name__}. Must be a string"
    if data["category"] not in ALLOWED_CATEGORIES:
        return (
            False,
            f"Invalid category: {data['category']!r}. Must be one of {sorted(ALLOWED_CATEGORIES)}",
        )
    if not isinstance(data["tags"], list):
        return False, "Tags must be a list"
    if not isinstance(data["sources"], list):
        return False, "Sources must be a list"
    if "aliases" in data:
        if not isinstance(data["aliases"], list):
            return False, "aliases must be a list"
        if len(data["aliases"]) == 0:
            warnings.warn("aliases is an empty list")
    else:
        warnings.warn("aliases field is missing")
    return True, ""
