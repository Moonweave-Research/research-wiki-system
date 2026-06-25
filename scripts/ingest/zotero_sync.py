import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import yaml
import re
from pathlib import Path


ACRONYM_TAGS = {"mems", "ito", "mri", "afm", "xrd", "fea", "sem", "tem", "uv", "ir", "nir"}


def make_safe_key(citation_key: str) -> str:
    return "".join([c if c.isalnum() or c in " _-가-힣" else "_" for c in citation_key])


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "")).strip().lower()


def index_titles(output_dir: Path) -> dict[str, str]:
    """Map normalized note title -> existing note path, for cross-citekey dedup.

    The note's title is its first alias (zotero_sync writes aliases:[title]). Used
    so a paper already present under a different citekey (e.g. a BBT a/b
    disambiguation of the same Zotero item) is detected instead of re-created.
    """
    by_title: dict[str, str] = {}
    for note in sorted(output_dir.glob("*.md")):
        try:
            fm, _ = parse_existing_note(note)
        except Exception:
            continue
        aliases = fm.get("aliases") or []
        title = aliases[0] if aliases else ""
        norm = _norm_title(title)
        if len(norm) > 8:
            by_title.setdefault(norm, str(note))
    return by_title


def _normalize_tag(tag: str) -> str:
    if tag == "zotero_import":
        return tag
    if tag.lower() in ACRONYM_TAGS:
        return tag.upper()
    return tag.title()


def apply_auto_tags(title: str, abstract: str, existing_tags: list) -> list:
    tags_list = existing_tags.copy()
    text_to_search = (title + " " + abstract).lower()
    auto_keywords = [
        "actuator",
        "sensor",
        "graphene",
        "polymer",
        "elastomer",
        "liquid crystal",
        "hydrogel",
        "piezoelectric",
        "dielectric",
        "electrostatic",
        "mems",
        "robotics",
        "wearable",
        "energy storage",
        "supercapacitor",
        "solar cell",
        "ferroelectric",
        "nanocomposite",
        "3d printing",
        "4d printing",
        "soft robotics",
        "artificial muscle",
        "ionic",
        "biomimetic",
        "stretchable",
        "flexible",
    ]

    current_tags_lower = [t.lower() for t in tags_list]
    for kw in auto_keywords:
        if kw in text_to_search and kw not in current_tags_lower:
            tags_list.append(kw)

    if not tags_list:
        tags_list = ["zotero_import"]

    normalized_tags = list(dict.fromkeys(_normalize_tag(t) for t in tags_list))
    return normalized_tags


def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_abstract_from_crossref(doi: str) -> str:
    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "ResearchWiki/1.0 (mailto:research@example.com)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
            abstract = body["message"]["abstract"]
            return strip_html_tags(abstract)
    except Exception:
        return ""


def build_collection_map(data: dict) -> dict[int, list[str]]:
    collection_map: dict[int, list[str]] = {}
    for uuid, coll in data.get("collections", {}).items():
        for item_id in coll.get("items", []):
            if item_id not in collection_map:
                collection_map[item_id] = []
            collection_map[item_id].append(uuid)
    return collection_map


def resolve_category(
    item_id: int,
    collection_map: dict[int, list[str]],
    collections_dict: dict,
    fallback_category: str | None,
) -> str | None:
    uuids = collection_map.get(item_id, [])
    if not uuids:
        return fallback_category

    leaf_uuids = [
        uuid for uuid in uuids if not collections_dict.get(uuid, {}).get("collections", [])
    ]

    if leaf_uuids:
        return collections_dict[leaf_uuids[0]]["name"]
    return collections_dict[uuids[0]]["name"]


def parse_existing_note(file_path: Path) -> tuple[dict, str]:
    text = file_path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    body = text[m.end() :]
    return fm, body


def merge_frontmatter(existing_fm: dict, new_data: dict, domain: str, category: str) -> dict:
    merged = existing_fm.copy()

    # Preserve existing status — never overwrite
    if "status" not in merged:
        merged["status"] = "unread"

    # Merge tags: union of existing and new, deduplicated, normalized
    existing_tags = [_normalize_tag(t) for t in (merged.get("tags") or [])]
    new_tags = [_normalize_tag(t) for t in (new_data.get("tags_list") or [])]
    merged["tags"] = list(
        dict.fromkeys(existing_tags + [t for t in new_tags if t not in existing_tags])
    )

    # Fill abstract only if empty
    if not merged.get("abstract") and new_data.get("abstract"):
        merged["abstract"] = new_data["abstract"]

    # Fill pdf_path only if empty
    if not merged.get("pdf_path") and new_data.get("pdf_path"):
        merged["pdf_path"] = new_data["pdf_path"]

    # Always update these fields
    merged["aliases"] = [new_data["title"]]
    merged["domain"] = domain
    merged["sources"] = (
        [new_data["zotero_link"]]
        if new_data.get("zotero_link")
        else [f"zotero:{new_data['citation_key']}"]
    )

    # Set category if not already present
    if "category" not in merged:
        merged["category"] = category

    return merged


def generate_merged_markdown(data: dict, domain: str, category: str, file_path: Path) -> Path:
    existing_fm, body = parse_existing_note(file_path)
    original_body = body
    merged_fm = merge_frontmatter(existing_fm, data, domain, category)

    # Replace abstract section body if existing is empty and new data has abstract
    new_abstract = data.get("abstract", "")
    if new_abstract:
        abstract_section_match = re.search(r"## Abstract\n(.*?)(?=\n## |\Z)", body, re.DOTALL)
        if abstract_section_match and not abstract_section_match.group(1).strip():
            replacement = "## Abstract\n" + new_abstract + "\n"
            body = re.sub(
                r"## Abstract\n.*?(?=\n## |\Z)",
                lambda m: replacement,
                body,
                count=1,
                flags=re.DOTALL,
            )

    # Skip the write when nothing changed semantically — re-dumping unchanged
    # frontmatter reflows folded scalars (e.g. abstract) and churns the file on
    # every sync, which is pure noise in the working tree.
    if merged_fm == existing_fm and body == original_body:
        return file_path

    yaml_str = yaml.dump(merged_fm, sort_keys=False, allow_unicode=True)
    content = f"---\n{yaml_str}---{body}"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def extract_item_data(item: dict) -> dict:
    # Handle authors
    authors_list = item.get("author", [])
    author_names = []
    for a in authors_list:
        if "literal" in a:
            author_names.append(a["literal"])
        else:
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                author_names.append(name)
    authors_str = ", ".join(author_names) if author_names else "Unknown"

    # Handle year
    year = "Unknown"
    issued = item.get("issued", {})
    if "date-parts" in issued and issued["date-parts"] and issued["date-parts"][0]:
        year = str(issued["date-parts"][0][0])

    title = item.get("title", "Untitled")
    abstract = item.get("abstract", "")

    # Extract existing tags if any
    raw_tags = item.get("tags", [])
    existing_tags = []
    for t in raw_tags:
        if isinstance(t, dict) and t.get("tag"):
            existing_tags.append(t.get("tag").strip())
        elif isinstance(t, str):
            existing_tags.append(t.strip())

    tags_list = apply_auto_tags(title, abstract, existing_tags)

    return {
        "citation_key": str(item.get("id", "unknown")),
        "title": title,
        "authors": authors_str,
        "year": year,
        "abstract": abstract,
        "doi": item.get("DOI", ""),
        "url": item.get("URL") or item.get("DOI", ""),
        "zotero_link": "",
        "tags_list": tags_list,
        "pdf_path": "",
    }


def generate_markdown(data: dict, domain: str, category: str, output_dir: Path) -> Path:
    # Existing frontmatter tags
    item_tags = data.get("tags_list", ["zotero_import"])

    frontmatter = {
        "aliases": [data["title"]],
        "domain": domain,
        "category": category,
        "tags": item_tags,
        "sources": [data["zotero_link"]]
        if data.get("zotero_link")
        else [f"zotero:{data['citation_key']}"],
        "pdf_path": data.get("pdf_path", ""),
        "status": "unread",
    }

    yaml_str = yaml.dump(frontmatter, sort_keys=False, allow_unicode=True)

    zotero_action = (
        f"- **Zotero:** [Open in Zotero]({data['zotero_link']})" if data.get("zotero_link") else ""
    )

    content = f"""---
{yaml_str}---
# {data["title"]}

- **Authors:** {data["authors"]}
- **Year:** {data["year"]}
- **URL/DOI:** {data["url"]}
{zotero_action}

## Abstract
{data["abstract"]}
"""

    safe_key = make_safe_key(data["citation_key"])
    file_path = output_dir / f"{safe_key}.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def extract_from_bib(content: str) -> list[dict]:
    # Regex to find @type{key, ...}
    entries = re.finditer(r"@(\w+)\{([^,]+),\s*(.*?)\n\}", content, re.DOTALL)
    results = []

    for entry in entries:
        citation_key = entry.group(2).strip()
        fields_str = entry.group(3)

        # Simple regex to extract fields like: title = { ... },
        field_matches = re.finditer(
            r'\s*(\w+)\s*=\s*[\{"](.*?)[\}"]\s*[,]?\s*\n', fields_str, re.DOTALL
        )
        fields = {m.group(1).lower(): m.group(2).strip() for m in field_matches}

        # Normalize authors (replace { ... } or inner braces)
        authors = fields.get("author", "Unknown").replace("{", "").replace("}", "")

        title = fields.get("title", "Untitled").replace("{", "").replace("}", "")
        abstract = fields.get("abstract", "")
        # BibTeX usually doesn't have easily parsed tags, so start empty
        tags_list = apply_auto_tags(title, abstract, [])

        results.append(
            {
                "citation_key": citation_key,
                "title": title,
                "authors": authors,
                "year": fields.get("year", "Unknown"),
                "abstract": abstract,
                "doi": fields.get("doi", ""),
                "url": fields.get("url") or fields.get("doi", ""),
                "zotero_link": "",
                "tags_list": tags_list,
                "pdf_path": "",
            }
        )
    return results


def extract_from_bb_json(data: dict) -> list[dict]:
    items = data.get("items", [])
    collections_dict = data.get("collections", {})
    collection_map = build_collection_map(data)
    results = []
    for item in items:
        # Better BibTeX JSON uses 'citationKey' (CamelCase)
        # Fallback to 'itemKey' (Zotero Native) if missing
        citation_key = item.get("citationKey") or item.get("citekey") or item.get("itemKey")
        if not citation_key:
            continue

        # Only process primary items (not attachments/notes directly)
        item_type = item.get("itemType")
        if item_type in ["attachment", "note"]:
            continue

        # Extract authors
        creators = item.get("creators", [])
        author_names = []
        for c in creators:
            name = f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
            if not name:
                name = c.get("name", "")
            if name:
                author_names.append(name)
        authors_str = ", ".join(author_names) if author_names else "Unknown"

        # Date
        date = item.get("date", "Unknown")
        year = date[:4] if len(date) >= 4 and date[:4].isdigit() else "Unknown"

        # Capture existing tags
        raw_tags = item.get("tags", [])
        existing_tags = []
        for t in raw_tags:
            tag_str = ""
            if isinstance(t, dict) and t.get("tag"):
                tag_str = t.get("tag").strip()
            elif isinstance(t, str):
                tag_str = t.strip()

            # Filter out single characters, pure numbers, or strange symbols
            if (
                tag_str
                and not tag_str.isdigit()
                and len(tag_str) > 1
                and not re.match(r"^[^a-zA-Z0-9가-힣]*$", tag_str)
            ):
                existing_tags.append(tag_str)

        title = item.get("title", "Untitled")
        abstract = item.get("abstractNote") or item.get("abstract", "")
        tags_list = apply_auto_tags(title, abstract, existing_tags)

        # Extract first PDF attachment path (Google Drive absolute path from Zotero)
        pdf_path = ""
        for att in item.get("attachments", []):
            p = att.get("path", "")
            if p.lower().endswith(".pdf"):
                pdf_path = p
                break

        item_id = item.get("itemID")
        category = resolve_category(item_id, collection_map, collections_dict, None)

        results.append(
            {
                "citation_key": citation_key,
                "title": title,
                "authors": authors_str,
                "year": year,
                "abstract": abstract,
                "doi": item.get("DOI", ""),
                "url": item.get("uri", "") or item.get("DOI", ""),
                "zotero_link": item.get("select", ""),
                "tags_list": tags_list,
                "pdf_path": pdf_path,
                "category": category,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sync Zotero JSON (CSL/BetterBibTeX) or BibTeX to LLM Wiki Markdown."
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to the Zotero JSON or BibTeX export file"
    )
    parser.add_argument(
        "--domain", required=True, choices=["research", "general"], help="Target domain"
    )
    parser.add_argument("--category", required=False, default=None, help="Target category")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    mode_group.add_argument(
        "--merge",
        action="store_true",
        help="Smart update: preserves status/tags, fills empty abstract/pdf_path",
    )

    parser.add_argument(
        "--fetch-abstracts",
        action="store_true",
        help="Fetch missing abstracts from Crossref via DOI",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview counts without writing any files"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable same-title dedup (otherwise a paper already present under a "
        "different citekey — e.g. a BBT a/b disambiguation — is skipped, not re-created)",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path("wiki") / args.domain
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    items_data = []
    is_bb_json = False

    if args.input_file.suffix.lower() == ".json":
        with open(args.input_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):  # CSL-JSON
                    items_data = [extract_item_data(item) for item in data]
                elif isinstance(data, dict) and "items" in data:  # Better BibTeX JSON
                    is_bb_json = True
                    items_data = extract_from_bb_json(data)
                else:
                    print("Error: Unknown JSON format.", file=sys.stderr)
                    sys.exit(1)
            except json.JSONDecodeError:
                print("Error: Invalid JSON file.", file=sys.stderr)
                sys.exit(1)
    elif args.input_file.suffix.lower() == ".bib":
        with open(args.input_file, "r", encoding="utf-8") as f:
            content = f.read()
            items_data = extract_from_bib(content)
    else:
        print(f"Error: Unsupported file format: {args.input_file.suffix}", file=sys.stderr)
        sys.exit(1)

    if not is_bb_json and args.category is None:
        print("Error: --category is required for CSL-JSON and BibTeX inputs.", file=sys.stderr)
        sys.exit(1)

    if args.fetch_abstracts:
        for item in items_data:
            doi = item.get("doi", "")
            if not item.get("abstract") and doi:
                print(f"Fetching abstract for {doi}...", end=" ", flush=True)
                result = fetch_abstract_from_crossref(doi)
                if result:
                    item["abstract"] = result
                    item["tags_list"] = apply_auto_tags(item["title"], result, item["tags_list"])
                print("OK" if result else "not found")
                time.sleep(0.5)

    created_count = 0
    updated_count = 0
    skipped_count = 0
    duplicate_count = 0
    title_index = {} if args.no_dedup else index_titles(output_dir)

    for item_data in items_data:
        item_category = item_data.get("category") or args.category or "uncategorized"
        file_path = output_dir / f"{make_safe_key(item_data['citation_key'])}.md"
        norm_title = _norm_title(item_data.get("title", ""))

        if not file_path.exists():
            existing = title_index.get(norm_title) if len(norm_title) > 8 else None
            if existing and existing != str(file_path):
                duplicate_count += 1
                print(f"  duplicate of {Path(existing).name}: skip {file_path.name}")
                continue
            if not args.dry_run:
                generate_markdown(item_data, args.domain, item_category, output_dir)
            created_count += 1
            if len(norm_title) > 8:
                title_index.setdefault(norm_title, str(file_path))
        elif args.overwrite:
            if not args.dry_run:
                generate_markdown(item_data, args.domain, item_category, output_dir)
            updated_count += 1
        elif args.merge:
            if not args.dry_run:
                generate_merged_markdown(item_data, args.domain, item_category, file_path)
            updated_count += 1
        else:
            skipped_count += 1

    if args.dry_run:
        print(
            f"Would create: {created_count} / "
            f"Would update: {updated_count} / "
            f"Would skip: {skipped_count} / "
            f"Would skip as duplicate: {duplicate_count}"
        )
        return

    print(
        f"Sync complete! Created: {created_count}, "
        f"Updated: {updated_count}, Skipped: {skipped_count}, "
        f"Duplicates skipped: {duplicate_count}"
    )

    from datetime import date

    update_index(Path("wiki"))

    log_path = Path("wiki/log.md")
    if log_path.exists():
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n## [{date.today()}] sync | Created: {created_count}, "
                f"Updated: {updated_count}, Skipped: {skipped_count}"
                f" | {args.input_file.name} → wiki/{args.domain}/\n"
            )


def update_index(wiki_dir: Path) -> None:
    sections: dict[str, list[str]] = {"research": [], "general": []}
    skip_names = {"index.md", "log.md", "00_Dashboard.md", "Schema.md"}

    for md_file in sorted(wiki_dir.rglob("*.md")):
        if md_file.name in skip_names:
            continue
        if "meta" in md_file.parts:
            continue

        text = md_file.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not m:
            continue
        try:
            fm = yaml.safe_load(m.group(1))
        except yaml.YAMLError:
            continue

        domain = fm.get("domain", md_file.parent.name)
        aliases = fm.get("aliases") or []
        title = aliases[0] if aliases else md_file.stem

        abstract_m = re.search(r"## Abstract\n(.+)", text)
        snippet = (abstract_m.group(1)[:80] + "…") if abstract_m else ""

        # Use standard markdown links, not [[wikilinks]], to avoid graph view pollution
        rel_path = md_file.relative_to(wiki_dir)
        entry = f"- [{title[:70]}]({rel_path}) `{md_file.stem}`"
        if snippet:
            entry += f"\n  {snippet}"

        if domain in sections:
            sections[domain].append(entry)

    from datetime import date as _date

    lines = [f"# Wiki Index\n\n*Updated: {_date.today()}*\n"]
    for section, entries in sections.items():
        if entries:
            lines.append(f"\n## {section} ({len(entries)})\n\n")
            lines.extend(e + "\n" for e in entries)

    (wiki_dir / "index.md").write_text("".join(lines), encoding="utf-8")
    print(f"Index updated: {sum(len(v) for v in sections.values())} pages")


if __name__ == "__main__":
    main()
