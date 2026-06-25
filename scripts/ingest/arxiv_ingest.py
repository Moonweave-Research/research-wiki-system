import argparse
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import yaml

try:
    from scripts.ingest.zotero_sync import _normalize_tag, apply_auto_tags
except ModuleNotFoundError:
    from zotero_sync import _normalize_tag, apply_auto_tags  # type: ignore[no-redef]

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


def safe_arxiv_id(raw_id: str) -> tuple[str, str]:
    """Return (safe_id, version_str). safe_id has no dots, slashes, or version suffixes."""
    raw_id = raw_id.strip()
    version = ""
    version_match = re.search(r"v(\d+)$", raw_id)
    if version_match:
        version = f"v{version_match.group(1)}"
        raw_id = raw_id[: version_match.start()]
    # Replace dots and slashes with underscores
    sanitized = re.sub(r"[./]", "_", raw_id)
    return f"arxiv_{sanitized}", version


def canonical_arxiv_id(raw_id: str) -> str:
    """Return the ID without version suffix, with original separators preserved."""
    raw_id = raw_id.strip()
    return re.sub(r"v\d+$", "", raw_id)


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    """Fetch and parse arXiv Atom XML for the given ID. Returns parsed metadata dict."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "ResearchWiki/1.0 (mailto:research@example.com)"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        xml_bytes = resp.read()
    return parse_atom_xml(xml_bytes)


def parse_atom_xml(xml_bytes: bytes) -> dict:
    """Parse arXiv Atom XML bytes into a metadata dict."""
    root = ET.fromstring(xml_bytes)
    entry = root.find(f"{{{ATOM_NS}}}entry")
    if entry is None:
        raise ValueError("No entry found in arXiv Atom response")

    # arXiv signals "ID not found" by returning a 200 with an entry whose <id>
    # points to api/errors. Without this guard, we'd happily create a note with
    # empty title/authors/abstract and a confusing arxiv: source.
    id_el = entry.find(f"{{{ATOM_NS}}}id")
    if id_el is not None and id_el.text and "/api/errors" in id_el.text:
        raise ValueError(f"arXiv returned an error entry: {id_el.text.strip()}")

    def text(tag: str, ns: str = ATOM_NS) -> str:
        el = entry.find(f"{{{ns}}}{tag}")
        return el.text.strip() if el is not None and el.text else ""

    title = text("title").replace("\n", " ").replace("  ", " ")
    abstract = text("summary").replace("\n", " ").strip()
    published = text("published")  # e.g. "2024-01-08T00:00:00Z"
    year = published[:4] if len(published) >= 4 else "Unknown"

    # Authors
    authors = []
    for author_el in entry.findall(f"{{{ATOM_NS}}}author"):
        name_el = author_el.find(f"{{{ATOM_NS}}}name")
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    # primary_category (arxiv namespace)
    primary_cat_el = entry.find(f"{{{ARXIV_NS}}}primary_category")
    if primary_cat_el is None:
        # fallback: category element in atom namespace
        primary_cat_el = entry.find(f"{{{ATOM_NS}}}category")
    primary_category = ""
    if primary_cat_el is not None:
        primary_category = primary_cat_el.get("term", "")

    # DOI (arxiv namespace)
    doi_el = entry.find(f"{{{ARXIV_NS}}}doi")
    doi = doi_el.text.strip() if doi_el is not None and doi_el.text else ""

    # journal_ref (arxiv namespace) — used for publisher filters
    journal_ref_el = entry.find(f"{{{ARXIV_NS}}}journal_ref")
    journal_ref = (
        journal_ref_el.text.strip() if journal_ref_el is not None and journal_ref_el.text else ""
    )

    # PDF link
    pdf_url = ""
    for link_el in entry.findall(f"{{{ATOM_NS}}}link"):
        if link_el.get("title") == "pdf":
            pdf_url = link_el.get("href", "")
            break

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "year": year,
        "primary_category": primary_category,
        "doi": doi,
        "journal_ref": journal_ref,
        "pdf_url": pdf_url,
    }


# MDPI publisher detection. We rely on the DOI prefix only — substring matching
# against journal_ref produces false positives for major non-MDPI journals
# ("Materials Today", "Advanced Materials", "Sensors and Actuators", etc.) which
# share substrings with MDPI titles ("Materials", "Sensors"). DOI 10.3390/ is
# unambiguous: it's MDPI's CrossRef publisher prefix and nobody else uses it.
def is_mdpi(meta: dict) -> bool:
    doi = (meta.get("doi") or "").lower()
    return doi.startswith("10.3390/")


def download_pdf(url: str, dest: Path) -> None:
    """Download a PDF atomically: stream to <dest>.part then os.replace.
    Prevents truncated PDFs from being treated as cached on next run.
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": "ResearchWiki/1.0 (mailto:research@example.com)"}
    )
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            tmp.write_bytes(resp.read())
        import os

        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def build_frontmatter(
    *,
    raw_id: str,
    safe_id: str,
    version: str,
    meta: dict,
    pdf_path: str,
    domain: str,
    category: str,
) -> dict:
    initial_tags = [meta["primary_category"]] if meta["primary_category"] else []
    tags = apply_auto_tags(meta["title"], meta["abstract"], initial_tags)

    return {
        "aliases": [meta["title"]],
        "domain": domain,
        "category": category,
        "tags": tags,
        "sources": [f"arxiv:{raw_id}"],
        "pdf_path": pdf_path,
        "status": "unread",
        "abstract": meta["abstract"],
        "arxiv_id": raw_id,
        "arxiv_version": version,
    }


def merge_arxiv_frontmatter(existing_fm: dict, new_fm: dict) -> dict:
    """Merge arXiv frontmatter: preserve status, union tags, fill empty fields only."""
    merged = existing_fm.copy()

    # Preserve existing status
    if "status" not in merged:
        merged["status"] = "unread"

    # Union tags
    existing_tags = [_normalize_tag(t) for t in (merged.get("tags") or [])]
    new_tags = [_normalize_tag(t) for t in (new_fm.get("tags") or [])]
    merged["tags"] = list(
        dict.fromkeys(existing_tags + [t for t in new_tags if t not in existing_tags])
    )

    # Fill abstract only if empty
    if not merged.get("abstract") and new_fm.get("abstract"):
        merged["abstract"] = new_fm["abstract"]

    # Fill pdf_path only if empty
    if not merged.get("pdf_path") and new_fm.get("pdf_path"):
        merged["pdf_path"] = new_fm["pdf_path"]

    # Always update these
    merged["aliases"] = new_fm["aliases"]
    merged["domain"] = new_fm["domain"]
    merged["sources"] = new_fm["sources"]
    merged["arxiv_id"] = new_fm["arxiv_id"]
    merged["arxiv_version"] = new_fm["arxiv_version"]

    if "category" not in merged:
        merged["category"] = new_fm["category"]

    return merged


def build_note_body(meta: dict) -> str:
    authors_str = ", ".join(meta["authors"]) if meta["authors"] else "Unknown"
    abstract_text = meta["abstract"] if meta["abstract"] else ""
    doi_line = f"- **DOI:** {meta['doi']}" if meta["doi"] else ""

    lines = [
        f"# {meta['title']}",
        "",
        f"- **Authors:** {authors_str}",
        f"- **Year:** {meta['year']}",
    ]
    if doi_line:
        lines.append(doi_line)
    lines.extend(
        [
            "",
            "## Abstract",
            abstract_text,
            "",
        ]
    )
    return "\n".join(lines)


def write_note(file_path: Path, frontmatter: dict, body: str) -> None:
    yaml_str = yaml.dump(frontmatter, sort_keys=False, allow_unicode=True)
    content = f"---\n{yaml_str}---\n{body}"
    file_path.write_text(content, encoding="utf-8")


def append_log(log_path: Path, created: int, merged: int, skipped: int, failed: int) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = (
        f"\n## [{now_str}] arxiv_ingest | "
        f"created {created}, merged {merged}, skipped {skipped}, failed {failed}\n"
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def ingest_one(
    *,
    raw_id: str,
    vault: Path,
    cache_dir: Path,
    domain: str,
    category: str,
    merge: bool,
    overwrite: bool,
    dry_run: bool,
    no_download: bool,
    emit_json: bool,
    filter_mdpi: bool = True,
) -> dict:
    """Process a single arXiv ID. Returns a result dict with keys: arxiv_id, action, note_path, pdf_path, error."""
    canon_id = canonical_arxiv_id(raw_id)
    safe_id, version = safe_arxiv_id(raw_id)

    result: dict = {
        "arxiv_id": canon_id,
        "action": "failed",
        "note_path": "",
        "pdf_path": "",
        "error": "",
    }

    try:
        meta = fetch_arxiv_metadata(canon_id)
    except Exception as exc:
        result["error"] = str(exc)
        if emit_json:
            print(json.dumps(result))
        return result

    if filter_mdpi and is_mdpi(meta):
        result["action"] = "filtered_mdpi"
        result["error"] = f"journal_ref={meta.get('journal_ref', '')} doi={meta.get('doi', '')}"
        if emit_json:
            print(json.dumps(result))
        return result

    # Determine pdf_path
    pdf_dest = cache_dir / f"{safe_id}.pdf"
    pdf_path_str = str(pdf_dest) if not no_download else ""

    note_dir = vault / "wiki" / domain
    note_path = note_dir / f"{safe_id}.md"
    result["note_path"] = str(note_path)
    result["pdf_path"] = str(pdf_dest)

    if note_path.exists():
        if merge:
            if not dry_run:
                existing_fm_text = note_path.read_text(encoding="utf-8")
                m = re.match(r"^---\n(.*?)\n---", existing_fm_text, re.DOTALL)
                existing_fm: dict = {}
                if m:
                    try:
                        existing_fm = yaml.safe_load(m.group(1)) or {}
                    except yaml.YAMLError as exc:
                        # Refuse to clobber the user's existing note with a fresh
                        # frontmatter when the existing one is malformed — that
                        # would silently destroy custom edits the user made.
                        result["action"] = "failed"
                        result["error"] = f"existing_frontmatter_malformed: {exc}"
                        if emit_json:
                            print(json.dumps(result))
                        return result
                new_fm = build_frontmatter(
                    raw_id=canon_id,
                    safe_id=safe_id,
                    version=version,
                    meta=meta,
                    pdf_path=pdf_path_str,
                    domain=domain,
                    category=category,
                )
                merged_fm = merge_arxiv_frontmatter(existing_fm, new_fm)
                body = build_note_body(meta)
                write_note(note_path, merged_fm, body)
            result["action"] = "merged"
        elif overwrite:
            if not dry_run:
                new_fm = build_frontmatter(
                    raw_id=canon_id,
                    safe_id=safe_id,
                    version=version,
                    meta=meta,
                    pdf_path=pdf_path_str,
                    domain=domain,
                    category=category,
                )
                body = build_note_body(meta)
                write_note(note_path, new_fm, body)
            result["action"] = "merged"
        else:
            result["action"] = "skipped"
            if emit_json:
                print(json.dumps(result))
            return result
    else:
        if not dry_run:
            note_dir.mkdir(parents=True, exist_ok=True)
            new_fm = build_frontmatter(
                raw_id=canon_id,
                safe_id=safe_id,
                version=version,
                meta=meta,
                pdf_path=pdf_path_str,
                domain=domain,
                category=category,
            )
            body = build_note_body(meta)
            write_note(note_path, new_fm, body)
        result["action"] = "created"

    # PDF download. On failure we must NOT leave a note claiming a phantom pdf_path:
    # the downstream wiki_pdf_cache would record a 'missing pdf' error on every refresh,
    # and the orchestrator would treat this ingest as a success.
    if not no_download and not dry_run and meta.get("pdf_url"):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if not pdf_dest.exists():
                download_pdf(meta["pdf_url"], pdf_dest)
        except Exception as exc:
            result["error"] = f"pdf_download_failed: {exc}"
            result["action"] = "failed_pdf"
            # Rewrite the note with pdf_path cleared so the next refresh-graph
            # run does not record a phantom 'missing pdf' against this note.
            if note_path.exists():
                try:
                    text = note_path.read_text(encoding="utf-8")
                    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
                    if m:
                        try:
                            current_fm = yaml.safe_load(m.group(1)) or {}
                            current_fm["pdf_path"] = ""
                            body = text[m.end() :]
                            write_note(note_path, current_fm, body)
                        except yaml.YAMLError:
                            pass
                except OSError:
                    pass
            result["pdf_path"] = ""

    if emit_json:
        print(json.dumps(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into the research wiki.")
    parser.add_argument("--ids", required=True, help="Comma-separated arXiv IDs")
    parser.add_argument("--vault", type=Path, default=Path("."), help="Vault root directory")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "research-wiki" / "arxiv",
        help="Directory to store downloaded PDFs",
    )
    parser.add_argument("--domain", default="research", choices=["research", "general"])
    parser.add_argument("--category", default="arxiv")
    parser.add_argument("--merge", action="store_true", help="Merge into existing notes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing notes")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    parser.add_argument("--no-download", action="store_true", help="Skip PDF download")
    parser.add_argument("--json", action="store_true", dest="emit_json", help="Emit JSON per ID")
    parser.add_argument(
        "--no-filter-mdpi",
        action="store_true",
        help="Disable MDPI filter (default: skip MDPI-published papers via DOI 10.3390/ or journal_ref hint)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any ID fails (default: exit 0 unless all failed)",
    )

    args = parser.parse_args(argv)

    ids = [raw_id.strip() for raw_id in args.ids.split(",") if raw_id.strip()]
    if not ids:
        print("Error: no IDs provided", file=sys.stderr)
        return 1

    results = []
    for i, raw_id in enumerate(ids):
        if i > 0:
            time.sleep(4)
        result = ingest_one(
            raw_id=raw_id,
            vault=args.vault,
            cache_dir=args.cache_dir,
            domain=args.domain,
            category=args.category,
            merge=args.merge,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            no_download=args.no_download,
            emit_json=args.emit_json,
            filter_mdpi=not args.no_filter_mdpi,
        )
        results.append(result)

    counts = {
        "created": 0,
        "merged": 0,
        "skipped": 0,
        "filtered_mdpi": 0,
        "failed_pdf": 0,
        "failed": 0,
    }
    for res in results:
        action = res["action"]
        if action in counts:
            counts[action] += 1
        else:
            counts["failed"] += 1

    print(
        f"arxiv_ingest complete: created={counts['created']}, "
        f"merged={counts['merged']}, skipped={counts['skipped']}, "
        f"filtered_mdpi={counts['filtered_mdpi']}, "
        f"failed_pdf={counts['failed_pdf']}, failed={counts['failed']}"
    )

    # Append to log.md unless dry-run
    if not args.dry_run:
        log_path = args.vault / "wiki" / "log.md"
        if log_path.exists():
            append_log(
                log_path,
                counts["created"],
                counts["merged"],
                counts["skipped"],
                counts["failed"],
            )

    hard_failures = counts["failed"] + counts["failed_pdf"]
    if args.strict and hard_failures > 0:
        return 1
    if hard_failures > 0 and hard_failures == len(results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
