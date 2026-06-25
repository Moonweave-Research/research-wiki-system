#!/usr/bin/env python3
"""Build the SQLite link-graph index from the markdown vault.

Notes are nodes; wikilinks (``[[target]]``) are edges; backlinks are the reverse
of links. This is the lightweight Obsidian-style link graph that the future
read/browse UI reads — NOT the frozen GraphRAG semantic graph in legacy/graphrag.

The index is a full rebuild each run (idempotent and fast at the current vault
scale) and is gitignored under .graph_index/. Usage:

    uv run python -m scripts.core.linkgraph build
    uv run python -m scripts.core.linkgraph report   # broken links + orphans
"""

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# [[target]], [[target|alias]], [[target#heading]] — capture the target only.
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---(?:\r?\n|$)", re.DOTALL)

DEFAULT_VAULT = "wiki"
DEFAULT_DB = ".graph_index/linkgraph.sqlite"
INDEX_VERSION = 2

SCHEMA = """
CREATE TABLE notes (
    path TEXT PRIMARY KEY,
    title TEXT,
    domain TEXT,
    category TEXT,
    status TEXT,
    aliases_json TEXT,
    tags_json TEXT,
    mtime REAL
);
CREATE TABLE links (
    src_path TEXT,
    dst_target TEXT,
    dst_resolved INTEGER,
    kind TEXT
);
CREATE TABLE tags (
    tag TEXT,
    note_path TEXT
);
CREATE INDEX idx_links_dst ON links(dst_target);
CREATE INDEX idx_links_src ON links(src_path);
CREATE INDEX idx_tags_tag ON tags(tag);
CREATE VIRTUAL TABLE notes_fts USING fts5(path, title, body);
"""


def _meta_path(db_path: Path) -> Path:
    return db_path.with_suffix(db_path.suffix + ".meta.json")


def _vault_signature(vault_dir: Path) -> list[list[str | int]]:
    return [
        [str(path.relative_to(vault_dir)), path.stat().st_size, path.stat().st_mtime_ns]
        for path in sorted(vault_dir.rglob("*.md"))
    ]


def _current_index_counts(db_path: Path, signature: list[list[str | int]]) -> tuple[int, int] | None:
    meta_path = _meta_path(db_path)
    if not db_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != INDEX_VERSION:
            return None
        if meta.get("signature") != signature:
            return None
        return int(meta["notes"]), int(meta["links"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_meta(
    db_path: Path,
    signature: list[list[str | int]],
    notes_count: int,
    links_count: int,
) -> None:
    _meta_path(db_path).write_text(
        json.dumps(
            {
                "signature": signature,
                "notes": notes_count,
                "links": links_count,
                "version": INDEX_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass
class Note:
    path: str  # vault-relative posix path, e.g. "research/foo.md"
    stem: str  # filename without .md
    title: str
    domain: str | None
    category: str | None
    status: str | None
    aliases: list[str]
    tags: list[str]
    mtime: float
    targets: list[str] = field(default_factory=list)  # raw wikilink targets
    body: str = ""  # markdown after the frontmatter, for full-text search


@dataclass
class Link:
    src_path: str
    dst_target: str  # resolved note path if resolved, else the raw target
    dst_resolved: bool
    kind: str = "wikilink"


def _parse_frontmatter(text: str) -> dict:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _scan_one(md_path: Path, vault_dir: Path) -> Note:
    """Parse a single markdown file into a Note (no link resolution)."""
    rel = md_path.relative_to(vault_dir).as_posix()
    text = md_path.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    aliases = _as_list(fm.get("aliases"))
    parts = text.split("---", 2)
    body = parts[2] if text.startswith("---") and len(parts) >= 3 else text
    return Note(
        path=rel,
        stem=md_path.stem,
        title=str(fm.get("title") or (aliases[0] if aliases else md_path.stem)),
        domain=fm.get("domain"),
        category=fm.get("category"),
        status=fm.get("status"),
        aliases=aliases,
        tags=_as_list(fm.get("tags")),
        mtime=md_path.stat().st_mtime,
        targets=[m.strip() for m in WIKILINK_RE.findall(text)],
        body=body,
    )


def scan_vault(vault_dir: Path) -> tuple[list[Note], list[Link]]:
    """Parse every markdown file under vault_dir into notes + resolved links."""
    notes = [_scan_one(md_path, vault_dir) for md_path in sorted(vault_dir.rglob("*.md"))]

    # Resolution map: stem and each alias point to the note's path.
    key_to_path: dict[str, str] = {}
    for note in notes:
        key_to_path.setdefault(note.stem, note.path)
        for alias in note.aliases:
            key_to_path.setdefault(alias, note.path)

    links: list[Link] = []
    for note in notes:
        seen: set[tuple[str, str, bool, str]] = set()
        for target in note.targets:
            resolved_path = key_to_path.get(target)
            link = Link(
                src_path=note.path,
                dst_target=resolved_path if resolved_path else target,
                dst_resolved=resolved_path is not None,
            )
            key = (link.src_path, link.dst_target, link.dst_resolved, link.kind)
            if key in seen:
                continue
            seen.add(key)
            links.append(
                link
            )
    return notes, links


def broken_links(links: list[Link]) -> list[Link]:
    return [link for link in links if not link.dst_resolved]


def orphan_notes(notes: list[Note], links: list[Link]) -> list[Note]:
    """Notes with no outbound and no inbound resolved links."""
    has_outbound = {link.src_path for link in links if link.dst_resolved}
    has_inbound = {link.dst_target for link in links if link.dst_resolved}
    connected = has_outbound | has_inbound
    return [note for note in notes if note.path not in connected]


def build_index(vault_dir: Path, db_path: Path) -> tuple[int, int]:
    signature = _vault_signature(vault_dir)
    current = _current_index_counts(db_path, signature)
    if current is not None:
        return current
    notes, links = scan_vault(vault_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    n.path,
                    n.title,
                    n.domain,
                    n.category,
                    n.status,
                    json.dumps(n.aliases, ensure_ascii=False),
                    json.dumps(n.tags, ensure_ascii=False),
                    n.mtime,
                )
                for n in notes
            ],
        )
        conn.executemany(
            "INSERT INTO links VALUES (?, ?, ?, ?)",
            [(link.src_path, link.dst_target, int(link.dst_resolved), link.kind) for link in links],
        )
        conn.executemany(
            "INSERT INTO tags VALUES (?, ?)",
            [(tag, n.path) for n in notes for tag in n.tags],
        )
        conn.executemany(
            "INSERT INTO notes_fts (path, title, body) VALUES (?, ?, ?)",
            [(n.path, n.title, n.body) for n in notes],
        )
        conn.commit()
    finally:
        conn.close()
    _write_meta(db_path, signature, len(notes), len(links))
    return len(notes), len(links)


def _resolution_from_index(conn: sqlite3.Connection) -> dict[str, str]:
    """stem + alias -> path, rebuilt from the index's notes table (no vault scan)."""
    mapping: dict[str, str] = {}
    for path, aliases_json in conn.execute("SELECT path, aliases_json FROM notes"):
        mapping.setdefault(Path(path).stem, path)
        for alias in json.loads(aliases_json or "[]"):
            mapping.setdefault(alias, path)
    return mapping


def update_note(db_path: Path, vault_dir: Path, rel_path: str) -> None:
    """Incrementally sync one note's rows after it was written or deleted — the
    save-time fast path (no full vault rescan). Outbound links are re-resolved
    against the rest of the index. When a newly created/updated note satisfies
    previously unresolved inbound targets, those broken links are resolved in-place.
    Removing aliases/stems can still leave stale inbound resolution until the next
    full build_index. Body/status/tag edits (the common case) leave aliases
    unchanged, so the incremental result equals a full rebuild.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM notes WHERE path = ?", (rel_path,))
        conn.execute("DELETE FROM links WHERE src_path = ?", (rel_path,))
        conn.execute("DELETE FROM tags WHERE note_path = ?", (rel_path,))
        conn.execute("DELETE FROM notes_fts WHERE path = ?", (rel_path,))
        md_path = vault_dir / rel_path
        if md_path.is_file():
            note = _scan_one(md_path, vault_dir)
            mapping = _resolution_from_index(conn)  # self row already deleted above
            mapping.setdefault(note.stem, note.path)
            for alias in note.aliases:
                mapping.setdefault(alias, note.path)
            conn.execute(
                "INSERT INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    note.path,
                    note.title,
                    note.domain,
                    note.category,
                    note.status,
                    json.dumps(note.aliases, ensure_ascii=False),
                    json.dumps(note.tags, ensure_ascii=False),
                    note.mtime,
                ),
            )
            conn.executemany(
                "INSERT INTO links VALUES (?, ?, ?, ?)",
                list(
                    dict.fromkeys(
                        (
                            note.path,
                            mapping[target] if target in mapping else target,
                            int(target in mapping),
                            "wikilink",
                        )
                        for target in note.targets
                    )
                ),
            )
            conn.executemany(
                "INSERT INTO tags VALUES (?, ?)", [(tag, note.path) for tag in note.tags]
            )
            conn.execute(
                "INSERT INTO notes_fts (path, title, body) VALUES (?, ?, ?)",
                (note.path, note.title, note.body),
            )
            inbound_targets = sorted({note.stem, *note.aliases})
            if inbound_targets:
                placeholders = ",".join("?" for _ in inbound_targets)
                conn.execute(
                    f"UPDATE links SET dst_target = ?, dst_resolved = 1 "
                    f"WHERE dst_resolved = 0 AND dst_target IN ({placeholders})",
                    (note.path, *inbound_targets),
                )
        conn.commit()
    finally:
        conn.close()


def index_is_stale(db_path: Path, vault_dir: Path) -> bool:
    """True if the vault has markdown files the index does not reflect — added,
    removed, or modified (mtime) since the index recorded them. Stat-only, no reads."""
    if not db_path.exists():
        return True
    conn = sqlite3.connect(db_path)
    try:
        indexed = {path: mtime for path, mtime in conn.execute("SELECT path, mtime FROM notes")}
    finally:
        conn.close()
    current = {
        md.relative_to(vault_dir).as_posix(): md.stat().st_mtime for md in vault_dir.rglob("*.md")
    }
    if set(indexed) != set(current):
        return True
    return any(current[path] > indexed[path] + 1e-6 for path in current)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Markdown vault link-graph index.")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="Rebuild the SQLite link-graph index.")
    build_parser.add_argument("--vault", default=DEFAULT_VAULT)
    build_parser.add_argument("--db", default=DEFAULT_DB)

    report_parser = sub.add_parser("report", help="Report broken wikilinks and orphan notes.")
    report_parser.add_argument("--vault", default=DEFAULT_VAULT)

    args = parser.parse_args(argv)

    if args.command == "build":
        n_notes, n_links = build_index(Path(args.vault), Path(args.db))
        print(f"link-graph index built: {n_notes} notes, {n_links} links -> {args.db}")
        return 0

    notes, links = scan_vault(Path(args.vault))
    broken = broken_links(links)
    orphans = orphan_notes(notes, links)
    print(f"notes={len(notes)} links={len(links)} broken={len(broken)} orphans={len(orphans)}")
    for link in broken:
        print(f"  broken: {link.src_path} -> [[{link.dst_target}]]")
    for note in orphans:
        print(f"  orphan: {note.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
