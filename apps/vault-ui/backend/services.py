"""Read-only vault + link-graph services for the vault-ui backend.

Reads the link-graph SQLite index (built by scripts.core.linkgraph) for tree and
backlinks, and reads markdown files live from the vault for note bodies. Wikilink
resolution reuses scripts.core.linkgraph so the UI agrees with the index exactly.
"""

import json
import hashlib
import logging
import math
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from scripts.core.linkgraph import (
    WIKILINK_RE,
    _parse_frontmatter,
    index_is_stale,
    update_note,
)

logger = logging.getLogger("vault_ui.services")

VAULT_DIR = Path(os.getenv("VAULT_UI_VAULT_DIR", "wiki"))
LINKGRAPH_DB = Path(os.getenv("VAULT_UI_LINKGRAPH_DB", ".graph_index/linkgraph.sqlite"))
RELATED_DB = Path(os.getenv("VAULT_UI_RELATED_DB", ".graph_index/related.sqlite"))
COMMUNITIES_DB = Path(os.getenv("VAULT_UI_COMMUNITIES_DB", ".graph_index/communities.sqlite"))
GLOBAL_LAYOUT_ALGORITHM = "community-ring-v1"
LOCAL_LAYOUT_ALGORITHM = "local-radial-v1"


def _connect(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def health(db_path: Path | None = None, vault_dir: Path | None = None) -> dict[str, Any]:
    db_path = db_path or LINKGRAPH_DB
    vault_dir = vault_dir or VAULT_DIR
    conn = _connect(db_path)
    if conn is None:
        return {"status": "ok", "stale": True, "notes": 0, "links": 0}
    try:
        notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    finally:
        conn.close()
    return {
        "status": "ok",
        "stale": index_is_stale(db_path, vault_dir),
        "notes": notes,
        "links": links,
    }


def build_tree(db_path: Path | None = None) -> dict[str, Any]:
    """Nested folder/note tree from the notes table (single query)."""
    db_path = db_path or LINKGRAPH_DB
    conn = _connect(db_path)
    root: dict[str, Any] = {"name": "", "path": "", "folders": {}, "notes": []}
    if conn is None:
        return {"stale": True, "tree": _finalize_tree(root)}
    try:
        rows = conn.execute("SELECT path, title FROM notes ORDER BY path").fetchall()
    finally:
        conn.close()
    for row in rows:
        parts = row["path"].split("/")
        node = root
        for folder in parts[:-1]:
            node = node["folders"].setdefault(
                folder, {"name": folder, "path": "", "folders": {}, "notes": []}
            )
        node["notes"].append({"path": row["path"], "title": row["title"]})
    return {"stale": False, "tree": _finalize_tree(root)}


def _finalize_tree(node: dict[str, Any]) -> dict[str, Any]:
    """Convert the folder dict into a sorted children list."""
    children = [_finalize_tree(child) for _, child in sorted(node["folders"].items())]
    return {
        "name": node["name"],
        "children": children,
        "notes": node["notes"],
    }


def _resolution_map(conn: sqlite3.Connection) -> dict[str, str]:
    """stem + alias -> note path, mirroring scripts.core.linkgraph.scan_vault."""
    mapping: dict[str, str] = {}
    for row in conn.execute("SELECT path, aliases_json FROM notes"):
        stem = Path(row["path"]).stem
        mapping.setdefault(stem, row["path"])
        for alias in json.loads(row["aliases_json"] or "[]"):
            mapping.setdefault(alias, row["path"])
    return mapping


def _safe_vault_path(vault_dir: Path, rel_path: str) -> Path | None:
    """Resolve rel_path under vault_dir, rejecting traversal outside it."""
    base = vault_dir.resolve()
    target = (base / rel_path).resolve()
    if base != target and base not in target.parents:
        return None
    return target


def get_note(
    rel_path: str, vault_dir: Path | None = None, db_path: Path | None = None
) -> dict[str, Any] | None:
    vault_dir = vault_dir or VAULT_DIR
    db_path = db_path or LINKGRAPH_DB
    target = _safe_vault_path(vault_dir, rel_path)
    if target is None or target.suffix != ".md" or not target.is_file():
        return None

    text = target.read_text(encoding="utf-8")
    frontmatter = _parse_frontmatter(text)
    fm_block = text.split("---", 2)
    body = fm_block[2].lstrip("\n") if text.startswith("---") and len(fm_block) >= 3 else text

    aliases = frontmatter.get("aliases") or []
    title = str(frontmatter.get("title") or (aliases[0] if aliases else target.stem))

    link_resolution: dict[str, str | None] = {}
    conn = _connect(db_path)
    if conn is not None:
        try:
            mapping = _resolution_map(conn)
        finally:
            conn.close()
        for raw in WIKILINK_RE.findall(body):
            link_resolution[raw.strip()] = mapping.get(raw.strip())

    return {
        "path": rel_path,
        "title": title,
        "frontmatter": frontmatter,
        "markdown": body,
        "link_resolution": link_resolution,
        "mtime": target.stat().st_mtime,
    }


class ConflictError(Exception):
    """Raised when a save would clobber a note modified since it was loaded."""


def _atomic_write(target: Path, text: str) -> None:
    """Write text to target via a tempfile + os.replace so readers never see a
    half-written file; clean up the tempfile on any failure."""
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _reindex_or_flag_stale(db_path: Path, vault_dir: Path, rel_path: str) -> bool:
    """Incrementally reindex a just-written note. Returns True (index lags the
    file) if there is no index yet or the reindex failed — the markdown write has
    already succeeded, so callers report success while flagging the staleness
    rather than turning a saved file into a 500."""
    if not db_path.exists():
        return True
    try:
        update_note(db_path, vault_dir, rel_path)
        return False
    except Exception:
        logger.exception("reindex failed after writing %s; index is stale", rel_path)
        return True


def save_note(
    rel_path: str,
    body: str,
    vault_dir: Path | None = None,
    db_path: Path | None = None,
    base_mtime: float | None = None,
) -> dict[str, Any] | None:
    """Atomically write a new body for an existing note, preserving its YAML
    frontmatter verbatim, then incrementally reindex it so backlinks/tree/search
    stay consistent. Raises ValueError for traversal / non-.md paths; ConflictError
    if base_mtime is given and the file changed since (blind-overwrite guard);
    returns None if the note does not exist; returns the saved path on success."""
    vault_dir = vault_dir or VAULT_DIR
    db_path = db_path or LINKGRAPH_DB
    target = _safe_vault_path(vault_dir, rel_path)
    if target is None or target.suffix != ".md":
        raise ValueError(f"unsafe or non-markdown path: {rel_path}")
    if not target.is_file():
        return None
    if base_mtime is not None and target.stat().st_mtime > base_mtime + 1e-6:
        raise ConflictError(f"note changed since it was loaded: {rel_path}")

    original = target.read_text(encoding="utf-8")
    # Preserve the original frontmatter block byte-for-byte; only the body changes.
    parts = original.split("---", 2)
    if original.startswith("---") and len(parts) >= 3:
        frontmatter_raw = f"---{parts[1]}---"
        new_text = f"{frontmatter_raw}\n{body}"
    else:
        new_text = body
    if not new_text.endswith("\n"):
        new_text += "\n"

    _atomic_write(target, new_text)
    index_stale = _reindex_or_flag_stale(db_path, vault_dir, rel_path)
    return {"path": rel_path, "ok": True, "index_stale": index_stale}


VALID_STATUSES = frozenset({"unread", "reading", "done", "archived", "irrelevant"})
_STATUS_LINE_RE = re.compile(r"(?m)^status:.*$")


def set_status(
    rel_path: str,
    status: str,
    vault_dir: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    """Surgically set the frontmatter ``status:`` of an existing note (inserting the
    field if absent), preserving every other frontmatter line byte-for-byte to avoid
    YAML re-dump reflow, then incrementally reindex. Raises ValueError for an invalid
    status or unsafe/non-.md path; returns None if the note is missing."""
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    vault_dir = vault_dir or VAULT_DIR
    db_path = db_path or LINKGRAPH_DB
    target = _safe_vault_path(vault_dir, rel_path)
    if target is None or target.suffix != ".md":
        raise ValueError(f"unsafe or non-markdown path: {rel_path}")
    if not target.is_file():
        return None

    text = target.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if not (text.startswith("---") and len(parts) >= 3):
        raise ValueError(f"note has no frontmatter: {rel_path}")
    frontmatter, body = parts[1], parts[2]
    if _STATUS_LINE_RE.search(frontmatter):
        frontmatter = _STATUS_LINE_RE.sub(f"status: {status}", frontmatter, count=1)
    else:
        frontmatter = f"{frontmatter.rstrip(chr(10))}\nstatus: {status}\n"

    _atomic_write(target, f"---{frontmatter}---{body}")
    index_stale = _reindex_or_flag_stale(db_path, vault_dir, rel_path)
    return {"path": rel_path, "status": status, "ok": True, "index_stale": index_stale}


def _slug(title: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip()
    return re.sub(r"[\s_-]+", "_", cleaned)


def create_note(
    title: str,
    folder: str = "general",
    vault_dir: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Create a new note with schema-valid frontmatter (domain/category general,
    status unread) and reindex it. Raises ValueError for an empty/unsafe title,
    unsafe folder, or an existing path."""
    vault_dir = vault_dir or VAULT_DIR
    db_path = db_path or LINKGRAPH_DB
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("title is required")
    stem = _slug(clean_title)
    if not stem:
        raise ValueError(f"title has no filename-safe characters: {title}")
    rel_path = f"{folder.strip('/')}/{stem}.md"
    target = _safe_vault_path(vault_dir, rel_path)
    if target is None or target.suffix != ".md":
        raise ValueError(f"unsafe path: {rel_path}")
    if target.exists():
        raise ValueError(f"note already exists: {rel_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    title_yaml = clean_title.replace("'", "''")
    frontmatter = (
        f"---\naliases: ['{title_yaml}']\ndomain: general\ncategory: general\n"
        f"tags: []\nsources: []\nstatus: unread\n---\n"
    )
    _atomic_write(target, f"{frontmatter}# {clean_title}\n")
    index_stale = _reindex_or_flag_stale(db_path, vault_dir, rel_path)
    return {"path": rel_path, "ok": True, "index_stale": index_stale}


def get_backlinks(rel_path: str, db_path: Path | None = None) -> dict[str, Any]:
    db_path = db_path or LINKGRAPH_DB
    conn = _connect(db_path)
    if conn is None:
        return {"stale": True, "backlinks": []}
    try:
        rows = conn.execute(
            """
            SELECT l.src_path AS src_path, n.title AS src_title
            FROM links l
            LEFT JOIN notes n ON n.path = l.src_path
            WHERE l.dst_target = ? AND l.dst_resolved = 1
            GROUP BY l.src_path
            ORDER BY l.src_path
            """,
            (rel_path,),
        ).fetchall()
    finally:
        conn.close()
    backlinks = [
        {"src_path": row["src_path"], "src_title": row["src_title"] or row["src_path"]}
        for row in rows
    ]
    return {"stale": False, "backlinks": backlinks}


def _fts_query(q: str) -> str:
    """Prefix-MATCH over word tokens; dropping non-word chars keeps FTS5 operators
    and punctuation out of the query so they can't raise a syntax error."""
    return " ".join(f"{token}*" for token in re.findall(r"\w+", q, re.UNICODE))


def search_notes(q: str, limit: int = 30, db_path: Path | None = None) -> dict[str, Any]:
    db_path = db_path or LINKGRAPH_DB
    q = q.strip()
    if not q:
        return {"stale": False, "results": []}
    conn = _connect(db_path)
    if conn is None:
        return {"stale": True, "results": []}
    try:
        rows: list[sqlite3.Row] = []
        match = _fts_query(q)
        if match:
            try:
                rows = conn.execute(
                    "SELECT path, title FROM notes_fts WHERE notes_fts MATCH ?"
                    " AND path NOT LIKE 'meta/%' ORDER BY rank LIMIT ?",
                    (match, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []  # pre-FTS index -> LIKE fallback below
        if not rows:
            pattern = f"%{q.lower()}%"
            rows = conn.execute(
                "SELECT path, title FROM notes WHERE (LOWER(title) LIKE ? OR LOWER(path) LIKE ?)"
                " AND path NOT LIKE 'meta/%' ORDER BY title LIMIT ?",
                (pattern, pattern, limit),
            ).fetchall()
    finally:
        conn.close()
    return {
        "stale": False,
        "results": [{"path": row["path"], "title": row["title"]} for row in rows],
    }


SYSTEM_TAGS = frozenset({"zotero_import"})


def list_tags(db_path: Path | None = None) -> dict[str, Any]:
    """All tags with note counts, most-used first; system markers excluded so the
    facet shows only semantic tags."""
    db_path = db_path or LINKGRAPH_DB
    conn = _connect(db_path)
    if conn is None:
        return {"stale": True, "tags": []}
    try:
        rows = conn.execute(
            "SELECT tag, COUNT(*) AS c FROM tags GROUP BY tag ORDER BY c DESC, tag"
        ).fetchall()
    finally:
        conn.close()
    tags = [{"tag": row["tag"], "count": row["c"]} for row in rows if row["tag"] not in SYSTEM_TAGS]
    return {"stale": False, "tags": tags}


def notes_by_tag(tag: str, db_path: Path | None = None) -> dict[str, Any]:
    """Notes carrying a given tag (path + title), title-sorted."""
    db_path = db_path or LINKGRAPH_DB
    conn = _connect(db_path)
    if conn is None:
        return {"stale": True, "notes": []}
    try:
        rows = conn.execute(
            "SELECT n.path AS path, n.title AS title FROM tags t "
            "JOIN notes n ON n.path = t.note_path WHERE t.tag = ? ORDER BY n.title",
            (tag,),
        ).fetchall()
    finally:
        conn.close()
    return {"stale": False, "notes": [{"path": row["path"], "title": row["title"]} for row in rows]}


MAX_GRAPH_NODES = 240

# Only these folders are graph citizens; everything else (meta, tags, scaffolding)
# is excluded so the semantic web is paper<->concept, not paper<->tag.
GRAPH_FOLDERS = frozenset({"research", "concepts"})


def _folder_of(path: str) -> str:
    return path.split("/")[0] if "/" in path else ""


def _unit_hash(value: str, *, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _deterministic_global_positions(
    paths: set[str],
    communities: dict[str, int],
) -> dict[str, dict[str, float]]:
    if not paths:
        return {}
    community_sizes: dict[int | None, int] = {}
    for path in paths:
        community = communities.get(path)
        community_sizes[community] = community_sizes.get(community, 0) + 1
    ranked = sorted(
        community_sizes,
        key=lambda community: (-community_sizes[community], -1 if community is None else community),
    )
    graph_radius = 24 * math.sqrt(len(paths))
    anchor_by_community: dict[int | None, tuple[float, float]] = {}
    for index, community in enumerate(ranked):
        angle = (index * math.pi * (3 - math.sqrt(5))) % (2 * math.pi)
        anchor_by_community[community] = (
            math.cos(angle) * graph_radius,
            math.sin(angle) * graph_radius,
        )

    positions: dict[str, dict[str, float]] = {}
    for path in sorted(paths):
        community = communities.get(path)
        anchor_x, anchor_y = anchor_by_community[community]
        spread = 8 + 5 * math.sqrt(community_sizes[community])
        angle = _unit_hash(path, salt="angle") * 2 * math.pi
        radius = spread * math.sqrt(_unit_hash(path, salt="radius"))
        positions[path] = {
            "x": round(anchor_x + math.cos(angle) * radius, 3),
            "y": round(anchor_y + math.sin(angle) * radius, 3),
        }
    return positions


def _deterministic_local_positions(
    root: str,
    distances: dict[str, int],
) -> dict[str, dict[str, float]]:
    positions: dict[str, dict[str, float]] = {}
    for path, distance in distances.items():
        if path == root:
            positions[path] = {"x": 0, "y": 0}
            continue
        ring = max(1, distance)
        angle = _unit_hash(path, salt=f"local-angle-{ring}") * 2 * math.pi
        radius = 82 * ring + 18 * _unit_hash(path, salt="local-radius")
        positions[path] = {
            "x": round(math.cos(angle) * radius, 3),
            "y": round(math.sin(angle) * radius, 3),
        }
    return positions


# The relatedness index keeps top-8 per paper (good for the Related panel), but a
# graph drawn with all 8 is a uniform hairball — measured modularity 0.55 at top-8
# vs 0.71 at top-2 with the same 5% isolation, so the graph caps each node to its
# strongest few similar edges to reveal clusters instead of a blob.
GRAPH_SIMILAR_TOP_K = 2


def _node_communities(db_path: Path) -> dict[str, int]:
    """path -> Leiden community id (precomputed); empty if the index is absent."""
    conn = _connect(db_path)
    if conn is None:
        return {}
    try:
        return {
            row["path"]: row["community"]
            for row in conn.execute("SELECT path, community FROM community")
        }
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def _similar_pairs(
    related_db: Path, in_set: set[str], top_k: int = GRAPH_SIMILAR_TOP_K
) -> set[tuple[str, str]]:
    """Undirected similar-paper pairs from the relatedness index, both endpoints
    in-set, keeping only each source's top_k strongest edges."""
    conn = _connect(related_db)
    if conn is None:
        return set()
    try:
        rows = conn.execute("SELECT src_path, dst_path FROM related ORDER BY score DESC").fetchall()
    finally:
        conn.close()
    kept_per_src: dict[str, int] = {}
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        src, dst = row["src_path"], row["dst_path"]
        if src not in in_set or dst not in in_set or src == dst:
            continue
        if kept_per_src.get(src, 0) >= top_k:
            continue
        kept_per_src[src] = kept_per_src.get(src, 0) + 1
        pairs.add((src, dst) if src < dst else (dst, src))
    return pairs


def get_graph(
    rel_path: str,
    depth: int = 1,
    include_similar: bool = False,
    db_path: Path | None = None,
    related_db: Path | None = None,
) -> dict[str, Any]:
    """Local semantic graph around rel_path. Nodes are only research/ + concepts/
    notes; explicit edges are wikilinks where BOTH endpoints are in that set (drops
    meta/tag links). When include_similar, adds relatedness-index edges between
    in-set papers. BFS over both edge kinds up to `depth` hops, capped at
    MAX_GRAPH_NODES. Each emitted link carries kind: link | similar."""
    db_path = db_path or LINKGRAPH_DB
    related_db = related_db or RELATED_DB
    depth = max(1, min(depth, 3))
    conn = _connect(db_path)
    if conn is None:
        return {"stale": True, "nodes": [], "links": []}
    try:
        titles = {
            row["path"]: row["title"]
            for row in conn.execute("SELECT path, title FROM notes")
            if _folder_of(row["path"]) in GRAPH_FOLDERS
        }
        adjacency: dict[str, set[str]] = {}
        link_pairs: set[tuple[str, str]] = set()
        for row in conn.execute("SELECT src_path, dst_target FROM links WHERE dst_resolved = 1"):
            src, dst = row["src_path"], row["dst_target"]
            if src not in titles or dst not in titles or src == dst:
                continue
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)
            link_pairs.add((src, dst) if src < dst else (dst, src))
    finally:
        conn.close()

    if rel_path not in titles:
        return {"stale": False, "nodes": [], "links": []}

    similar_pairs: set[tuple[str, str]] = set()
    if include_similar:
        similar_pairs = _similar_pairs(related_db, set(titles))
        for src, dst in similar_pairs:
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)

    visited = {rel_path}
    distances = {rel_path: 0}
    frontier = {rel_path}
    for current_depth in range(depth):
        nxt: set[str] = set()
        for node in frontier:
            for neighbor in adjacency.get(node, ()):  # noqa: SIM118
                if neighbor not in visited and len(visited) < MAX_GRAPH_NODES:
                    visited.add(neighbor)
                    distances[neighbor] = current_depth + 1
                    nxt.add(neighbor)
        frontier = nxt
        if not frontier:
            break

    communities = _node_communities(COMMUNITIES_DB)
    positions = _deterministic_local_positions(rel_path, distances)
    nodes = [
        {
            "id": path,
            "title": titles.get(path, path),
            "folder": _folder_of(path),
            "current": path == rel_path,
            "community": communities.get(path),
            **positions.get(path, {}),
        }
        for path in sorted(visited)
    ]
    # Explicit links win over similar for the same pair, so no double-drawn edge.
    links = [
        {"source": src, "target": dst, "kind": "link"}
        for src, dst in link_pairs
        if src in visited and dst in visited
    ]
    links.extend(
        {"source": src, "target": dst, "kind": "similar"}
        for src, dst in similar_pairs
        if src in visited and dst in visited and (src, dst) not in link_pairs
    )
    return {
        "stale": False,
        "nodes": nodes,
        "links": links,
        "layout": {"algorithm": LOCAL_LAYOUT_ALGORITHM, "deterministic": True},
    }


MAX_GLOBAL_NODES = 3000


def get_global_graph(
    current: str | None = None,
    include_similar: bool = False,
    db_path: Path | None = None,
    related_db: Path | None = None,
) -> dict[str, Any]:
    """The whole semantic web: only research/ + concepts/ notes as nodes, with
    explicit wikilink edges between in-set notes (degree for sizing). When
    include_similar, also adds relatedness-index edges between in-set papers.
    Capped at MAX_GLOBAL_NODES; if exceeded, keep highest-degree nodes and their
    edges. Each link carries kind: link | similar."""
    db_path = db_path or LINKGRAPH_DB
    related_db = related_db or RELATED_DB
    conn = _connect(db_path)
    if conn is None:
        return {"stale": True, "nodes": [], "links": []}
    try:
        titles = {
            row["path"]: row["title"]
            for row in conn.execute("SELECT path, title FROM notes")
            if _folder_of(row["path"]) in GRAPH_FOLDERS
        }
        link_pairs: set[tuple[str, str]] = set()
        for row in conn.execute("SELECT src_path, dst_target FROM links WHERE dst_resolved = 1"):
            src, dst = row["src_path"], row["dst_target"]
            if src in titles and dst in titles and src != dst:
                link_pairs.add((src, dst) if src < dst else (dst, src))
    finally:
        conn.close()

    similar_pairs: set[tuple[str, str]] = set()
    if include_similar:
        similar_pairs = _similar_pairs(related_db, set(titles))

    # Degree from emitted edges (link + similar) so sizing matches what is drawn.
    degree: dict[str, int] = {}
    for src, dst in link_pairs | similar_pairs:
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1

    keep = set(titles)
    if len(titles) > MAX_GLOBAL_NODES:
        ranked = sorted(titles, key=lambda p: degree.get(p, 0), reverse=True)
        keep = set(ranked[:MAX_GLOBAL_NODES])

    communities = _node_communities(COMMUNITIES_DB)
    positions = _deterministic_global_positions(keep, communities)
    nodes = [
        {
            "id": path,
            "title": titles[path],
            "folder": _folder_of(path),
            "degree": degree.get(path, 0),
            "current": path == current,
            "community": communities.get(path),
            **positions[path],
        }
        for path in keep
    ]
    links = [
        {"source": src, "target": dst, "kind": "link"}
        for src, dst in link_pairs
        if src in keep and dst in keep
    ]
    links.extend(
        {"source": src, "target": dst, "kind": "similar"}
        for src, dst in similar_pairs
        if src in keep and dst in keep and (src, dst) not in link_pairs
    )
    return {
        "stale": False,
        "nodes": nodes,
        "links": links,
        "truncated": len(titles) > MAX_GLOBAL_NODES,
        "layout": {"algorithm": GLOBAL_LAYOUT_ALGORITHM, "deterministic": True},
    }


def get_concept_audit() -> dict[str, Any]:
    from scripts.search.concept_graph_audit import build_audit_report

    return build_audit_report(
        vault_dir=VAULT_DIR,
        linkgraph_db=LINKGRAPH_DB,
        communities_db=COMMUNITIES_DB,
        related_db=RELATED_DB,
    )


def get_related(
    rel_path: str,
    limit: int = 8,
    related_db: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Semantically related papers for rel_path, from the relatedness index."""
    related_db = related_db or RELATED_DB
    db_path = db_path or LINKGRAPH_DB
    conn = _connect(related_db)
    if conn is None:
        return {"stale": True, "related": []}
    try:
        rows = conn.execute(
            "SELECT dst_path, score FROM related WHERE src_path = ? ORDER BY score DESC LIMIT ?",
            (rel_path, limit),
        ).fetchall()
    finally:
        conn.close()

    titles: dict[str, str] = {}
    note_conn = _connect(db_path)
    if note_conn is not None:
        try:
            titles = {
                r["path"]: r["title"] for r in note_conn.execute("SELECT path, title FROM notes")
            }
        finally:
            note_conn.close()

    related = [
        {
            "path": row["dst_path"],
            "title": titles.get(row["dst_path"], row["dst_path"]),
            "score": row["score"],
        }
        for row in rows
    ]
    return {"stale": False, "related": related}
