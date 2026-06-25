import sqlite3
import time
import json

from scripts.core import linkgraph


def _write(vault, rel, frontmatter_aliases, body):
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    alias_line = f"aliases: {frontmatter_aliases}\n" if frontmatter_aliases is not None else ""
    path.write_text(
        f"---\n{alias_line}domain: research\ncategory: materials\ntags: [t1]\n"
        f"sources: []\n---\n{body}\n",
        encoding="utf-8",
    )


def test_scan_resolves_links_and_backlinks(tmp_path):
    _write(tmp_path, "research/a.md", None, "links to [[b]] and [[Missing Note]].")
    _write(tmp_path, "research/b.md", ["B Alias"], "links back to [[a]].")
    _write(tmp_path, "research/c.md", None, "refers to [[B Alias]] by alias.")

    notes, links = linkgraph.scan_vault(tmp_path)

    assert {n.path for n in notes} == {"research/a.md", "research/b.md", "research/c.md"}

    resolved = {(link.src_path, link.dst_target) for link in links if link.dst_resolved}
    assert ("research/a.md", "research/b.md") in resolved
    assert ("research/b.md", "research/a.md") in resolved
    # alias resolution
    assert ("research/c.md", "research/b.md") in resolved

    broken = linkgraph.broken_links(links)
    assert [link.dst_target for link in broken] == ["Missing Note"]


def test_orphan_detection(tmp_path):
    _write(tmp_path, "research/a.md", None, "see [[b]].")
    _write(tmp_path, "research/b.md", None, "back to [[a]].")
    _write(tmp_path, "research/lonely.md", None, "no links here.")

    notes, links = linkgraph.scan_vault(tmp_path)
    orphans = {n.path for n in linkgraph.orphan_notes(notes, links)}
    assert orphans == {"research/lonely.md"}


def test_build_index_writes_sqlite(tmp_path):
    _write(tmp_path, "research/a.md", None, "see [[b]] and [[ghost]].")
    _write(tmp_path, "research/b.md", None, "back to [[a]].")
    db = tmp_path / "out" / "linkgraph.sqlite"

    n_notes, n_links = linkgraph.build_index(tmp_path, db)
    assert (n_notes, n_links) == (2, 3)

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM links WHERE dst_resolved=0").fetchone()[0] == 1
        # backlinks of b = reverse links resolved to b's path
        backlinks = conn.execute(
            "SELECT src_path FROM links WHERE dst_target=? AND dst_resolved=1",
            ("research/b.md",),
        ).fetchall()
        assert backlinks == [("research/a.md",)]
        assert conn.execute("SELECT COUNT(*) FROM tags WHERE tag='t1'").fetchone()[0] == 2
    finally:
        conn.close()


def test_build_index_deduplicates_repeated_wikilink_edges(tmp_path):
    _write(tmp_path, "research/a.md", None, "see [[b]], [[b]], and [[B Alias]].")
    _write(tmp_path, "research/b.md", ["B Alias"], "target.")
    db = tmp_path / "out" / "linkgraph.sqlite"

    n_notes, n_links = linkgraph.build_index(tmp_path, db)

    assert (n_notes, n_links) == (2, 1)
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT src_path, dst_target, dst_resolved FROM links ORDER BY src_path, dst_target"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("research/a.md", "research/b.md", 1)]


def test_build_index_meta_records_index_version(tmp_path):
    _write(tmp_path, "research/a.md", None, "see [[b]].")
    _write(tmp_path, "research/b.md", None, "target.")
    db = tmp_path / "out" / "linkgraph.sqlite"

    linkgraph.build_index(tmp_path, db)

    meta = json.loads((tmp_path / "out" / "linkgraph.sqlite.meta.json").read_text())
    assert meta["version"] == linkgraph.INDEX_VERSION


def test_build_index_skips_current_output_without_rewriting(tmp_path):
    _write(tmp_path, "research/a.md", None, "see [[b]].")
    _write(tmp_path, "research/b.md", None, "back to [[a]].")
    db = tmp_path / "lg.sqlite"
    first = linkgraph.build_index(tmp_path, db)
    mtime_ns = db.stat().st_mtime_ns
    time.sleep(0.01)

    second = linkgraph.build_index(tmp_path, db)

    assert second == first
    assert db.stat().st_mtime_ns == mtime_ns


def _write_status(vault, rel, status, body):
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ndomain: research\ncategory: materials\ntags: [t1]\nsources: []\n"
        f"status: {status}\n---\n{body}\n",
        encoding="utf-8",
    )


def test_update_note_reflects_edited_status_incrementally(tmp_path):
    _write_status(tmp_path, "research/a.md", "unread", "see [[b]].")
    _write(tmp_path, "research/b.md", None, "back to [[a]].")
    db = tmp_path / "lg.sqlite"
    linkgraph.build_index(tmp_path, db)

    _write_status(tmp_path, "research/a.md", "done", "see [[b]].")
    linkgraph.update_note(db, tmp_path, "research/a.md")

    conn = sqlite3.connect(db)
    try:
        assert (
            conn.execute("SELECT status FROM notes WHERE path=?", ("research/a.md",)).fetchone()[0]
            == "done"
        )
        # only a changed; b's row is untouched
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 2
    finally:
        conn.close()


def test_update_note_re_resolves_outbound_links(tmp_path):
    _write(tmp_path, "research/a.md", None, "see [[b]].")
    _write(tmp_path, "research/b.md", None, "x")
    _write(tmp_path, "research/c.md", None, "x")
    db = tmp_path / "lg.sqlite"
    linkgraph.build_index(tmp_path, db)

    _write(tmp_path, "research/a.md", None, "now see [[c]].")
    linkgraph.update_note(db, tmp_path, "research/a.md")

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT dst_target, dst_resolved FROM links WHERE src_path=?", ("research/a.md",)
        ).fetchall()
        assert rows == [("research/c.md", 1)]
    finally:
        conn.close()


def test_update_note_removes_deleted_note(tmp_path):
    _write(tmp_path, "research/a.md", None, "see [[b]].")
    _write(tmp_path, "research/b.md", None, "back to [[a]].")
    db = tmp_path / "lg.sqlite"
    linkgraph.build_index(tmp_path, db)

    (tmp_path / "research/a.md").unlink()
    linkgraph.update_note(db, tmp_path, "research/a.md")

    conn = sqlite3.connect(db)
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM notes WHERE path=?", ("research/a.md",)).fetchone()[
                0
            ]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM links WHERE src_path=?", ("research/a.md",)
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM tags WHERE note_path=?", ("research/a.md",)
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_index_is_stale_detects_added_file(tmp_path):
    _write(tmp_path, "research/a.md", None, "x")
    db = tmp_path / "lg.sqlite"
    linkgraph.build_index(tmp_path, db)
    assert linkgraph.index_is_stale(db, tmp_path) is False

    _write(tmp_path, "research/b.md", None, "y")
    assert linkgraph.index_is_stale(db, tmp_path) is True


def test_index_is_stale_detects_edited_file(tmp_path):
    import os

    _write(tmp_path, "research/a.md", None, "x")
    db = tmp_path / "lg.sqlite"
    linkgraph.build_index(tmp_path, db)

    target = tmp_path / "research/a.md"
    stat = target.stat()
    os.utime(target, (stat.st_atime, stat.st_mtime + 10))
    assert linkgraph.index_is_stale(db, tmp_path) is True


def test_update_note_keeps_index_fresh(tmp_path):
    _write_status(tmp_path, "research/a.md", "unread", "x")
    db = tmp_path / "lg.sqlite"
    linkgraph.build_index(tmp_path, db)

    _write_status(tmp_path, "research/a.md", "done", "x")
    linkgraph.update_note(db, tmp_path, "research/a.md")
    assert linkgraph.index_is_stale(db, tmp_path) is False


def _fts_paths(db, term):
    conn = sqlite3.connect(db)
    try:
        return [
            row[0]
            for row in conn.execute("SELECT path FROM notes_fts WHERE notes_fts MATCH ?", (term,))
        ]
    finally:
        conn.close()


def test_build_index_indexes_body_for_fts(tmp_path):
    _write(tmp_path, "research/a.md", None, "Dielectric elastomer actuators flex under field.")
    _write(tmp_path, "research/b.md", None, "Unrelated content here.")
    db = tmp_path / "lg.sqlite"
    linkgraph.build_index(tmp_path, db)
    # a body word finds only a; the path token finds both.
    assert _fts_paths(db, "elastomer") == ["research/a.md"]
    assert set(_fts_paths(db, "research")) == {"research/a.md", "research/b.md"}


def test_update_note_reindexes_body_in_fts(tmp_path):
    _write(tmp_path, "research/a.md", None, "mentions piezoelectric only.")
    db = tmp_path / "lg.sqlite"
    linkgraph.build_index(tmp_path, db)

    _write(tmp_path, "research/a.md", None, "now mentions magnetostriction instead.")
    linkgraph.update_note(db, tmp_path, "research/a.md")
    assert _fts_paths(db, "piezoelectric") == []  # old body word gone
    assert _fts_paths(db, "magnetostriction") == ["research/a.md"]  # new body word indexed
