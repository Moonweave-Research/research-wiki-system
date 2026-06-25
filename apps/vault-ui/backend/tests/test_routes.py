import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import services
from backend.main import app, mount_spa
from scripts.core.linkgraph import build_index


def test_mount_spa_serves_index_when_dir_set(tmp_path):
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>VAULT-SPA-ROOT", encoding="utf-8")
    test_app = FastAPI()
    assert mount_spa(test_app, str(static)) is True
    with TestClient(test_app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "VAULT-SPA-ROOT" in resp.text


def test_mount_spa_serves_index_with_revalidation_header(tmp_path):
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>VAULT-SPA-ROOT", encoding="utf-8")
    test_app = FastAPI()
    assert mount_spa(test_app, str(static)) is True

    with TestClient(test_app) as client:
        resp = client.get("/")

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-cache"


def test_mount_spa_closes_websocket_without_staticfiles_assertion(tmp_path):
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>VAULT-SPA-ROOT", encoding="utf-8")
    test_app = FastAPI()
    assert mount_spa(test_app, str(static)) is True

    with TestClient(test_app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/"):
                pass


def test_mount_spa_noop_without_dir():
    test_app = FastAPI()
    assert mount_spa(test_app, None) is False
    assert mount_spa(test_app, "/no/such/dir") is False


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A tiny vault + built link-graph index, wired into services via env paths."""
    vault_dir = tmp_path / "wiki"
    (vault_dir / "research").mkdir(parents=True)
    (vault_dir / "research" / "a.md").write_text(
        "---\naliases: [Note A]\ndomain: research\ncategory: materials\ntags: [t1]\n"
        "sources: []\n---\nSee [[b]] and [[ghost]]. Again [[b]] for dedup.\n",
        encoding="utf-8",
    )
    (vault_dir / "research" / "b.md").write_text(
        "---\ndomain: research\ncategory: materials\ntags: []\nsources: []\n---\nBack to [[a]].\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "linkgraph.sqlite"
    build_index(vault_dir, db_path)

    # minimal relatedness index: a -> b
    related_db = tmp_path / "related.sqlite"
    rconn = sqlite3.connect(related_db)
    rconn.execute("CREATE TABLE related (src_path TEXT, dst_path TEXT, score REAL)")
    rconn.execute("INSERT INTO related VALUES ('research/a.md', 'research/b.md', 0.97)")
    rconn.commit()
    rconn.close()

    monkeypatch.setattr(services, "VAULT_DIR", vault_dir)
    monkeypatch.setattr(services, "LINKGRAPH_DB", db_path)
    monkeypatch.setattr(services, "RELATED_DB", related_db)
    return vault_dir


def test_health(vault):
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["stale"] is False
    assert body["notes"] == 2


def test_tree_shape(vault):
    with TestClient(app) as client:
        body = client.get("/api/tree").json()
    assert body["stale"] is False
    research = next(c for c in body["tree"]["children"] if c["name"] == "research")
    note_paths = {n["path"] for n in research["notes"]}
    assert note_paths == {"research/a.md", "research/b.md"}


def test_note_content_and_resolution(vault):
    with TestClient(app) as client:
        body = client.get("/api/note", params={"path": "research/a.md"}).json()
    assert body["title"] == "Note A"
    assert "See [[b]]" in body["markdown"]
    assert "domain" in body["frontmatter"]
    # resolution: [[b]] -> path; [[ghost]] -> None (broken)
    assert body["link_resolution"]["b"] == "research/b.md"
    assert body["link_resolution"]["ghost"] is None


def test_backlinks(vault):
    with TestClient(app) as client:
        body = client.get("/api/backlinks", params={"path": "research/b.md"}).json()
    assert body["stale"] is False
    assert body["backlinks"] == [{"src_path": "research/a.md", "src_title": "Note A"}]


def test_note_path_traversal_rejected(vault):
    with TestClient(app) as client:
        resp = client.get("/api/note", params={"path": "../../etc/passwd"})
    assert resp.status_code == 404


def test_note_missing_is_404(vault):
    with TestClient(app) as client:
        resp = client.get("/api/note", params={"path": "research/nope.md"})
    assert resp.status_code == 404


def test_save_updates_body_and_preserves_frontmatter(vault):
    with TestClient(app) as client:
        before = client.get("/api/note", params={"path": "research/a.md"}).json()
        resp = client.post("/api/note", json={"path": "research/a.md", "body": "Rewritten body.\n"})
        assert resp.status_code == 200
        after = client.get("/api/note", params={"path": "research/a.md"}).json()
    assert after["markdown"] == "Rewritten body.\n"
    # Frontmatter survives the round-trip untouched.
    assert after["frontmatter"] == before["frontmatter"]
    # And it is byte-present on disk, not re-serialized (catches a yaml.dump regression).
    raw = (vault / "research" / "a.md").read_text(encoding="utf-8")
    assert "aliases: [Note A]" in raw
    assert "domain: research" in raw
    assert raw.endswith("Rewritten body.\n")


def test_save_reindexes_so_backlinks_update(vault):
    with TestClient(app) as client:
        # a links to b; rewriting a to drop that link must update the index on save.
        client.post("/api/note", json={"path": "research/a.md", "body": "No more links.\n"})
        body = client.get("/api/backlinks", params={"path": "research/b.md"}).json()
    assert body["backlinks"] == []


def test_health_stale_after_out_of_band_edit(vault):
    with TestClient(app) as client:
        assert client.get("/api/health").json()["stale"] is False
        # a file appears in the vault without going through the API -> index is stale.
        (vault / "research" / "c.md").write_text(
            "---\ndomain: research\ncategory: materials\ntags: []\nsources: []\n---\nnew.\n",
            encoding="utf-8",
        )
        assert client.get("/api/health").json()["stale"] is True


def test_health_fresh_after_api_save(vault):
    with TestClient(app) as client:
        client.post("/api/note", json={"path": "research/a.md", "body": "edited.\n"})
        assert client.get("/api/health").json()["stale"] is False


def test_get_note_returns_mtime(vault):
    with TestClient(app) as client:
        body = client.get("/api/note", params={"path": "research/a.md"}).json()
    assert isinstance(body["mtime"], (int, float))
    assert body["mtime"] > 0


def test_save_conflict_when_file_changed_externally(vault):
    import os

    with TestClient(app) as client:
        before = client.get("/api/note", params={"path": "research/a.md"}).json()
        target = vault / "research" / "a.md"
        stat = target.stat()
        os.utime(target, (stat.st_atime, stat.st_mtime + 10))  # external edit bumps mtime
        resp = client.post(
            "/api/note",
            json={"path": "research/a.md", "body": "clobber\n", "base_mtime": before["mtime"]},
        )
    assert resp.status_code == 409
    raw = (vault / "research" / "a.md").read_text(encoding="utf-8")
    assert "clobber" not in raw  # the external change was not blindly overwritten


def test_save_succeeds_with_matching_base_mtime(vault):
    with TestClient(app) as client:
        before = client.get("/api/note", params={"path": "research/a.md"}).json()
        resp = client.post(
            "/api/note",
            json={"path": "research/a.md", "body": "ok body\n", "base_mtime": before["mtime"]},
        )
    assert resp.status_code == 200


def test_save_path_traversal_rejected(vault):
    with TestClient(app) as client:
        resp = client.post("/api/note", json={"path": "../../etc/passwd", "body": "x"})
    assert resp.status_code == 400


def test_save_non_markdown_rejected(vault):
    with TestClient(app) as client:
        resp = client.post("/api/note", json={"path": "research/a.txt", "body": "x"})
    assert resp.status_code == 400


def test_save_missing_note_is_404(vault):
    with TestClient(app) as client:
        resp = client.post("/api/note", json={"path": "research/nope.md", "body": "x"})
    assert resp.status_code == 404


def test_set_status_inserts_then_updates_without_duplicating(vault):
    with TestClient(app) as client:
        # a.md has no status field -> first call inserts, second updates in place.
        client.post("/api/note/status", json={"path": "research/a.md", "status": "reading"})
        client.post("/api/note/status", json={"path": "research/a.md", "status": "done"})
        after = client.get("/api/note", params={"path": "research/a.md"}).json()
    assert after["frontmatter"]["status"] == "done"
    raw = (vault / "research" / "a.md").read_text(encoding="utf-8")
    assert raw.count("status:") == 1  # no duplicate status lines
    assert "aliases: [Note A]" in raw  # surgical edit preserves the rest of the frontmatter


def test_set_status_invalid_value_rejected(vault):
    with TestClient(app) as client:
        resp = client.post("/api/note/status", json={"path": "research/a.md", "status": "bogus"})
    assert resp.status_code == 400


def test_set_status_reindexes_so_index_stays_fresh(vault):
    with TestClient(app) as client:
        client.post("/api/note/status", json={"path": "research/a.md", "status": "done"})
        assert client.get("/api/health").json()["stale"] is False


def test_set_status_missing_note_404(vault):
    with TestClient(app) as client:
        resp = client.post("/api/note/status", json={"path": "research/nope.md", "status": "done"})
    assert resp.status_code == 404


def test_set_status_path_traversal_rejected(vault):
    with TestClient(app) as client:
        resp = client.post("/api/note/status", json={"path": "../../etc/passwd", "status": "done"})
    assert resp.status_code == 400


def test_create_note_writes_valid_frontmatter_and_indexes(vault):
    with TestClient(app) as client:
        resp = client.post("/api/note/create", json={"title": "My New Idea"})
        assert resp.status_code == 200
        path = resp.json()["path"]
        results = client.get("/api/search", params={"q": "my new idea"}).json()["results"]
        assert any(r["path"] == path for r in results)  # reindexed, findable immediately
        note = client.get("/api/note", params={"path": path}).json()
    assert note["title"] == "My New Idea"
    assert note["frontmatter"]["status"] == "unread"
    assert note["frontmatter"]["domain"] == "general"


def test_create_note_resolves_existing_broken_wikilink(vault):
    with TestClient(app) as client:
        before = client.get("/api/note", params={"path": "research/a.md"}).json()
        assert before["link_resolution"]["ghost"] is None

        resp = client.post("/api/note/create", json={"title": "ghost"})
        assert resp.status_code == 200
        path = resp.json()["path"]

        backlinks = client.get("/api/backlinks", params={"path": path}).json()["backlinks"]
        after = client.get("/api/note", params={"path": "research/a.md"}).json()

    assert {"src_path": "research/a.md", "src_title": "Note A"} in backlinks
    assert after["link_resolution"]["ghost"] == path


def test_create_note_duplicate_rejected(vault):
    with TestClient(app) as client:
        client.post("/api/note/create", json={"title": "Dup Note"})
        resp = client.post("/api/note/create", json={"title": "Dup Note"})
    assert resp.status_code == 400


def test_create_note_empty_title_rejected(vault):
    with TestClient(app) as client:
        resp = client.post("/api/note/create", json={"title": "   "})
    assert resp.status_code == 400


def test_create_note_folder_traversal_rejected(vault):
    with TestClient(app) as client:
        resp = client.post("/api/note/create", json={"title": "X Note", "folder": "../../etc"})
    assert resp.status_code == 400


def test_search_by_title(vault):
    with TestClient(app) as client:
        body = client.get("/api/search", params={"q": "note"}).json()
    assert body["stale"] is False
    paths = {r["path"] for r in body["results"]}
    assert "research/a.md" in paths


def test_search_by_folder_path(vault):
    with TestClient(app) as client:
        body = client.get("/api/search", params={"q": "research"}).json()
    paths = {r["path"] for r in body["results"]}
    assert {"research/a.md", "research/b.md"} == paths


def test_search_empty_q(vault):
    with TestClient(app) as client:
        body = client.get("/api/search", params={"q": ""}).json()
    assert body["results"] == []


def test_search_no_match(vault):
    with TestClient(app) as client:
        body = client.get("/api/search", params={"q": "zzznomatch"}).json()
    assert body["results"] == []


def test_search_excludes_meta_scaffolding(tmp_path, monkeypatch):
    vault_dir = tmp_path / "wiki"
    (vault_dir / "research").mkdir(parents=True)
    (vault_dir / "meta" / "Tags").mkdir(parents=True)
    (vault_dir / "research" / "r.md").write_text(
        "---\ndomain: research\ncategory: materials\ntags: []\nsources: []\n---\ngraphene electrode\n",
        encoding="utf-8",
    )
    (vault_dir / "meta" / "Tags" / "Graphene.md").write_text(
        "---\ndomain: general\ncategory: concepts\ntags: []\nsources: []\n---\ngraphene tag page\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "lg.sqlite"
    build_index(vault_dir, db_path)
    monkeypatch.setattr(services, "VAULT_DIR", vault_dir)
    monkeypatch.setattr(services, "LINKGRAPH_DB", db_path)
    with TestClient(app) as client:
        paths = {
            r["path"] for r in client.get("/api/search", params={"q": "graphene"}).json()["results"]
        }
    assert "research/r.md" in paths
    assert "meta/Tags/Graphene.md" not in paths  # tag scaffolding stays out of search


def test_search_matches_body_content(vault):
    # "ghost" appears only in a.md's BODY, not its title/path -> full-text search finds it.
    with TestClient(app) as client:
        body = client.get("/api/search", params={"q": "ghost"}).json()
    paths = {r["path"] for r in body["results"]}
    assert "research/a.md" in paths


def test_search_special_chars_dont_crash(vault):
    # punctuation/FTS operators must be sanitized, not passed raw into MATCH.
    with TestClient(app) as client:
        resp = client.get("/api/search", params={"q": 'ghost "('})
    assert resp.status_code == 200


def test_list_tags_returns_counts(vault):
    with TestClient(app) as client:
        body = client.get("/api/tags").json()
    tags = {row["tag"]: row["count"] for row in body["tags"]}
    assert tags.get("t1") == 1  # only a.md carries t1


def test_list_tags_excludes_system_tag(tmp_path, monkeypatch):
    vault_dir = tmp_path / "wiki"
    (vault_dir / "research").mkdir(parents=True)
    (vault_dir / "research" / "x.md").write_text(
        "---\ndomain: research\ncategory: materials\ntags: [Graphene, zotero_import]\n"
        "sources: []\n---\nbody\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "lg.sqlite"
    build_index(vault_dir, db_path)
    monkeypatch.setattr(services, "VAULT_DIR", vault_dir)
    monkeypatch.setattr(services, "LINKGRAPH_DB", db_path)
    with TestClient(app) as client:
        tags = {row["tag"] for row in client.get("/api/tags").json()["tags"]}
    assert "Graphene" in tags
    assert "zotero_import" not in tags  # import marker is not a semantic facet


def test_notes_by_tag_returns_notes(vault):
    with TestClient(app) as client:
        body = client.get("/api/notes-by-tag", params={"tag": "t1"}).json()
    paths = {row["path"] for row in body["notes"]}
    assert paths == {"research/a.md"}


def test_similar_pairs_caps_per_source(tmp_path):
    related = tmp_path / "related.sqlite"
    conn = sqlite3.connect(related)
    conn.execute("CREATE TABLE related (src_path TEXT, dst_path TEXT, score REAL)")
    conn.executemany(
        "INSERT INTO related VALUES (?, ?, ?)",
        [
            ("research/a.md", "research/b.md", 0.9),
            ("research/a.md", "research/c.md", 0.8),
            ("research/a.md", "research/d.md", 0.7),
        ],
    )
    conn.commit()
    conn.close()
    in_set = {"research/a.md", "research/b.md", "research/c.md", "research/d.md"}
    pairs = services._similar_pairs(related, in_set, top_k=2)
    assert ("research/a.md", "research/b.md") in pairs  # top-2 by score kept
    assert ("research/a.md", "research/c.md") in pairs
    assert ("research/a.md", "research/d.md") not in pairs  # weakest dropped


def test_graph_local_neighborhood(vault):
    with TestClient(app) as client:
        body = client.get("/api/graph", params={"path": "research/a.md", "depth": 1}).json()
    ids = {n["id"] for n in body["nodes"]}
    assert ids == {"research/a.md", "research/b.md"}  # a<->b resolved; ghost is broken
    assert any(n["current"] for n in body["nodes"] if n["id"] == "research/a.md")
    edge = {frozenset((link["source"], link["target"])) for link in body["links"]}
    assert frozenset(("research/a.md", "research/b.md")) in edge


def test_graph_local_includes_deterministic_layout_coordinates(vault):
    with TestClient(app) as client:
        first = client.get("/api/graph", params={"path": "research/a.md", "depth": 1}).json()
        second = client.get("/api/graph", params={"path": "research/a.md", "depth": 1}).json()

    first_coords = {node["id"]: (node["x"], node["y"]) for node in first["nodes"]}
    second_coords = {node["id"]: (node["x"], node["y"]) for node in second["nodes"]}

    assert first["layout"] == {"algorithm": "local-radial-v1", "deterministic": True}
    assert first_coords == second_coords
    assert first_coords["research/a.md"] == (0, 0)
    assert all(
        isinstance(value, int | float) for coords in first_coords.values() for value in coords
    )


def test_related(vault):
    with TestClient(app) as client:
        body = client.get("/api/related", params={"path": "research/a.md"}).json()
    assert body["stale"] is False
    assert len(body["related"]) == 1
    top = body["related"][0]
    assert top["path"] == "research/b.md"
    assert top["score"] == 0.97
    assert top["title"]  # resolved from the link-graph notes table


def test_graph_global(vault):
    with TestClient(app) as client:
        body = client.get("/api/graph/global", params={"current": "research/b.md"}).json()
    ids = {n["id"] for n in body["nodes"]}
    assert ids == {"research/a.md", "research/b.md"}  # whole vault
    assert body["truncated"] is False
    # degree present for sizing; a<->b is one undirected edge (deduped)
    assert {n["id"]: n["degree"] for n in body["nodes"]}["research/b.md"] >= 1
    assert len(body["links"]) == 1
    assert any(n["current"] for n in body["nodes"] if n["id"] == "research/b.md")


def test_concept_audit_endpoint(monkeypatch):
    expected = {
        "summary": {
            "nodes": 5,
            "research": 3,
            "concept_linked": 2,
            "zero_concept": 1,
            "isolated": 1,
            "isolation_percent": 20.0,
        },
        "overlap": {
            "concept_paper_rows": 4,
            "unique_edges": 3,
            "duplicate_rows": 1,
            "row_avg_concepts_per_paper": 2.0,
            "unique_avg_concepts_per_paper": 1.5,
            "row_multi_concept_percent": 50.0,
            "unique_multi_concept_percent": 50.0,
        },
        "communities": {
            "wikilink_communities": 2,
            "wikilink_modularity": 0.5,
            "overlay_communities": 2,
            "overlay_modularity": 0.6,
            "communities_db_count": 2,
            "largest_community": 3,
        },
        "rules": [
            {
                "target_slug": "Dielectric_Elastomer_Actuator",
                "target": "concepts/Dielectric_Elastomer_Actuator.md",
                "required_tags": ["Dielectric", "Elastomer", "Actuator"],
                "suppressed_parent_slugs": ["Dielectric", "Elastomer", "Actuator"],
                "qualifying_count": 2,
                "routed_count": 2,
                "target_reused_or_created": "reused",
                "suppressed_parent_counts": {
                    "concepts/Actuator.md": 2,
                    "concepts/Dielectric.md": 2,
                    "concepts/Elastomer.md": 2,
                },
            }
        ],
        "hotspots": {
            "top_concepts": [
                {
                    "concept": "concepts/Dielectric_Elastomer_Actuator.md",
                    "title": "Dielectric Elastomer Actuator",
                    "papers": 2,
                }
            ],
            "top_overlapping_papers": [
                {
                    "path": "research/a.md",
                    "title": "A",
                    "row_concepts": 2,
                    "unique_concepts": 2,
                    "concepts": ["Actuator", "Elastomer"],
                }
            ],
            "uncovered_tag_pairs": [
                {
                    "tags": ["Actuator", "Electrode"],
                    "normalized_tags": ["actuator", "electrode"],
                    "count": 2,
                    "samples": [{"path": "research/a.md", "title": "A"}],
                }
            ],
        },
    }
    monkeypatch.setattr(services, "get_concept_audit", lambda: expected, raising=False)

    with TestClient(app) as client:
        resp = client.get("/api/concept-audit")

    assert resp.status_code == 200
    assert resp.json() == expected


@pytest.fixture
def semantic_vault(tmp_path, monkeypatch):
    """Vault exercising the semantic-graph rules: an in-set concept link, a
    similar-only pair (no explicit link), and a meta note that must be excluded."""
    vault_dir = tmp_path / "wiki"
    (vault_dir / "research").mkdir(parents=True)
    (vault_dir / "concepts").mkdir(parents=True)
    (vault_dir / "meta").mkdir(parents=True)
    (vault_dir / "research" / "a.md").write_text(
        "---\naliases: [Note A]\ndomain: research\ncategory: materials\ntags: []\n"
        "sources: []\n---\nSee [[dea]].\n",
        encoding="utf-8",
    )
    (vault_dir / "research" / "b.md").write_text(
        "---\ndomain: research\ncategory: materials\ntags: []\nsources: []\n---\nNo links here.\n",
        encoding="utf-8",
    )
    (vault_dir / "concepts" / "dea.md").write_text(
        "---\naliases: [dea]\ndomain: general\ncategory: concept\ntags: []\nsources: []\n---\n"
        "A concept.\n",
        encoding="utf-8",
    )
    (vault_dir / "meta" / "tag-x.md").write_text(
        "---\ndomain: general\ncategory: tag\ntags: []\nsources: []\n---\nTagged [[a]].\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "linkgraph.sqlite"
    build_index(vault_dir, db_path)

    related_db = tmp_path / "related.sqlite"
    rconn = sqlite3.connect(related_db)
    rconn.execute("CREATE TABLE related (src_path TEXT, dst_path TEXT, score REAL)")
    # a–b are similar but have NO explicit link -> exercises kind:"similar".
    rconn.execute("INSERT INTO related VALUES ('research/a.md', 'research/b.md', 0.95)")
    # stale row pointing at a deleted paper must be ignored.
    rconn.execute("INSERT INTO related VALUES ('research/a.md', 'research/gone.md', 0.9)")
    rconn.commit()
    rconn.close()

    monkeypatch.setattr(services, "VAULT_DIR", vault_dir)
    monkeypatch.setattr(services, "LINKGRAPH_DB", db_path)
    monkeypatch.setattr(services, "RELATED_DB", related_db)
    return vault_dir


def test_graph_excludes_meta_and_tags(semantic_vault):
    with TestClient(app) as client:
        body = client.get("/api/graph", params={"path": "research/a.md", "depth": 2}).json()
    ids = {n["id"] for n in body["nodes"]}
    assert "meta/tag-x.md" not in ids  # scaffolding is not a graph citizen
    assert "concepts/dea.md" in ids  # in-set concept link traversed
    # default (include_similar off) -> no similar edges, b not reached.
    assert all(link["kind"] == "link" for link in body["links"])
    assert "research/b.md" not in ids


def test_graph_include_similar_adds_similar_edges(semantic_vault):
    with TestClient(app) as client:
        body = client.get(
            "/api/graph",
            params={"path": "research/a.md", "depth": 2, "include_similar": "true"},
        ).json()
    ids = {n["id"] for n in body["nodes"]}
    assert "research/b.md" in ids  # similar pair pulls b into the neighborhood
    assert "meta/tag-x.md" not in ids
    kinds = {frozenset((link["source"], link["target"])): link["kind"] for link in body["links"]}
    assert kinds[frozenset(("research/a.md", "research/b.md"))] == "similar"
    assert kinds[frozenset(("research/a.md", "concepts/dea.md"))] == "link"
    # stale related row (research/gone.md) is silently dropped.
    assert "research/gone.md" not in ids


def test_global_graph_includes_community(semantic_vault, tmp_path, monkeypatch):
    comm_db = tmp_path / "communities.sqlite"
    conn = sqlite3.connect(comm_db)
    conn.execute("CREATE TABLE community (path TEXT, community INTEGER)")
    conn.executemany(
        "INSERT INTO community VALUES (?, ?)",
        [("research/a.md", 0), ("research/b.md", 0), ("concepts/dea.md", 1)],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(services, "COMMUNITIES_DB", comm_db)
    with TestClient(app) as client:
        body = client.get("/api/graph/global").json()
    comm = {n["id"]: n.get("community") for n in body["nodes"]}
    assert comm["research/a.md"] == 0
    assert comm["concepts/dea.md"] == 1


def test_global_graph_includes_deterministic_layout_coordinates(
    semantic_vault, tmp_path, monkeypatch
):
    comm_db = tmp_path / "communities.sqlite"
    conn = sqlite3.connect(comm_db)
    conn.execute("CREATE TABLE community (path TEXT, community INTEGER)")
    conn.executemany(
        "INSERT INTO community VALUES (?, ?)",
        [("research/a.md", 0), ("research/b.md", 0), ("concepts/dea.md", 1)],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(services, "COMMUNITIES_DB", comm_db)

    with TestClient(app) as client:
        first = client.get("/api/graph/global", params={"current": "research/a.md"}).json()
        second = client.get("/api/graph/global", params={"current": "research/b.md"}).json()

    first_coords = {node["id"]: (node["x"], node["y"]) for node in first["nodes"]}
    second_coords = {node["id"]: (node["x"], node["y"]) for node in second["nodes"]}

    assert first["layout"] == {"algorithm": "community-ring-v1", "deterministic": True}
    assert first_coords == second_coords
    assert all(
        isinstance(value, int | float) for coords in first_coords.values() for value in coords
    )


def test_graph_global_include_similar(semantic_vault):
    with TestClient(app) as client:
        plain = client.get("/api/graph/global").json()
        withsim = client.get("/api/graph/global", params={"include_similar": "true"}).json()
    assert all(n["folder"] in {"research", "concepts"} for n in plain["nodes"])
    assert "meta/tag-x.md" not in {n["id"] for n in plain["nodes"]}
    assert all(link["kind"] == "link" for link in plain["links"])
    sim_edges = [link for link in withsim["links"] if link["kind"] == "similar"]
    assert any(
        frozenset((link["source"], link["target"])) == frozenset(("research/a.md", "research/b.md"))
        for link in sim_edges
    )


def test_save_reports_index_fresh_on_success(vault):
    with TestClient(app) as client:
        body = client.post("/api/note", json={"path": "research/a.md", "body": "y\n"}).json()
    assert body["ok"] is True
    assert body["index_stale"] is False


def test_save_reports_index_stale_when_reindex_fails(vault, monkeypatch):
    def boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(services, "update_note", boom)
    with TestClient(app) as client:
        resp = client.post("/api/note", json={"path": "research/a.md", "body": "kept\n"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["index_stale"] is True
    # the write still landed on disk even though the reindex failed
    raw = (vault / "research" / "a.md").read_text(encoding="utf-8")
    assert raw.endswith("kept\n")


def test_set_status_reports_index_stale_when_reindex_fails(vault, monkeypatch):
    def boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(services, "update_note", boom)
    with TestClient(app) as client:
        resp = client.post("/api/note/status", json={"path": "research/a.md", "status": "done"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["index_stale"] is True


def test_create_note_reports_index_fresh_on_success(vault):
    with TestClient(app) as client:
        body = client.post("/api/note/create", json={"title": "Fresh One"}).json()
    assert body["ok"] is True
    assert body["index_stale"] is False
