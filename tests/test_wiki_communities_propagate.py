import json
import sqlite3
from pathlib import Path


def test_propagate_to_facts(tmp_path: Path) -> None:
    from legacy.graphrag.wiki_communities import propagate_to_facts

    db = tmp_path / "g.sqlite"
    con = sqlite3.connect(db)
    con.executescript("""
      CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, label TEXT, props_json TEXT);
      CREATE TABLE edges (source TEXT, type TEXT, target TEXT, props_json TEXT);
    """)
    con.execute(
        "INSERT INTO nodes VALUES ('paper:p1','Paper','p1', ?)",
        (json.dumps({"community_id": 5}),),
    )
    con.execute("INSERT INTO nodes VALUES ('fact:p1:c0','Fact','f', '{}')")
    con.execute("INSERT INTO edges VALUES ('paper:p1','HAS_FACT','fact:p1:c0', '{}')")
    con.commit()

    propagate_to_facts(con)

    row = con.execute("SELECT props_json FROM nodes WHERE id='fact:p1:c0'").fetchone()
    props = json.loads(row[0])
    assert props["community_id"] == 5


def test_propagate_to_concepts_majority_vote(tmp_path: Path) -> None:
    from legacy.graphrag.wiki_communities import propagate_to_concept_like

    db = tmp_path / "g.sqlite"
    con = sqlite3.connect(db)
    con.executescript("""
      CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, label TEXT, props_json TEXT);
      CREATE TABLE edges (source TEXT, type TEXT, target TEXT, props_json TEXT);
    """)
    for i, cid in enumerate([1, 1, 1, 2]):
        con.execute(
            "INSERT INTO nodes VALUES (?,'Paper','p',?)",
            (f"paper:p{i}", json.dumps({"community_id": cid})),
        )
        con.execute(
            "INSERT INTO nodes VALUES (?,'Fact','f','{}')",
            (f"fact:p{i}:c0",),
        )
        con.execute(
            "INSERT INTO edges VALUES (?, 'HAS_FACT', ?, '{}')",
            (f"paper:p{i}", f"fact:p{i}:c0"),
        )
        con.execute(
            "INSERT INTO edges VALUES (?, 'ABOUT', 'concept:Foo', '{}')",
            (f"fact:p{i}:c0",),
        )
    con.execute("INSERT INTO nodes VALUES ('concept:Foo','Concept','Foo','{}')")
    con.commit()

    n = propagate_to_concept_like(con, kinds=("Concept",))
    assert n == 1
    props = json.loads(
        con.execute("SELECT props_json FROM nodes WHERE id='concept:Foo'").fetchone()[0]
    )
    assert props["community_id"] == 1
