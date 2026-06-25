"""Foundation fact-query primitives: tokenization, keyword/semantic scoring, and the
keyword/semantic `query_facts` retrieval over the fact corpus.

Extracted from the frozen `legacy.graphrag.wiki_query` so the embedding-search foundation
no longer imports the legacy graph layer. Self-contained: only stdlib + IS_A concept
descendant expansion (also foundation).
"""

import json
import math
import re
import sqlite3
from pathlib import Path

QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "address",
    "as",
    "at",
    "by",
    "chang",
    "changes",
    "class",
    "driven",
    "enabl",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "review",
    "the",
    "to",
    "using",
    "very",
    "with",
}
QUERY_TERM_ALIASES = {
    "adjustable": ("tunable",),
    "arrayed": ("array",),
    "boost": ("outperform", "fivefold", "improve", "increase"),
    "boosts": ("outperform", "fivefold", "improve", "increase"),
    "bending": ("bimorph", "strain"),
    "capable": ("exceeded", "exceed"),
    "carrier": ("charge",),
    "carbon": ("cnt",),
    "composite": ("composit",),
    "conductive": ("conductor",),
    "control": ("manipulate", "control"),
    "crystalline": ("crystallite", "crystal"),
    "crystal": ("lce", "liquid", "crystalline"),
    "current": ("dc", "voltage", "resistive"),
    "depth": ("level",),
    "deformation": ("strain",),
    "droplet": ("microliquid", "droplets"),
    "droplets": ("microliquid", "droplet"),
    "electrical": ("electric", "voltage", "v"),
    "elastomer": ("lce", "elastomers"),
    "heated": ("heating", "resistive", "electrothermal"),
    "input": ("voltage", "v"),
    "five": ("5",),
    "large": ("134", "100", "strain"),
    "light": ("optically", "photo"),
    "liquid": ("microliquid", "lce"),
    "magnetoelectric": ("magneto",),
    "nanotube": ("cnt",),
    "one": ("1",),
    "operation": ("actuation", "operating"),
    "polymer": ("ipmc", "eap"),
    "pump": ("pumps",),
    "reducing": ("reduction", "reduce", "low"),
    "rubber": ("elastomer", "elastomers"),
    "sheet": ("paper", "film"),
    "small": ("low", "voltage", "v"),
    "storage": ("trapping", "memory"),
    "stretch": ("strain",),
    "transparent": ("translucent",),
    "hundred": ("100",),
}
VALID_FACT_MATCH_MODES = {"keyword", "semantic"}
TAG_BOOST_WEIGHT = 0.15

DEFAULT_GRAPH_DB = Path(".graph_index/graph.sqlite")


def expand_concept_descendants(
    target_id: str | None,
    graph_db: str | Path = DEFAULT_GRAPH_DB,
) -> set[str]:
    """Return {target_id} ∪ all IS_A descendants. Non-concept targets pass through."""
    if not target_id:
        return set()
    if not target_id.startswith("concept:"):
        return {target_id}

    db_path = Path(graph_db)
    if not db_path.exists():
        return {target_id}

    expanded = {target_id}
    frontier = {target_id}
    conn = sqlite3.connect(db_path)
    try:
        while frontier:
            placeholders = ",".join("?" for _ in frontier)
            rows = conn.execute(
                f"SELECT source FROM edges WHERE type='IS_A' AND target IN ({placeholders})",
                list(frontier),
            ).fetchall()
            new_children = {r[0] for r in rows} - expanded
            if not new_children:
                break
            expanded.update(new_children)
            frontier = new_children
    finally:
        conn.close()
    return expanded


def normalize_query_token(token: str) -> str:
    token = token.strip().lower()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("des") and len(token) > 5:
        return token[:-1]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def tokenize_fact_query(text: str) -> list[str]:
    normalized = re.sub(r"[‐‑‒–—−]", "-", text.lower())
    tokens = []
    for raw_token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", normalized):
        token = normalize_query_token(raw_token)
        tokens.append(token)
        if "-" in token:
            tokens.extend(normalize_query_token(part) for part in token.split("-") if part)
    return [token for token in tokens if token and token not in QUERY_STOPWORDS]


def fact_query_term_matches(term: str, haystack: str) -> bool:
    candidates = {term, *QUERY_TERM_ALIASES.get(term, ())}
    for candidate in candidates:
        normalized_candidate = normalize_query_token(candidate)
        if normalized_candidate in haystack:
            return True
    return False


def expanded_fact_query_tokens(text: str) -> list[str]:
    tokens = []
    for token in tokenize_fact_query(text):
        tokens.append(token)
        tokens.extend(normalize_query_token(alias) for alias in QUERY_TERM_ALIASES.get(token, ()))
    return tokens


def token_counts(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    if dot == 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm)


def fact_semantic_score(query_text: str, fact_text: str, evidence: str) -> float:
    query_vector = token_counts(expanded_fact_query_tokens(query_text))
    fact_vector = token_counts(tokenize_fact_query(f"{fact_text} {evidence}"))
    return cosine_similarity(query_vector, fact_vector)


def load_paper_tags(index_dir: str | Path) -> dict[str, set[str]]:
    db_path = Path(index_dir) / "graph.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Graph index not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        result: dict[str, set[str]] = {}
        for row in conn.execute(
            """
            select e.source, n.label
            from edges e join nodes n on n.id = e.target
            where e.type = 'HAS_TAG'
            """
        ):
            result.setdefault(row[0], set()).add(str(row[1] or "").strip())
        return result
    finally:
        conn.close()


def tag_overlap_score(query_tokens: list[str], paper_tags: set[str]) -> float:
    if not query_tokens or not paper_tags:
        return 0.0
    tag_tokens: set[str] = set()
    for tag in paper_tags:
        tag_tokens.update(tokenize_fact_query(tag))
    overlap = set(query_tokens) & tag_tokens
    return len(overlap) / max(len(query_tokens), 1)


def read_paper_metadata(index_dir: str | Path, paper_ids: list[str]) -> dict[str, dict]:
    if not paper_ids:
        return {}
    db_path = Path(index_dir) / "graph.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Graph index not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        metadata: dict[str, dict] = {}
        for paper_id in sorted(set(paper_ids)):
            row = conn.execute(
                """
                select id, label, props_json
                from nodes
                where id = ? and kind = 'Paper'
                """,
                (paper_id,),
            ).fetchone()
            if row is None:
                continue
            props = json.loads(row["props_json"] or "{}")
            pdf_row = conn.execute(
                """
                select nodes.label, nodes.props_json
                from edges
                join nodes on nodes.id = edges.target
                where edges.source = ? and edges.type = 'HAS_PDF'
                order by nodes.id
                limit 1
                """,
                (paper_id,),
            ).fetchone()
            pdf_path = ""
            if pdf_row is not None:
                pdf_props = json.loads(pdf_row["props_json"] or "{}")
                pdf_path = str(pdf_props.get("path") or pdf_row["label"] or "")
            metadata[paper_id] = {
                "canonical_paper_id": props.get("canonical_paper_id", paper_id),
                "duplicate_of": props.get("duplicate_of", ""),
                "paper_title": row["label"],
                "paper_path": props.get("rel_path", ""),
                "paper_status": props.get("status", ""),
                "pdf_path": pdf_path,
            }
        return metadata
    finally:
        conn.close()


def query_facts(
    index_dir: str | Path,
    paper_id: str | None = None,
    target: str | None = None,
    fact_type: str | None = None,
    text: str | None = None,
    limit: int | None = 25,
    match_mode: str = "keyword",
    expand_concept: bool = True,
) -> list[dict]:
    db_path = Path(index_dir) / "graph.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Graph index not found: {db_path}")

    target_set: set[str] | None = None
    if target:
        if expand_concept:
            target_set = expand_concept_descendants(target, db_path)
        else:
            target_set = {target}

    if match_mode not in VALID_FACT_MATCH_MODES:
        raise ValueError(f"invalid fact match mode: {match_mode}")

    if limit is not None and limit <= 0:
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            select id, label, props_json
            from nodes
            where kind = 'Fact'
            order by id
            """
        ).fetchall()
        parsed_rows = []
        fact_paper_ids = []
        for row in rows:
            props = json.loads(row["props_json"] or "{}")
            fact_paper_id = str(props.get("paper_id", ""))
            if fact_paper_id:
                fact_paper_ids.append(fact_paper_id)
            parsed_rows.append((row, props, fact_paper_id))

        paper_metadata = {}
        if paper_id:
            paper_metadata = read_paper_metadata(index_dir, fact_paper_ids)
        paper_filter = (
            paper_metadata.get(str(paper_id), {}).get("canonical_paper_id") if paper_id else None
        )
        if paper_id and (paper_filter in (None, "")):
            paper_filter = paper_id
        terms = tokenize_fact_query(text or "")
        paper_tags = load_paper_tags(index_dir) if terms else {}
        candidate_rows = []
        for row, props, fact_paper_id in parsed_rows:
            if paper_id:
                fact_paper_canonical = (
                    paper_metadata.get(fact_paper_id, {}).get("canonical_paper_id") or fact_paper_id
                )
                if fact_paper_canonical != paper_filter:
                    continue
            row_fact_type = props.get("fact_type", "")
            if fact_type and row_fact_type != fact_type:
                continue

            fact_text = row["label"]
            evidence = props.get("evidence", "")
            haystack = " ".join(tokenize_fact_query(f"{fact_text} {evidence}"))
            score = None
            if terms:
                if match_mode == "semantic":
                    score = fact_semantic_score(text or "", fact_text, evidence)
                    if score <= 0:
                        continue
                elif not all(fact_query_term_matches(term, haystack) for term in terms):
                    continue

            candidate_rows.append(
                (row, props, fact_paper_id, row_fact_type, fact_text, evidence, score)
            )

        targets_by_fact: dict[str, list[str]] = {row["id"]: [] for row, *_ in candidate_rows}
        if candidate_rows:
            candidate_ids = [row["id"] for row, *_ in candidate_rows]
            batch_size = 500
            for start in range(0, len(candidate_ids), batch_size):
                batch = candidate_ids[start : start + batch_size]
                placeholders = ",".join("?" for _ in batch)
                target_rows = conn.execute(
                    f"""
                    select source, target
                    from edges
                    where type = 'ABOUT' and source in ({placeholders})
                    order by source, target
                    """,
                    batch,
                ).fetchall()
                for target_row in target_rows:
                    targets_by_fact.setdefault(target_row["source"], []).append(
                        target_row["target"]
                    )

        results = []
        for row, props, fact_paper_id, row_fact_type, fact_text, evidence, score in candidate_rows:
            targets = targets_by_fact.get(row["id"], [])
            if target_set and not target_set.intersection(targets):
                continue

            tags_for_paper = paper_tags.get(fact_paper_id, set())
            overlap = tag_overlap_score(terms, tags_for_paper)
            base_score = score if score is not None else 0.0
            boosted_score = base_score + TAG_BOOST_WEIGHT * overlap

            result = {
                "fact_id": row["id"],
                "paper_id": fact_paper_id,
                "fact_type": row_fact_type,
                "text": fact_text,
                "evidence": evidence,
                "confidence": props.get("confidence"),
                "targets": targets,
            }
            if score is not None or overlap > 0:
                result["score"] = round(boosted_score, 6)
            if overlap > 0:
                result["tag_overlap"] = round(overlap, 6)
            results.append(result)

        results.sort(
            key=lambda r: (
                -r.get("score", 0.0),
                r["paper_id"],
                r["fact_id"],
            )
        )
        if limit is not None:
            return results[:limit]
        return results
    finally:
        conn.close()
