import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np

from scripts.search.fact_query import (
    fact_semantic_score,
    query_facts,
    read_paper_metadata,
    tokenize_fact_query,
)


EMBEDDING_VECTOR_FILE = "fact_embeddings.npz"
EMBEDDING_MANIFEST_FILE = "fact_embeddings_manifest.jsonl"
EMBEDDING_META_FILE = "fact_embeddings_meta.json"
EMBEDDING_CACHE_DB_FILE = "fact_embeddings_cache.sqlite"
try:
    from scripts.search.wiki_embedding_defaults import (
        DEFAULT_FASTEMBED_MODEL,
        EMBEDDING_DIM_BY_MODEL,
    )
except ModuleNotFoundError:
    from wiki_embedding_defaults import DEFAULT_FASTEMBED_MODEL, EMBEDDING_DIM_BY_MODEL  # type: ignore[no-redef]
DEFAULT_CANDIDATE_LIMIT = 80
DEFAULT_EMBEDDING_BATCH_SIZE = 128
DEFAULT_SCORE_MARGIN = 0.02
DEFAULT_KEYWORD_WEIGHT = 0.3
TITLE_SCORE_WEIGHT = 0.45
TITLE_STRONG_MATCH_BONUS = 0.5
TITLE_STRONG_MATCH_THRESHOLD = 0.62
BODY_OVERLAP_WEIGHT = 0.35
BODY_OVERLAP_BONUS = 0.08
BODY_OVERLAP_BONUS_THRESHOLD = 0.7
TITLE_EVIDENCE_THRESHOLD = 0.2
REVIEW_INTENT_BONUS = 0.25
NON_REVIEW_TITLE_PENALTY = 0.08
LOW_VOLTAGE_HIGH_FORCE_TITLE_BONUS = 0.06
NUMERIC_METRIC_EVIDENCE_BONUS = 0.08
SHEET_EVIDENCE_BONUS = 0.07
REVIEW_QUERY_RE = re.compile(r"\b(review|survey|overview)\b", re.IGNORECASE)
REVIEW_TITLE_RE = re.compile(r"\b(review|survey|overview|perspective)\b", re.IGNORECASE)
HYPHEN_RE = re.compile(r"[‐‑‒–—−]")
ONE_HUNDRED_RE = re.compile(r"(?:\b100\s*%|\b100\s*percent|\bexceed(?:ed|s)?\s+100)")


class EmbeddingProvider(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_E5_MODEL_PREFIX = ("intfloat/multilingual-e5", "intfloat/e5")


class FastEmbedProvider:
    def __init__(self, model_name: str = DEFAULT_FASTEMBED_MODEL):
        self.model_name = model_name
        self._model = None
        self._query_cache: dict[str, list[float]] = {}

    def _embedding_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "fastembed is not installed. Install it before using --provider fastembed."
                ) from exc
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _is_e5(self) -> bool:
        return self.model_name.startswith(_E5_MODEL_PREFIX)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._is_e5():
            texts = [f"passage: {text}" for text in texts]
        return [vector.tolist() for vector in self._embedding_model().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        query_text = f"query: {text}" if self._is_e5() else text
        vec = next(self._embedding_model().query_embed(query_text)).tolist()
        if len(self._query_cache) < 256:
            self._query_cache[text] = vec
        return vec


def _provider_dim(provider: "EmbeddingProvider") -> int | None:
    dim = getattr(provider, "dim", None)
    if dim is not None:
        return int(dim)
    return EMBEDDING_DIM_BY_MODEL.get(provider.model_name)


_PROVIDER_CACHE: dict[tuple[str, str], "EmbeddingProvider"] = {}


def get_cached_provider(provider_name: str, model_name: str) -> "EmbeddingProvider":
    key = (provider_name, model_name)
    cached = _PROVIDER_CACHE.get(key)
    if cached is not None:
        return cached
    if provider_name == "fastembed":
        provider: EmbeddingProvider = FastEmbedProvider(model_name=model_name)
    else:
        raise ValueError(f"unknown provider: {provider_name}")
    _PROVIDER_CACHE[key] = provider
    return provider


# The fact corpus, mirroring legacy.graphrag.wiki_graph._load_all_facts (which
# fed graph.sqlite): rule-based facts, then PDF-derived, then the LLM-augment
# overlay, deduped by fact_id in that precedence. Reading them directly removes
# the embeddings build's dependency on the (now orphaned) graph.sqlite.
FACT_SOURCE_FILES = ("facts.jsonl", "pdf_facts.jsonl", "facts_llm_augment.jsonl")


def _fact_source_paths(index_dir: str | Path) -> list[Path]:
    return [Path(index_dir) / name for name in FACT_SOURCE_FILES]


def read_fact_records(index_dir: str | Path) -> list[dict]:
    """Load fact records from the fact-corpus JSONL files (facts.jsonl +
    pdf_facts.jsonl + facts_llm_augment.jsonl), deduped by fact_id in that
    precedence, ordered by fact_id. Replaces the legacy read of graph.sqlite
    nodes(kind='Fact') + edges(type='ABOUT'); the JSONL lines already carry
    id/paper/type/text/evidence/confidence/targets, so the mapping is 1:1."""
    by_id: dict[str, dict] = {}
    for path in _fact_source_paths(index_dir):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                raw = json.loads(stripped)
                fact_id = raw.get("fact_id")
                if not fact_id or fact_id in by_id:
                    continue
                by_id[fact_id] = {
                    "fact_id": fact_id,
                    "paper_id": raw.get("paper_id", ""),
                    "fact_type": raw.get("fact_type", ""),
                    "text": raw.get("text", ""),
                    "evidence": raw.get("evidence", ""),
                    "confidence": raw.get("confidence"),
                    "targets": list(raw.get("targets", [])),
                }
    return [by_id[fact_id] for fact_id in sorted(by_id)]


def fact_embedding_text(record: dict) -> str:
    return " ".join(
        part
        for part in (
            record.get("text", ""),
            record.get("evidence", ""),
            " ".join(record.get("targets", [])),
        )
        if part
    )


def embed_documents_in_batches(
    provider: EmbeddingProvider,
    texts: list[str],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(provider.embed_documents(texts[start : start + batch_size]))
    return vectors


def _write_manifest(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _read_manifest(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict) or "fact_id" not in row:
                raise ValueError(f"invalid embedding manifest row at line {line_number}")
            records.append(row)
    return records


def _embedding_record_key(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _embedding_cache_path(index_path: Path) -> Path:
    return index_path / EMBEDDING_CACHE_DB_FILE


def _fastembed_version() -> str:
    # The fastembed version selects the pooling for e5 models (0.5.1 = CLS,
    # >=0.8.0 = mean). The cache/npz key on model name only, so a version change
    # would silently mix incompatible vectors; we record it to invalidate reuse.
    try:
        from importlib.metadata import version

        return version("fastembed")
    except Exception:
        return "unknown"


def _meta_version_status(index_path: Path, fastembed_version: str) -> str:
    # "match"    -> existing index built with this fastembed; reuse is safe.
    # "mismatch" -> index built with a different (or unrecorded) version; its
    #               pooling may differ, so prior vectors must be purged.
    # "absent"   -> no meta yet (fresh or interrupted build); keep the cache so a
    #               crashed run can resume its partial batches.
    meta_path = index_path / EMBEDDING_META_FILE
    if not meta_path.exists():
        return "absent"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "mismatch"
    return "match" if meta.get("fastembed_version") == fastembed_version else "mismatch"


def _clear_embedding_cache(index_path: Path) -> None:
    try:
        _embedding_cache_path(index_path).unlink()
    except FileNotFoundError:
        pass


def _ensure_embedding_cache(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists embedding_cache (
            model text not null,
            record_key text not null,
            record_json text not null,
            dim integer not null,
            vector blob not null,
            primary key (model, record_key)
        )
        """
    )


def _load_reusable_embedding_vectors(
    index_path: Path,
    provider: EmbeddingProvider,
) -> dict[str, np.ndarray]:
    vector_path = index_path / EMBEDDING_VECTOR_FILE
    manifest_path = index_path / EMBEDDING_MANIFEST_FILE
    meta_path = index_path / EMBEDDING_META_FILE
    if not vector_path.exists() or not manifest_path.exists() or not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model") != provider.model_name:
            return {}
        old_records = _read_manifest(manifest_path)
        with np.load(vector_path) as data:
            old_vectors = data["vectors"]
        if len(old_records) != len(old_vectors):
            return {}
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return {}
    return {
        _embedding_record_key(record): old_vectors[index]
        for index, record in enumerate(old_records)
    }


def _load_cached_embedding_vectors(
    index_path: Path,
    provider: EmbeddingProvider,
) -> dict[str, np.ndarray]:
    cache_path = _embedding_cache_path(index_path)
    if not cache_path.exists():
        return {}
    conn = sqlite3.connect(cache_path)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_embedding_cache(conn)
        rows = conn.execute(
            "select record_key, dim, vector from embedding_cache where model = ?",
            (provider.model_name,),
        ).fetchall()
    finally:
        conn.close()
    cached = {}
    for row in rows:
        vector = np.frombuffer(row["vector"], dtype=np.float32)
        if vector.shape != (int(row["dim"]),):
            continue
        cached[row["record_key"]] = vector
    return cached


def _store_cached_embedding_vectors(
    index_path: Path,
    provider: EmbeddingProvider,
    records: list[dict],
    vectors: list[list[float]],
) -> None:
    if not records:
        return
    if len(records) != len(vectors):
        raise ValueError("embedding provider returned a different number of vectors")
    conn = sqlite3.connect(_embedding_cache_path(index_path))
    try:
        _ensure_embedding_cache(conn)
        rows = []
        for record, vector in zip(records, vectors):
            array = np.asarray(vector, dtype=np.float32)
            if array.ndim != 1:
                raise ValueError("embedding provider must return 1D vectors")
            rows.append(
                (
                    provider.model_name,
                    _embedding_record_key(record),
                    json.dumps(record, ensure_ascii=False, sort_keys=True),
                    int(array.shape[0]),
                    array.tobytes(),
                )
            )
        conn.executemany(
            """
            insert or replace into embedding_cache
                (model, record_key, record_json, dim, vector)
            values (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _current_embedding_report(
    index_path: Path,
    provider: EmbeddingProvider,
    records: list[dict],
) -> dict | None:
    vector_path = index_path / EMBEDDING_VECTOR_FILE
    manifest_path = index_path / EMBEDDING_MANIFEST_FILE
    meta_path = index_path / EMBEDDING_META_FILE
    if not vector_path.exists() or not manifest_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model") != provider.model_name:
            return None
        if int(meta.get("count", -1)) != len(records):
            return None
        if _read_manifest(manifest_path) != records:
            return None
        dimensions = int(meta.get("dim") or 0)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return {
        "model": provider.model_name,
        "facts": len(records),
        "dimensions": dimensions,
        "vectors": str(vector_path),
        "manifest": str(manifest_path),
        "reused": len(records),
        "embedded": 0,
        "skipped_current": True,
    }


def _facts_file_signature(index_path: Path) -> dict[str, dict[str, int]]:
    """Freshness signature of the fact-corpus JSONL files (mtime + size), so the
    embeddings build can skip when the corpus is unchanged — without requiring
    graph.sqlite to exist."""
    signature: dict[str, dict[str, int]] = {}
    for path in _fact_source_paths(index_path):
        if path.exists():
            stat = path.stat()
            signature[path.name] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    return signature


def _current_embedding_report_from_meta(
    index_path: Path,
    provider: EmbeddingProvider,
    facts_signature: dict[str, dict[str, int]],
) -> dict | None:
    vector_path = index_path / EMBEDDING_VECTOR_FILE
    manifest_path = index_path / EMBEDDING_MANIFEST_FILE
    meta_path = index_path / EMBEDDING_META_FILE
    if not vector_path.exists() or not manifest_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model") != provider.model_name:
            return None
        if meta.get("facts_signature") != facts_signature:
            return None
        count = int(meta.get("count", -1))
        dimensions = int(meta.get("dim") or 0)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return {
        "model": provider.model_name,
        "facts": count,
        "dimensions": dimensions,
        "vectors": str(vector_path),
        "manifest": str(manifest_path),
        "reused": count,
        "embedded": 0,
        "skipped_current": True,
    }


def build_fact_embedding_index(
    index_dir: str | Path,
    provider: EmbeddingProvider,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> dict:
    import datetime

    index_path = Path(index_dir)
    facts_signature = _facts_file_signature(index_path)
    fastembed_version = _fastembed_version()
    # A pooling-changing fastembed upgrade invalidates every prior vector: skip the
    # fast paths and reuse caches, and drop the stale cache so the rebuild is pure.
    version_status = _meta_version_status(index_path, fastembed_version)
    version_ok = version_status == "match"
    if version_ok:
        current_report = _current_embedding_report_from_meta(index_path, provider, facts_signature)
        if current_report is not None:
            return current_report
    records = read_fact_records(index_path)
    if version_ok:
        current_report = _current_embedding_report(index_path, provider, records)
        if current_report is not None:
            return current_report
    texts = [fact_embedding_text(record) for record in records]
    if version_status == "mismatch":
        reusable_vectors = {}
        _clear_embedding_cache(index_path)
    else:
        reusable_vectors = _load_reusable_embedding_vectors(index_path, provider)
        reusable_vectors.update(_load_cached_embedding_vectors(index_path, provider))
    vector_slots: list[np.ndarray | list[float] | None] = [None] * len(records)
    pending_texts: list[str] = []
    pending_indexes: list[int] = []
    reused_count = 0
    for index, record in enumerate(records):
        reused = reusable_vectors.get(_embedding_record_key(record))
        if reused is not None:
            vector_slots[index] = reused
            reused_count += 1
            continue
        pending_indexes.append(index)
        pending_texts.append(texts[index])

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(pending_texts), batch_size):
        batch_indexes = pending_indexes[start : start + batch_size]
        batch_texts = pending_texts[start : start + batch_size]
        batch_records = [records[index] for index in batch_indexes]
        batch_vectors = provider.embed_documents(batch_texts)
        _store_cached_embedding_vectors(index_path, provider, batch_records, batch_vectors)
        for index, vector in zip(batch_indexes, batch_vectors):
            vector_slots[index] = vector
    vectors = np.asarray(vector_slots, dtype=np.float32)
    if records and vectors.ndim != 2:
        raise ValueError("embedding provider must return a 2D vector list")

    vector_path = index_path / EMBEDDING_VECTOR_FILE
    manifest_path = index_path / EMBEDDING_MANIFEST_FILE
    meta_path = index_path / EMBEDDING_META_FILE
    np.savez_compressed(vector_path, vectors=vectors)
    _write_manifest(manifest_path, records)
    dimensions = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[0] else 0
    meta = {
        "model": provider.model_name,
        "dim": dimensions,
        "count": len(records),
        "facts_signature": facts_signature,
        "fastembed_version": fastembed_version,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The GraphRAG whole-index manifest.json (graph.sqlite, communities, ...) is a
    # legacy concern this foundation builder must not own; refresh-graph step 7 and
    # the standalone `index_manifest write` CLI refresh it after all artifacts exist.
    return {
        "model": provider.model_name,
        "facts": len(records),
        "dimensions": dimensions,
        "vectors": str(vector_path),
        "manifest": str(manifest_path),
        "reused": reused_count,
        "embedded": len(pending_texts),
        "skipped_current": False,
    }


def cosine_scores(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    if vectors.size == 0 or query.size == 0:
        return np.asarray([], dtype=np.float32)
    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query)
    denominator = vector_norms * query_norm
    denominator[denominator == 0] = 1.0
    return (vectors @ query) / denominator


def query_fact_embeddings(
    index_dir: str | Path,
    query: str,
    provider: EmbeddingProvider,
    target: str | None = None,
    fact_type: str | None = None,
    paper_id: str | None = None,
    limit: int = 5,
    min_score: float = 0.0,
    deduplicate_papers: bool = True,
    expand_concept: bool = True,
) -> list[dict]:
    if limit <= 0:
        return []
    index_path = Path(index_dir)
    vector_path = index_path / EMBEDDING_VECTOR_FILE
    manifest_path = index_path / EMBEDDING_MANIFEST_FILE
    if not vector_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Fact embedding index not found in: {index_path}")

    meta_path = index_path / EMBEDDING_META_FILE
    if meta_path.exists():
        sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
        sidecar_model = sidecar.get("model")
        sidecar_dim = sidecar.get("dim")
        provider_dim = _provider_dim(provider)
        if sidecar_model is not None and sidecar_model != provider.model_name:
            raise ValueError(
                f"embedding model mismatch: index built with '{sidecar_model}', "
                f"provider is '{provider.model_name}'"
            )
        if sidecar_dim is not None and provider_dim is not None and sidecar_dim != provider_dim:
            raise ValueError(
                f"embedding dim mismatch: index dim={sidecar_dim}, provider dim={provider_dim}"
            )

    target_set: set[str] | None = None
    if target:
        if expand_concept:
            from scripts.search.fact_query import expand_concept_descendants

            target_set = expand_concept_descendants(target, index_path / "graph.sqlite")
        else:
            target_set = {target}

    records = _read_manifest(manifest_path)
    vectors = np.load(vector_path)["vectors"].astype(np.float32)
    query_vector = np.asarray(provider.embed_query(query), dtype=np.float32)
    scores = cosine_scores(vectors, query_vector)
    paper_metadata = read_paper_metadata(
        index_path,
        [str(record.get("paper_id", "")) for record in records if record.get("paper_id")],
    )

    results = []
    best_by_paper: dict[str, dict] = {}
    for index, score in sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True):
        record = records[index]
        if target_set and not target_set.intersection(record.get("targets", [])):
            continue
        if fact_type and fact_type != record.get("fact_type"):
            continue
        if paper_id and paper_id != record.get("paper_id"):
            continue
        if float(score) < min_score:
            continue
        result = dict(record)
        result["score"] = round(float(score), 6)
        metadata = paper_metadata.get(str(result.get("paper_id", "")), {})
        result["canonical_paper_id"] = metadata.get(
            "canonical_paper_id",
            result.get("paper_id", ""),
        )
        if metadata.get("duplicate_of"):
            result["duplicate_of"] = metadata["duplicate_of"]
        if not deduplicate_papers:
            results.append(result)
            if len(results) >= limit:
                break
            continue
        paper_key = result.get("canonical_paper_id") or result.get("paper_id", "")
        current = best_by_paper.get(paper_key)
        if current is None or result["score"] > current["score"]:
            best_by_paper[paper_key] = result
        if len(best_by_paper) >= limit:
            break
    return list(best_by_paper.values()) if deduplicate_papers else results


def exact_token_overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize_fact_query(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize_fact_query(text))
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def query_specificity_bonus(query: str, paper_title: str, fact_text: str, evidence: str) -> float:
    query_lower = HYPHEN_RE.sub("-", query.lower())
    title_lower = HYPHEN_RE.sub("-", paper_title.lower())
    text_lower = HYPHEN_RE.sub("-", f"{paper_title} {fact_text} {evidence}".lower())
    query_tokens = set(tokenize_fact_query(query))

    bonus = 0.0
    if (
        "force" in query_tokens
        and "voltage" in query_tokens
        and "low-voltage" in title_lower
        and "high-force" in title_lower
    ):
        bonus += LOW_VOLTAGE_HIGH_FORCE_TITLE_BONUS
    if (
        "hundred" in query_tokens or "100" in query_tokens or "percent" in query_lower
    ) and ONE_HUNDRED_RE.search(text_lower):
        bonus += NUMERIC_METRIC_EVIDENCE_BONUS
    if ("sheet" in query_tokens or "sheets" in query_tokens) and re.search(
        r"\bsheets?\b",
        text_lower,
    ):
        bonus += SHEET_EVIDENCE_BONUS
    return bonus


def query_hybrid_fact_embeddings(
    index_dir: str | Path,
    query: str,
    provider: EmbeddingProvider,
    target: str | None = None,
    fact_type: str | None = None,
    paper_id: str | None = None,
    limit: int = 5,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    score_margin: float = DEFAULT_SCORE_MARGIN,
) -> list[dict]:
    embedding_candidates = query_fact_embeddings(
        index_dir,
        query,
        provider,
        target=target,
        fact_type=fact_type,
        paper_id=paper_id,
        limit=max(candidate_limit, limit),
        deduplicate_papers=False,
    )
    candidates_by_id = {candidate["fact_id"]: candidate for candidate in embedding_candidates}
    for lexical_candidate in query_facts(
        index_dir,
        target=target,
        fact_type=fact_type,
        paper_id=paper_id,
        text=query,
        limit=max(candidate_limit, limit),
        match_mode="semantic",
    ):
        candidates_by_id.setdefault(lexical_candidate["fact_id"], lexical_candidate)

    paper_metadata = read_paper_metadata(
        index_dir,
        [candidate.get("paper_id", "") for candidate in candidates_by_id.values()],
    )
    best_by_paper: dict[str, dict] = {}
    has_review_intent = bool(REVIEW_QUERY_RE.search(query))
    for candidate in candidates_by_id.values():
        metadata = paper_metadata.get(candidate.get("paper_id", ""), {})
        paper_title = metadata.get("paper_title", "")
        lexical_score = fact_semantic_score(
            query,
            candidate.get("text", ""),
            candidate.get("evidence", ""),
        )
        title_score = fact_semantic_score(
            query,
            paper_title,
            "",
        )
        body_overlap_score = exact_token_overlap_score(
            query,
            f"{candidate.get('text', '')} {candidate.get('evidence', '')}",
        )
        has_fact_evidence = (
            max(float(lexical_score), float(body_overlap_score)) >= TITLE_EVIDENCE_THRESHOLD
        )
        title_overlap_score = exact_token_overlap_score(query, paper_title)
        title_match_score = (
            TITLE_STRONG_MATCH_BONUS
            if has_fact_evidence
            and (
                title_score >= TITLE_STRONG_MATCH_THRESHOLD
                or title_overlap_score >= TITLE_STRONG_MATCH_THRESHOLD
            )
            else 0.0
        )
        title_signal_score = TITLE_SCORE_WEIGHT * title_score if has_fact_evidence else 0.0
        body_match_score = BODY_OVERLAP_WEIGHT * body_overlap_score
        if body_overlap_score >= BODY_OVERLAP_BONUS_THRESHOLD:
            body_match_score += BODY_OVERLAP_BONUS
        specificity_bonus = query_specificity_bonus(
            query,
            paper_title,
            candidate.get("text", ""),
            candidate.get("evidence", ""),
        )
        review_score = (
            REVIEW_INTENT_BONUS
            if has_review_intent and has_fact_evidence and REVIEW_TITLE_RE.search(paper_title)
            else 0.0
        )
        review_penalty = (
            NON_REVIEW_TITLE_PENALTY
            if not has_review_intent and REVIEW_TITLE_RE.search(paper_title)
            else 0.0
        )
        embedding_score = float(candidate.get("score", 0.0))
        if lexical_score >= 0.8:
            # Strong lexical evidence should outrank fuzzy embedding-only neighbors.
            combined_score = (
                1.0
                + lexical_score
                + title_signal_score
                + title_match_score
                + body_match_score
                + specificity_bonus
                + review_score
                - review_penalty
                + keyword_weight * embedding_score
            )
        else:
            combined_score = (
                embedding_score
                + keyword_weight * lexical_score
                + title_signal_score
                + title_match_score
                + body_match_score
                + specificity_bonus
                + review_score
                - review_penalty
            )
        result = dict(candidate)
        result["embedding_score"] = round(embedding_score, 6)
        result["lexical_score"] = round(float(lexical_score), 6)
        result["title_score"] = round(float(title_score), 6)
        result["title_signal_score"] = round(float(title_signal_score), 6)
        result["title_overlap_score"] = round(float(title_overlap_score), 6)
        result["title_match_score"] = round(float(title_match_score), 6)
        result["body_overlap_score"] = round(float(body_overlap_score), 6)
        result["body_match_score"] = round(float(body_match_score), 6)
        result["specificity_bonus"] = round(float(specificity_bonus), 6)
        result["review_score"] = round(float(review_score), 6)
        result["review_penalty"] = round(float(review_penalty), 6)
        result["score"] = round(combined_score, 6)
        result["canonical_paper_id"] = metadata.get(
            "canonical_paper_id",
            result.get("paper_id", ""),
        )
        if metadata.get("duplicate_of"):
            result["duplicate_of"] = metadata["duplicate_of"]
        paper = result.get("canonical_paper_id") or result.get("paper_id", "")
        current = best_by_paper.get(paper)
        if current is None or result["score"] > current["score"]:
            best_by_paper[paper] = result

    ranked = sorted(
        best_by_paper.values(),
        key=lambda result: (
            -result.get("score", 0.0),
            -result.get("lexical_score", 0.0),
            -result.get("title_score", 0.0),
            -result.get("title_overlap_score", 0.0),
            -result.get("body_overlap_score", 0.0),
            -result.get("embedding_score", 0.0),
            result["paper_id"],
            result["fact_id"],
        ),
    )
    if not ranked or limit <= 0:
        return []
    top_score = float(ranked[0].get("score", 0.0))
    return [
        result
        for result in ranked[:limit]
        if float(result.get("score", 0.0)) >= top_score - score_margin
    ]


def provider_from_name(provider_name: str, model_name: str) -> EmbeddingProvider:
    if provider_name == "fastembed":
        return FastEmbedProvider(model_name=model_name)
    raise ValueError(f"unsupported embedding provider: {provider_name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or query local fact embedding indexes.")
    parser.add_argument("command", choices=["build", "query"])
    parser.add_argument("--graph-index", default=".graph_index")
    parser.add_argument("--provider", choices=["fastembed"], default="fastembed")
    parser.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    parser.add_argument("--text", help="Query text for embedding search")
    parser.add_argument("--target")
    parser.add_argument("--fact-type")
    parser.add_argument("--paper")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)
    parser.add_argument("--score-margin", type=float, default=DEFAULT_SCORE_MARGIN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        provider = provider_from_name(args.provider, args.model)
        if args.command == "build":
            report = build_fact_embedding_index(
                args.graph_index,
                provider,
                batch_size=args.batch_size,
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True) if args.json else report)
            return
        if not args.text:
            print("embedding query requires --text")
            raise SystemExit(2)
        if args.hybrid:
            results = query_hybrid_fact_embeddings(
                args.graph_index,
                args.text,
                provider,
                target=args.target,
                fact_type=args.fact_type,
                paper_id=args.paper,
                limit=args.limit,
                candidate_limit=args.candidate_limit,
                keyword_weight=args.keyword_weight,
                score_margin=args.score_margin,
            )
        else:
            results = query_fact_embeddings(
                args.graph_index,
                args.text,
                provider,
                target=args.target,
                fact_type=args.fact_type,
                paper_id=args.paper,
                limit=args.limit,
                min_score=args.min_score,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    if args.json:
        print(json.dumps({"results": results}, ensure_ascii=False, sort_keys=True))
        return
    for result in results:
        print(f"{result['score']:.3f} {result['paper_id']} {result['fact_id']} {result['text']}")


if __name__ == "__main__":
    main()
