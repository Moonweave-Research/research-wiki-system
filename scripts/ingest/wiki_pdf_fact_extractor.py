import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

try:
    from scripts.ingest.wiki_fact_extractor import infer_targets, split_fact_units, target_score
    from legacy.graphrag.wiki_graph import parse_frontmatter
    from legacy.graphrag.wiki_semantic_schema import SemanticFact, fact_from_dict
except ModuleNotFoundError:
    from wiki_fact_extractor import infer_targets, split_fact_units, target_score  # type: ignore[no-redef]
    from wiki_semantic_schema import (  # type: ignore[no-redef]
        SemanticFact,
        fact_from_dict,
    )


PDF_FACT_CONFIDENCE = {
    "Claim": 0.6,
    "Mechanism": 0.66,
}
MIN_UNIT_CHARS = 24
MAX_UNIT_WORDS = 70
BOILERPLATE_RE = re.compile(
    r"\b(corresponding author|tel\.?|fax|all rights reserved|copyright|"
    r"www\.|http|\.com|\.org|indd)\b",
    re.IGNORECASE,
)
PDF_BULLET_LINE_RE = re.compile(r"^\s*[-*]\s+")
WEAK_PDF_TARGET_RE = re.compile(
    r"\b(effect of applied voltage|applied voltage was reported)\b", re.IGNORECASE
)
# Reference lists and table dumps leak metric keywords (paper titles, tabulated
# values) that the target regex matches even though they are not prose facts.
# Two or more inline citation brackets, or in-line math operators in a short
# unit, mark these layout fragments so a raised per-paper limit surfaces real
# facts instead of bibliography/equation noise.
PDF_CITATION_RE = re.compile(r"\[\d{1,3}\]")
PDF_EQUATION_RE = re.compile(r"[∂∫∑∇]")
PDF_EQUATION_MAX_WORDS = 40
PDF_TEXT_TRANSLATION = str.maketrans(
    {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
    }
)


def normalize_pdf_text(text: str) -> str:
    return text.translate(PDF_TEXT_TRANSLATION)


def read_cache_manifest(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict) or "paper_id" not in row or "cached" not in row:
                raise ValueError(f"invalid cached manifest row at line {line_number}")
            cache_path = str(row.get("cache_path") or row.get("markdown_path") or "")
            if row.get("cached") is True and cache_path:
                rows.append({"paper_id": str(row["paper_id"]), "cache_path": cache_path})
    return rows


def source_signature_for(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_signature_for(path: Path) -> str:
    if not path.exists():
        return ""
    stat = path.stat()
    payload = f"{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pdf_facts_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_manifest.jsonl")


def read_pdf_facts_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    signatures: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            paper_id = row.get("paper_id")
            source_sha256 = row.get("source_sha256")
            if isinstance(paper_id, str) and isinstance(source_sha256, str):
                signatures[paper_id] = source_sha256
    return signatures


def read_pdf_facts_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                return []
            if not isinstance(row, dict):
                return []
            if not all(isinstance(row.get(key), str) for key in ("paper_id", "source_sha256")):
                return []
            rows.append({str(key): str(value) for key, value in row.items()})
    return rows


def write_pdf_facts_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def pdf_facts_jsonl_content(facts: list[SemanticFact]) -> str:
    return "".join(json.dumps(fact.to_dict(), sort_keys=True) + "\n" for fact in facts)


def pdf_facts_manifest_content(rows: list[dict[str, str]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_text_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def write_pdf_facts_jsonl_if_changed(path: Path, facts: list[SemanticFact]) -> bool:
    return write_text_if_changed(path, pdf_facts_jsonl_content(facts))


def write_pdf_facts_manifest_if_changed(path: Path, rows: list[dict[str, str]]) -> bool:
    return write_text_if_changed(path, pdf_facts_manifest_content(rows))


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def pdf_facts_manifest_rows(
    manifest_path: Path,
    *,
    sample: int | None = None,
    wiki_dir: Path | None = None,
) -> list[dict[str, str]]:
    rows = read_cache_manifest(manifest_path)
    if sample is not None:
        rows = rows[:sample]
    irrelevant = load_irrelevant_paper_ids(wiki_dir) if wiki_dir is not None else set()
    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        if row["paper_id"] in irrelevant:
            continue
        cache_path = Path(row["cache_path"])
        if not cache_path.exists():
            continue
        manifest_rows.append(
            {
                "paper_id": row["paper_id"],
                "cache_path": row["cache_path"],
                "source_sha256": cache_signature_for(cache_path),
            }
        )
    return manifest_rows


def incremental_fast_path_report(
    output: Path,
    current_manifest_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    irrelevant_paper_ids: set[str],
) -> dict[str, object]:
    facts_count = count_jsonl_rows(output)
    missing_cache = 0
    irrelevant = 0
    for row in source_rows:
        if row["paper_id"] in irrelevant_paper_ids:
            irrelevant += 1
        elif not Path(row["cache_path"]).exists():
            missing_cache += 1
    return {
        "facts": facts_count,
        "unchanged": True,
        "papers": {
            "total": len(source_rows),
            "reused": len(current_manifest_rows),
            "extracted": 0,
            "missing_cache": missing_cache,
            "irrelevant": irrelevant,
            "facts_reused": facts_count,
            "facts_extracted": 0,
        },
        "slowest_papers": [],
        "files": {
            "facts_written": False,
            "manifest_written": False,
        },
    }


def pdf_fact_type(targets: list[str]) -> str:
    if any(target.startswith("mechanism:") for target in targets):
        return "Mechanism"
    return "Claim"


def is_pdf_layout_noise(text: str) -> bool:
    if len(PDF_CITATION_RE.findall(text)) >= 2:
        return True
    if PDF_EQUATION_RE.search(text) and len(text.split()) < PDF_EQUATION_MAX_WORDS:
        return True
    return False


def usable_pdf_unit(text: str) -> bool:
    if len(text) < MIN_UNIT_CHARS or len(text.split()) > MAX_UNIT_WORDS:
        return False
    if BOILERPLATE_RE.search(text) or WEAK_PDF_TARGET_RE.search(text):
        return False
    if is_pdf_layout_noise(text):
        return False
    return True


def pdf_candidate_score(targets: list[str]) -> int:
    score = target_score(targets)
    if any(target.startswith("mechanism:") for target in targets):
        score += 2
    if any(target.startswith("concept:") for target in targets) and not any(
        target.startswith("metric:") for target in targets
    ):
        score += 3
    if len(targets) > 1:
        score += 1
    return score


def split_pdf_fact_units(text: str) -> list[str]:
    bullet_lines: list[str] = []
    prose_lines: list[str] = []
    for line in text.splitlines():
        if PDF_BULLET_LINE_RE.match(line):
            bullet_lines.append(line)
        else:
            prose_lines.append(line)
    units = [
        *split_fact_units("\n".join(bullet_lines)),
        *split_fact_units("\n".join(prose_lines)),
    ]
    seen: set[str] = set()
    unique_units: list[str] = []
    for unit in units:
        normalized = re.sub(r"\s+", " ", unit.strip().lower())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_units.append(unit)
    return unique_units


def pdf_candidate_units(text: str) -> list[tuple[int, str]]:
    units = [unit for unit in split_pdf_fact_units(text) if usable_pdf_unit(unit)]
    candidates = [(order * 2, unit) for order, unit in enumerate(units)]
    for order, (left, right) in enumerate(zip(units, units[1:])):
        left_targets = set(infer_targets(left))
        right_targets = set(infer_targets(right))
        if not left_targets or not right_targets or not left_targets.intersection(right_targets):
            continue
        merged = f"{left} {right}"
        if usable_pdf_unit(merged):
            candidates.append((order * 2 + 1, merged))
    return candidates


def pdf_fact_id_for(paper_id: str, fact_type: str, text: str, occurrence: int) -> str:
    stem = paper_id.removeprefix("paper:")
    normalized = re.sub(r"\s+", " ", f"pdf {fact_type} {text} {occurrence}".strip().lower())
    digest = hashlib.blake2s(normalized.encode("utf-8"), digest_size=8).hexdigest()
    return f"fact:{stem}:pdf_{fact_type.lower()}:{digest}"


def extract_facts_from_cached_text(
    paper_id: str,
    cache_path: Path,
    limit_per_paper: int = 25,
) -> list[SemanticFact]:
    text = normalize_pdf_text(cache_path.read_text(encoding="utf-8"))
    candidates = []
    seen_units = set()
    for order, unit in pdf_candidate_units(text):
        normalized_unit = re.sub(r"\s+", " ", unit.strip().lower())
        if normalized_unit in seen_units:
            continue
        seen_units.add(normalized_unit)
        targets = infer_targets(unit)
        if not targets:
            continue
        candidates.append(
            (-pdf_candidate_score(targets), order, unit, pdf_fact_type(targets), targets)
        )

    facts = []
    occurrences: dict[tuple[str, str], int] = {}
    for _, _, unit, fact_type, targets in sorted(candidates):
        occurrence_key = (fact_type, re.sub(r"\s+", " ", unit.strip().lower()))
        occurrences[occurrence_key] = occurrences.get(occurrence_key, 0) + 1
        facts.append(
            SemanticFact(
                paper_id=paper_id,
                fact_id=pdf_fact_id_for(paper_id, fact_type, unit, occurrences[occurrence_key]),
                fact_type=fact_type,
                text=unit,
                evidence=f"PDF cache: {unit}",
                confidence=PDF_FACT_CONFIDENCE[fact_type],
                targets=targets,
            )
        )
        if len(facts) >= limit_per_paper:
            return facts
    return facts


# SemanticFact has these mandatory fields; rows missing any of them would crash
# fact_from_dict with KeyError during incremental rebuild. We screen them out here
# so a partially-written pdf_facts.jsonl (mid-flush crash, manual edit) recovers
# cleanly on the next run instead of taking down the whole pipeline.
_FACT_REQUIRED_FIELDS = ("paper_id", "fact_id", "fact_type", "text", "evidence", "confidence")


def read_existing_pdf_facts(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    result: dict[str, list[dict]] = {}
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(row, dict):
                skipped += 1
                continue
            if not all(field in row for field in _FACT_REQUIRED_FIELDS):
                skipped += 1
                continue
            pid = str(row["paper_id"])
            result.setdefault(pid, []).append(row)
    if skipped:
        print(
            f"[fact_extractor] skipped {skipped} malformed row(s) in {path}",
            file=sys.stderr,
        )
    return result


def load_irrelevant_paper_ids(wiki_dir: Path) -> set[str]:
    irrelevant: set[str] = set()
    linkgraph_db = Path(".graph_index/linkgraph.sqlite")
    if wiki_dir == Path("wiki") and linkgraph_db.exists():
        try:
            conn = sqlite3.connect(linkgraph_db)
            rows = conn.execute(
                "SELECT path FROM notes WHERE status = 'irrelevant' AND path LIKE 'research/%'"
            ).fetchall()
            conn.close()
            return {f"paper:{Path(path).stem}" for (path,) in rows}
        except sqlite3.Error:
            pass
    research_dir = wiki_dir / "research"
    if not research_dir.is_dir():
        return irrelevant
    for path in research_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        frontmatter, _ = parse_frontmatter(text)
        if str(frontmatter.get("status") or "") == "irrelevant":
            irrelevant.add(f"paper:{path.stem}")
    return irrelevant


def extract_pdf_facts(
    manifest_path: Path,
    limit_per_paper: int = 25,
    sample: int | None = None,
    wiki_dir: Path | None = None,
    incremental: bool = False,
    existing_facts_path: Path | None = None,
    existing_facts_manifest_path: Path | None = None,
) -> list[SemanticFact]:
    facts, _ = extract_pdf_facts_with_report(
        manifest_path,
        limit_per_paper=limit_per_paper,
        sample=sample,
        wiki_dir=wiki_dir,
        incremental=incremental,
        existing_facts_path=existing_facts_path,
        existing_facts_manifest_path=existing_facts_manifest_path,
    )
    return facts


def extract_pdf_facts_with_report(
    manifest_path: Path,
    limit_per_paper: int = 25,
    sample: int | None = None,
    wiki_dir: Path | None = None,
    incremental: bool = False,
    existing_facts_path: Path | None = None,
    existing_facts_manifest_path: Path | None = None,
) -> tuple[list[SemanticFact], dict[str, object]]:
    facts: list[SemanticFact] = []
    rows = read_cache_manifest(manifest_path)
    if sample is not None:
        rows = rows[:sample]
    irrelevant = load_irrelevant_paper_ids(wiki_dir) if wiki_dir is not None else set()
    prior: dict[str, list[dict]] = {}
    if incremental and existing_facts_path is not None:
        prior = read_existing_pdf_facts(existing_facts_path)
    prior_signatures = (
        read_pdf_facts_manifest(existing_facts_manifest_path)
        if incremental and existing_facts_manifest_path is not None
        else {}
    )
    paper_counts = {
        "total": len(rows),
        "reused": 0,
        "extracted": 0,
        "missing_cache": 0,
        "irrelevant": 0,
        "facts_reused": 0,
        "facts_extracted": 0,
    }
    paper_runs: list[dict[str, object]] = []
    for row in rows:
        started = time.perf_counter()
        paper_id = row["paper_id"]
        status = "extracted"
        paper_fact_count = 0
        if row["paper_id"] in irrelevant:
            status = "irrelevant"
            paper_counts["irrelevant"] += 1
            paper_runs.append(
                {
                    "paper_id": paper_id,
                    "status": status,
                    "facts": paper_fact_count,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                }
            )
            continue
        cache_path = Path(row["cache_path"])
        if not cache_path.exists():
            status = "missing_cache"
            paper_counts["missing_cache"] += 1
            paper_runs.append(
                {
                    "paper_id": paper_id,
                    "status": status,
                    "facts": paper_fact_count,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                }
            )
            continue
        cache_sha256 = cache_signature_for(cache_path)
        prior_signature = prior_signatures.get(row["paper_id"])
        if prior_signature is not None and prior_signature != cache_sha256:
            source_sha256 = source_signature_for(cache_path)
        else:
            source_sha256 = ""
        if (
            incremental
            and row["paper_id"] in prior
            and (prior_signature == cache_sha256 or prior_signature == source_sha256)
        ):
            status = "reused"
            for fact_dict in prior[row["paper_id"]]:
                facts.append(fact_from_dict(fact_dict))
                paper_fact_count += 1
            paper_counts["reused"] += 1
            paper_counts["facts_reused"] += paper_fact_count
            paper_runs.append(
                {
                    "paper_id": paper_id,
                    "status": status,
                    "facts": paper_fact_count,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                }
            )
            continue
        extracted = extract_facts_from_cached_text(
            row["paper_id"],
            cache_path,
            limit_per_paper=limit_per_paper,
        )
        paper_fact_count = len(extracted)
        facts.extend(extracted)
        paper_counts["extracted"] += 1
        paper_counts["facts_extracted"] += paper_fact_count
        paper_runs.append(
            {
                "paper_id": paper_id,
                "status": status,
                "facts": paper_fact_count,
                "elapsed_s": round(time.perf_counter() - started, 3),
            }
        )
    report: dict[str, object] = {
        "facts": len(facts),
        "papers": paper_counts,
        "slowest_papers": sorted(
            paper_runs,
            key=lambda run: float(run["elapsed_s"]),
            reverse=True,
        )[:10],
    }
    return facts, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract draft semantic facts from cached PDF text."
    )
    parser.add_argument("--manifest", default=".graph_index/pdf_text_manifest.jsonl")
    parser.add_argument("--output", default=".graph_index/pdf_facts.jsonl")
    parser.add_argument(
        "--facts-manifest",
        help="Sidecar manifest used to safely reuse facts only when cached source text is unchanged.",
    )
    parser.add_argument("--wiki-dir", default="wiki")
    parser.add_argument("--limit-per-paper", type=int, default=25)
    parser.add_argument(
        "--sample", type=int, help="Extract facts from only the first N cached rows"
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip paper_ids already present in the output facts file. Existing facts for those papers are preserved verbatim.",
    )
    args = parser.parse_args(argv)

    if args.limit_per_paper <= 0:
        print("limit-per-paper must be positive")
        raise SystemExit(2)
    if args.sample is not None and args.sample <= 0:
        print("sample must be positive")
        raise SystemExit(2)

    try:
        output = Path(args.output)
        facts_manifest = (
            Path(args.facts_manifest) if args.facts_manifest else pdf_facts_manifest_path(output)
        )
        manifest_path = Path(args.manifest)
        wiki_dir = Path(args.wiki_dir)
        source_rows = read_cache_manifest(manifest_path)
        if args.sample is not None:
            source_rows = source_rows[: args.sample]
        irrelevant_paper_ids = load_irrelevant_paper_ids(wiki_dir)
        manifest_rows = pdf_facts_manifest_rows(
            manifest_path,
            sample=args.sample,
            wiki_dir=wiki_dir,
        )
        if (
            args.incremental
            and output.exists()
            and facts_manifest.exists()
            and read_pdf_facts_manifest_rows(facts_manifest) == manifest_rows
        ):
            report = incremental_fast_path_report(
                output,
                manifest_rows,
                source_rows,
                irrelevant_paper_ids,
            )
            if args.json:
                print(json.dumps({"facts_written": str(output), **report}, sort_keys=True))
                return
            print(f"facts_written: {output}")
            print(f"facts: {report['facts']}")
            return
        facts, report = extract_pdf_facts_with_report(
            manifest_path,
            limit_per_paper=args.limit_per_paper,
            sample=args.sample,
            wiki_dir=wiki_dir,
            incremental=args.incremental,
            existing_facts_path=output if args.incremental else None,
            existing_facts_manifest_path=facts_manifest if args.incremental else None,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    facts_written = write_pdf_facts_jsonl_if_changed(output, facts)
    manifest_written = write_pdf_facts_manifest_if_changed(
        facts_manifest,
        manifest_rows,
    )
    report["files"] = {
        "facts_written": facts_written,
        "manifest_written": manifest_written,
    }
    if args.json:
        print(json.dumps({"facts_written": str(output), **report}, sort_keys=True))
        return
    print(f"facts_written: {output}")
    print(f"facts: {len(facts)}")


if __name__ == "__main__":
    main()
