import hashlib
import json as _json
import re
import sys
from pathlib import Path

from scripts.core.wiki_llm_provider import ProviderError, provider_call
from legacy.graphrag.wiki_semantic_schema import (
    ALLOWED_FACT_TYPES,
    SemanticFact,
    SemanticTriple,
    _stable_triple_id,
)

DEFAULT_MODEL_FLAG = "sonnet"  # passed to `claude --model <flag>` (claude provider only)


_TRIPLE_EXTRACTOR_SYSTEM = (
    "You extract (subject, predicate, object) triples from polymer/materials "
    "science research papers. Return JSON only — no prose, no markdown fences. "
    "Output a JSON array where each item has keys: subject, predicate, object, "
    "evidence, confidence. Empty array [] if nothing concrete is found."
)


def build_triple_prompt(paper_id: str, paper_text: str) -> str:
    return (
        "<task>\nExtract concrete (subject, predicate, object) triples grounded in "
        "the paper. Each triple's subject and object must be specific entities "
        "(materials, devices, concepts, mechanisms, properties, conditions); "
        "predicates must be short verb phrases.\n</task>\n\n"
        "<good_examples>\n"
        '{"subject": "PEDOT:PSS film", "predicate": "exhibits", '
        '"object": "electrical conductivity 1000 S/cm", '
        '"evidence": "The PEDOT:PSS film exhibits an electrical conductivity of 1000 S/cm.", '
        '"confidence": 0.9}\n'
        '{"subject": "azobenzene", "predicate": "isomerizes under", '
        '"object": "365 nm UV light", '
        '"evidence": "Azobenzene isomerizes from trans to cis under 365 nm UV light.", '
        '"confidence": 0.95}\n'
        "</good_examples>\n\n"
        "<bad_examples>\n"
        '{"subject": "this paper", "predicate": "shows", "object": "interesting results"}\n'
        "(too vague, no entities — REJECT)\n"
        "</bad_examples>\n\n"
        f"<paper_id>{paper_id}</paper_id>\n\n"
        f"<paper_text>\n{paper_text}\n</paper_text>\n\n"
        "Return a JSON array of triples. Empty array [] if nothing concrete is found."
    )


def _extract_first_json_array(text: str) -> str | None:
    """Return the first balanced [...] substring in `text`, or None if none.

    Tolerates JSON-string brackets via quote/escape tracking. Used when the
    LLM emits leading commentary or concatenates multiple arrays.
    """
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_json_array(inner: str, *, context: str) -> list:
    """Parse an LLM response into a JSON array, tolerating fences and surrounding text.

    Raises ClaudeCallError with `context` prefix when neither direct nor
    bracket-scoped parse succeeds. ClaudeCallError is resolved at call time,
    so the class can live further down in the module.
    """
    text = inner.strip()
    if text.startswith("```"):
        try:
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        except IndexError as exc:
            raise ClaudeCallError(f"{context}: malformed code fence: {inner[:200]}") from exc
    try:
        parsed = _json.loads(text)
    except _json.JSONDecodeError:
        candidate = _extract_first_json_array(text)
        if candidate is None:
            raise ClaudeCallError(f"{context}: no JSON array found: {inner[:200]}") from None
        try:
            parsed = _json.loads(candidate)
        except _json.JSONDecodeError as exc:
            raise ClaudeCallError(
                f"{context}: bracket-scoped parse failed: {candidate[:200]}"
            ) from exc
    if not isinstance(parsed, list):
        raise ClaudeCallError(f"{context}: expected JSON array, got {type(parsed).__name__}")
    return parsed


def _call_triple_extractor(
    user_prompt: str,
    model_flag: str = DEFAULT_MODEL_FLAG,
    timeout_s: int = 120,
) -> list[dict]:
    try:
        inner = provider_call(
            user_prompt,
            system=_TRIPLE_EXTRACTOR_SYSTEM,
            model=model_flag,
            timeout_s=timeout_s,
        )
    except ProviderError as exc:
        raise ClaudeCallError(str(exc)) from exc
    return _parse_json_array(inner, context="triple result")


def _triple_dedup_key(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
    return (subject.lower().strip(), predicate.lower().strip(), obj.lower().strip())


def extract_triples_for_paper(
    paper_id: str,
    paper_text: str,
    existing_triples: list[SemanticTriple] | None = None,
    model_flag: str = DEFAULT_MODEL_FLAG,
) -> list[SemanticTriple]:
    """Extract (subject, predicate, object) triples from a paper's text via LLM.

    Triples are deduplicated against `existing_triples` by case-folded SPO key.
    Returns only NEW triples not already in the corpus.
    """
    seen = {_triple_dedup_key(t.subject, t.predicate, t.object) for t in (existing_triples or [])}
    raw = _call_triple_extractor(build_triple_prompt(paper_id, paper_text), model_flag=model_flag)
    new_triples: list[SemanticTriple] = []
    malformed = 0
    schema_invalid = 0
    for item in raw:
        if not isinstance(item, dict):
            malformed += 1
            continue
        try:
            subj = str(item["subject"])
            pred = str(item["predicate"])
            obj = str(item["object"])
            evidence = str(item["evidence"])
            conf = float(item.get("confidence", 0.7))
        except (KeyError, TypeError, ValueError):
            malformed += 1
            continue
        key = _triple_dedup_key(subj, pred, obj)
        if key in seen:
            continue
        seen.add(key)
        try:
            new_triples.append(
                SemanticTriple(
                    paper_id=paper_id,
                    triple_id=_stable_triple_id(paper_id, subj, pred, obj),
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    evidence=evidence,
                    confidence=conf,
                )
            )
        except ValueError:
            schema_invalid += 1
            continue
    # Loud failure when every triple was rejected — that's a schema regression
    # the user must see, not silently absorb as "no concrete content".
    if raw and not new_triples and (malformed or schema_invalid):
        raise ClaudeCallError(
            f"all {len(raw)} triples were rejected "
            f"(malformed={malformed}, schema_invalid={schema_invalid})"
        )
    if malformed or schema_invalid:
        print(
            f"{paper_id}: dropped {malformed}+{schema_invalid}/{len(raw)} triples",
            file=sys.stderr,
        )
    return new_triples


def build_prompt(
    paper_id: str,
    existing_facts: list[dict],
    paper_text: str,
) -> str:
    schema = (
        "Metric: a measured value WITH NUMBERS AND UNITS, e.g. 'tensile strength 12 MPa', "
        "'response time 8 ms', 'energy density 4.2 Wh/kg'. Must contain a numeric "
        "magnitude and a unit. targets: ['metric:<snake_case>'].\n"
        "Material: a specific substance/composition used as a building block, e.g. "
        "'PDMS elastomer matrix', 'polyaniline thin film', '4-cyano-4'-pentylbiphenyl (5CB)'. "
        "targets: ['material:<snake_case>'].\n"
        "Application: a use-case or device the work targets, e.g. 'soft robotic gripper', "
        "'wearable strain sensor', 'flexible energy harvester'. "
        "targets: ['application:<snake_case>'].\n"
        "Mechanism: a 'how it works' statement explaining a physical/chemical "
        "process, e.g. 'PEDOT:PSS conducts via doped polaron hopping'. Only emit "
        "if it introduces a NEW mechanism target id not in the corpus's existing 15 "
        "mechanism nodes (buckling, charge_injection, cross_linking, diffusion, "
        "electroosmosis, electrothermal_actuation, electrowetting, ferroelectric_switching, "
        "joule_heating, magnetic_actuation, etc.)."
    )
    existing_block = "\n".join(
        f"- [{f['fact_type']}] {f['text'][:120]}" for f in existing_facts[:30]
    )
    return (
        f"<schema>\n{schema}\n</schema>\n\n"
        f"<paper_id>{paper_id}</paper_id>\n\n"
        f"<existing_facts>\n{existing_block}\n</existing_facts>\n\n"
        f"<paper_text>\n{paper_text}\n</paper_text>\n\n"
        "Emit a JSON array of NEW facts of types Metric / Material / Application / "
        "Mechanism that are present in <paper_text> but not in <existing_facts>. "
        "Each item must have keys: fact_type, text, evidence, targets. "
        "targets must be prefixed ids like 'metric:<snake_case>' / "
        "'material:<snake_case>' / 'application:<snake_case>' / 'mechanism:<snake_case>'. "
        "Prefer snake_case ids that read naturally. Return JSON ARRAY ONLY — no markdown "
        "fences, no commentary. Empty array [] if nothing new applies."
    )


_EXTRACTOR_SYSTEM = (
    "You extract structured facts from polymer/materials science research papers. "
    "When asked, return JSON only — no prose, no markdown fences. "
    "Output JSON arrays where each item has keys: fact_type, text, evidence, targets. "
    "Allowed fact_type values: Metric, Material, Application, Mechanism. "
    "targets must be prefixed ids like 'metric:<snake_case>', 'material:<snake_case>', "
    "'application:<snake_case>', or 'mechanism:<snake_case>'. "
    "If no facts apply, return []."
)


class ClaudeCallError(RuntimeError):
    """Raised when LLM subprocess fails. Name kept for backwards-compat."""


def _call_claude_subprocess(
    user_prompt: str,
    model_flag: str = DEFAULT_MODEL_FLAG,
    timeout_s: int = 120,
) -> list[dict]:
    """Run the LLM extractor (claude/codex via LLM_PROVIDER env) and parse a JSON array."""
    try:
        inner = provider_call(
            user_prompt, system=_EXTRACTOR_SYSTEM, model=model_flag, timeout_s=timeout_s
        )
    except ProviderError as exc:
        raise ClaudeCallError(str(exc)) from exc
    return _parse_json_array(inner, context="inner result")


def _evidence_overlap(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def _is_duplicate(candidate: dict, existing: list[SemanticFact]) -> bool:
    for f in existing:
        if f.fact_type != candidate["fact_type"]:
            continue
        if _evidence_overlap(f.evidence, candidate["evidence"]) >= 0.8:
            return True
    return False


def _stable_fact_id(paper_id: str, candidate: dict) -> str:
    paper_short = paper_id.removeprefix("paper:")
    h = hashlib.sha1(
        (candidate["fact_type"] + "|" + candidate["text"]).encode("utf-8")
    ).hexdigest()[:10]
    return f"fact:{paper_short}:llm:{h}"


def _stable_heuristic_fact_id(paper_id: str, candidate: dict) -> str:
    paper_short = paper_id.removeprefix("paper:")
    h = hashlib.sha1(
        (candidate["fact_type"] + "|" + candidate["text"]).encode("utf-8")
    ).hexdigest()[:10]
    return f"fact:{paper_short}:heuristic:{h}"


_TARGETED_FACT_TYPES = {"Metric", "Material", "Application", "Mechanism"}
_METRIC_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|MPa|GPa|kPa|Pa|kV|mV|V|ms|s|Hz|kHz|MHz|J/g|Wh/kg|S/m|mS/cm|nm|um|μm|mm|cm)\b",
    re.IGNORECASE,
)
_MATERIAL_PATTERNS = (
    ("PDMS", "pdms"),
    ("PEDOT:PSS", "pedot_pss"),
    ("polyaniline", "polyaniline"),
    ("polyurethane", "polyurethane"),
    ("graphene", "graphene"),
    ("carbon nanotube", "carbon_nanotube"),
    ("hydrogel", "hydrogel"),
    ("ionic liquid", "ionic_liquid"),
    ("liquid crystal elastomer", "liquid_crystal_elastomer"),
    ("dielectric elastomer", "dielectric_elastomer"),
    ("elastomer", "elastomer"),
)
_APPLICATION_PATTERNS = (
    ("wearable strain sensor", "wearable_strain_sensor"),
    ("wearable sensor", "wearable_sensor"),
    ("soft robotic gripper", "soft_robotic_gripper"),
    ("soft robot", "soft_robot"),
    ("energy harvester", "energy_harvester"),
    ("microrobot", "microrobot"),
    ("display", "display"),
    ("sensor", "sensor"),
    ("actuator", "actuator"),
)


def _sentence_for_match(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
    right_candidates = [pos for pos in (text.find(".", end), text.find("\n", end)) if pos != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right].strip()


def _metric_target(evidence: str, value: str) -> str:
    lower = evidence.lower()
    unit = value.lower()
    if "strain" in lower or "%" in unit:
        return "metric:strain"
    if "stress" in lower or "pressure" in lower or "mpa" in unit or "gpa" in unit or "kpa" in unit:
        return "metric:stress"
    if "voltage" in lower or unit.endswith("v"):
        return "metric:voltage"
    if "response" in lower or "ms" in unit:
        return "metric:response_time"
    if "frequency" in lower or "hz" in unit:
        return "metric:frequency"
    if "energy" in lower or "j/g" in unit or "wh/kg" in unit:
        return "metric:energy_density"
    if "conduct" in lower or "s/m" in unit or "ms/cm" in unit:
        return "metric:conductivity"
    return "metric:measured_value"


def _heuristic_candidates(paper_text: str) -> list[dict]:
    candidates: list[dict] = []
    for match in _METRIC_RE.finditer(paper_text):
        evidence = _sentence_for_match(paper_text, match.start(), match.end())
        candidates.append(
            {
                "fact_type": "Metric",
                "text": f"Measured value {match.group(0)}",
                "evidence": evidence or match.group(0),
                "targets": [_metric_target(evidence, match.group(0))],
            }
        )
        break
    lower = paper_text.lower()
    for label, target in _MATERIAL_PATTERNS:
        index = lower.find(label.lower())
        if index == -1:
            continue
        evidence = _sentence_for_match(paper_text, index, index + len(label))
        candidates.append(
            {
                "fact_type": "Material",
                "text": label,
                "evidence": evidence or label,
                "targets": [f"material:{target}"],
            }
        )
        break
    for label, target in _APPLICATION_PATTERNS:
        index = lower.find(label.lower())
        if index == -1:
            continue
        evidence = _sentence_for_match(paper_text, index, index + len(label))
        candidates.append(
            {
                "fact_type": "Application",
                "text": label,
                "evidence": evidence or label,
                "targets": [f"application:{target}"],
            }
        )
        break
    return candidates


def heuristic_augment_facts_for_paper(
    paper_id: str,
    existing_facts: list[SemanticFact],
    paper_text: str,
) -> list[SemanticFact]:
    new_facts: list[SemanticFact] = []
    for candidate in _heuristic_candidates(paper_text):
        if _is_duplicate(candidate, existing_facts):
            continue
        try:
            new_facts.append(
                SemanticFact(
                    paper_id=paper_id,
                    fact_id=_stable_heuristic_fact_id(paper_id, candidate),
                    fact_type=candidate["fact_type"],
                    text=candidate["text"],
                    evidence=candidate["evidence"],
                    confidence=0.55,
                    targets=tuple(candidate["targets"]),
                )
            )
        except ValueError:
            continue
    return new_facts


def augment_facts_for_paper(
    paper_id: str,
    existing_facts: list[SemanticFact],
    paper_text: str,
    model_flag: str = DEFAULT_MODEL_FLAG,
) -> list[SemanticFact]:
    if model_flag == "heuristic":
        return heuristic_augment_facts_for_paper(paper_id, existing_facts, paper_text)
    prompt = build_prompt(
        paper_id=paper_id,
        existing_facts=[f.to_dict() for f in existing_facts],
        paper_text=paper_text,
    )
    raw = _call_claude_subprocess(prompt, model_flag=model_flag)
    new_facts: list[SemanticFact] = []
    for item in raw:
        if item.get("fact_type") not in _TARGETED_FACT_TYPES:
            continue
        if item["fact_type"] not in ALLOWED_FACT_TYPES:
            continue
        if _is_duplicate(item, existing_facts):
            continue
        try:
            new_facts.append(
                SemanticFact(
                    paper_id=paper_id,
                    fact_id=_stable_fact_id(paper_id, item),
                    fact_type=item["fact_type"],
                    text=item["text"],
                    evidence=item["evidence"],
                    confidence=0.7,
                    targets=tuple(item.get("targets", [])),
                )
            )
        except ValueError:
            continue
    return new_facts


def _load_paper_text(paper_id: str, graph_index: Path) -> str:
    pdf_text_path = graph_index / "pdf_text" / (paper_id.removeprefix("paper:") + ".txt")
    if pdf_text_path.exists():
        # take first 8000 chars (~ first 5-6 pages) to keep context tight
        return pdf_text_path.read_text(encoding="utf-8")[:8000]
    return ""


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json as _stdlib_json
    import sys
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from legacy.graphrag.wiki_semantic_schema import read_facts_jsonl

    parser = argparse.ArgumentParser(description="LLM-augment facts.jsonl with Mechanism/Metric")
    parser.add_argument("--graph-index", default=".graph_index", type=Path)
    parser.add_argument(
        "--paper-ids-file",
        type=Path,
        help="JSONL or text file with one paper_id per line. If absent, all papers.",
    )
    parser.add_argument("--limit-papers", type=int, default=0, help="0 = all matching papers")
    parser.add_argument(
        "--model-flag",
        default=DEFAULT_MODEL_FLAG,
        help="claude --model flag (e.g. 'sonnet', 'haiku', 'opus')",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Truncate facts_llm_augment.jsonl before starting (default: resume — skip papers already in file)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel LLM workers (default 1 = sequential; recommend 5 for ~5x speedup)",
    )
    parser.add_argument(
        "--triples",
        action="store_true",
        help="ALSO extract (subject, predicate, object) triples and append to "
        "<graph-index>/triples_llm_augment.jsonl. Doubles LLM calls per paper.",
    )
    parser.add_argument(
        "--yes-cost-confirm",
        action="store_true",
        help="Bypass the --triples cost-guard prompt for large corpora.",
    )
    args = parser.parse_args(argv)

    facts_path = args.graph_index / "facts.jsonl"
    facts_by_paper: dict[str, list[SemanticFact]] = {}
    for f in read_facts_jsonl(facts_path):
        facts_by_paper.setdefault(f.paper_id, []).append(f)

    out_path = args.graph_index / "facts_llm_augment.jsonl"
    triples_out_path = args.graph_index / "triples_llm_augment.jsonl"

    done_facts: set[str] = set()
    done_triples: set[str] = set()
    if args.restart:
        if out_path.exists():
            out_path.unlink()
        # Symmetric: --restart must also clear the triples file when --triples
        # is enabled, otherwise reruns silently accumulate duplicate triples.
        if args.triples and triples_out_path.exists():
            triples_out_path.unlink()
    else:
        if out_path.exists():
            try:
                done_facts = {f.paper_id for f in read_facts_jsonl(out_path)}
                print(
                    f"Resume: {len(done_facts)} paper(s) already in facts file, skipping fact step",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"WARN: cannot resume from {out_path}: {exc}; starting fresh", file=sys.stderr
                )
                done_facts = set()
        if args.triples and triples_out_path.exists():
            try:
                from legacy.graphrag.wiki_semantic_schema import read_triples_jsonl

                done_triples = {t.paper_id for t in read_triples_jsonl(triples_out_path)}
                print(
                    f"Resume: {len(done_triples)} paper(s) already in triples file, "
                    "skipping triple step",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"WARN: cannot resume from {triples_out_path}: {exc}; starting fresh",
                    file=sys.stderr,
                )
                done_triples = set()

    target_ids: list[str]
    if args.paper_ids_file:
        target_ids = [
            line.strip()
            for line in args.paper_ids_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        target_ids = list(facts_by_paper)
    if args.limit_papers:
        target_ids = target_ids[: args.limit_papers]

    # A paper is "fully done" only if every enabled output has it. Without
    # --triples, that's facts only. With --triples, BOTH must agree, otherwise
    # the first-time-enablement case skips every paper and emits zero triples.
    def _is_done(pid: str) -> bool:
        if pid not in done_facts:
            return False
        if args.triples and pid not in done_triples:
            return False
        return True

    target_ids = [pid for pid in target_ids if not _is_done(pid)]

    # Cost guard: --triples adds a second LLM call per paper. For a real
    # corpus (~600 papers) this is a non-trivial spend. Require explicit
    # confirmation when running across more than 50 papers, unless the
    # caller has acknowledged the cost via --yes-cost-confirm. Dry-run
    # skips the prompt because no API calls are made.
    COST_GUARD_THRESHOLD = 50
    if args.triples and not args.dry_run and not args.yes_cost_confirm:
        per_paper_pending = sum(1 for pid in target_ids if pid not in done_triples)
        if per_paper_pending > COST_GUARD_THRESHOLD:
            print(
                f"--triples will run on {per_paper_pending} papers, adding "
                f"one LLM call per paper on top of the existing fact pass. "
                f"Threshold: {COST_GUARD_THRESHOLD}. Pass --yes-cost-confirm to "
                f"acknowledge, or use --limit-papers to scope the run.",
                file=sys.stderr,
            )
            return

    # Pre-load existing triples once so per-paper extraction can dedup against
    # the global corpus, not just the current paper's response.
    existing_triples_corpus: list[SemanticTriple] = []
    if args.triples and triples_out_path.exists():
        try:
            from legacy.graphrag.wiki_semantic_schema import read_triples_jsonl

            existing_triples_corpus = read_triples_jsonl(triples_out_path)
        except Exception:
            existing_triples_corpus = []

    def _augment_one(
        pid: str,
    ) -> tuple[str, list[SemanticFact], list[SemanticTriple], Exception | None]:
        existing = facts_by_paper.get(pid, [])
        text = _load_paper_text(pid, args.graph_index)
        if not text:
            text = "\n\n".join(f.evidence for f in existing[:60])

        # If facts were already extracted for this paper, don't re-call the
        # fact-extraction LLM. Triples may still need extraction.
        new_facts: list[SemanticFact] = []
        if pid not in done_facts:
            for attempt in range(3):
                try:
                    new_facts = augment_facts_for_paper(
                        pid, existing, text, model_flag=args.model_flag
                    )
                    break
                except ClaudeCallError as exc:
                    if attempt == 2:
                        return pid, [], [], exc
                    time.sleep(2**attempt)

        new_triples: list[SemanticTriple] = []
        if args.triples and pid not in done_triples:
            for attempt in range(3):
                try:
                    new_triples = extract_triples_for_paper(
                        pid,
                        text,
                        existing_triples=existing_triples_corpus,
                        model_flag=args.model_flag,
                    )
                    break
                except ClaudeCallError as exc:
                    if attempt == 2:
                        # Don't lose fact results when triples fail.
                        print(
                            f"{pid}: triple extraction ERROR after retries: {exc}",
                            file=sys.stderr,
                        )
                        break
                    time.sleep(2**attempt)

        return pid, new_facts, new_triples, None

    write_lock = threading.Lock()
    new_total = 0
    new_triples_total = 0
    failures = 0

    def _handle(
        pid: str,
        emitted: list[SemanticFact],
        emitted_triples: list[SemanticTriple],
        exc: Exception | None,
    ) -> None:
        nonlocal new_total, new_triples_total, failures
        if exc:
            failures += 1
            print(f"{pid}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            return
        new_total += len(emitted)
        new_triples_total += len(emitted_triples)
        suffix = f", +{len(emitted_triples)} triples" if args.triples else ""
        print(f"{pid}: +{len(emitted)} facts{suffix}", file=sys.stderr)
        if args.dry_run:
            return
        with write_lock:
            if emitted:
                with out_path.open("a", encoding="utf-8") as handle:
                    for fact in emitted:
                        handle.write(_stdlib_json.dumps(fact.to_dict(), sort_keys=True) + "\n")
            if emitted_triples:
                with triples_out_path.open("a", encoding="utf-8") as handle:
                    for triple in emitted_triples:
                        handle.write(_stdlib_json.dumps(triple.to_dict(), sort_keys=True) + "\n")

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_augment_one, pid): pid for pid in target_ids}
            for future in as_completed(futures):
                _handle(*future.result())
    else:
        for pid in target_ids:
            _handle(*_augment_one(pid))

    if args.dry_run:
        triples_msg = f", {new_triples_total} triples" if args.triples else ""
        print(f"DRY RUN: would emit {new_total} new facts{triples_msg}", file=sys.stderr)
        return

    triples_msg = (
        f"; appended {new_triples_total} triples to {triples_out_path}" if args.triples else ""
    )
    print(
        f"Appended {new_total} new facts to {out_path} "
        f"({len(done_facts)} resumed, {failures} failures){triples_msg}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
