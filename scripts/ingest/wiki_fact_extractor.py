import argparse
import hashlib
import re
from pathlib import Path

try:
    from legacy.graphrag.wiki_graph import (
        iter_markdown_files,
        parse_frontmatter,
        parse_note,
        resolve_wiki_root,
    )
    from legacy.graphrag.wiki_semantic_schema import SemanticFact, write_facts_jsonl
except ModuleNotFoundError:
    from wiki_graph import (  # type: ignore[no-redef]
        iter_markdown_files,
        parse_frontmatter,
        parse_note,
        resolve_wiki_root,
    )
    from wiki_semantic_schema import SemanticFact, write_facts_jsonl  # type: ignore[no-redef]


SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")

SECTION_FACT_TYPES = {
    "Abstract": "Claim",
    "핵심 메커니즘": "Mechanism",
    "주요 기여": "Claim",
    "방법론": "Method",
}

SECTION_CONFIDENCE = {
    "Abstract": 0.72,
    "핵심 메커니즘": 0.78,
    "주요 기여": 0.75,
    "방법론": 0.75,
    "장점·한계": 0.64,
}

TARGET_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("metric:voltage", (r"\blow[- ]voltage\b", r"\bvoltage\b", r"\b\d+(?:\.\d+)?\s?v\b")),
    ("metric:force", (r"\bforce\b", r"\b\d+(?:\.\d+)?\s?mn\b")),
    ("metric:curvature", (r"\bcurvature\b", r"\bcm[−-]1\b")),
    ("metric:strain", (r"\bstrain\b", r"\b\d+(?:\.\d+)?\s?%")),
    ("concept:Soft_Actuator", (r"\bsoft actuators?\b",)),
    (
        "concept:Charge_Trapping",
        (r"\bcharge trapping\b", r"\bcharge[- ]trapping\b"),
    ),
    (
        "concept:Dielectric_Elastomer_Actuator",
        (r"\bdielectric elastomer actuators?\b", r"\bDEAs?\b"),
    ),
    (
        "mechanism:electrothermal_actuation",
        (
            r"\belectrothermal\b",
            r"\bresistive heating\b",
            r"\bthermal expansion\b",
            r"\bthermal mismatch\b",
        ),
    ),
    (
        "method:electrohydrodynamic_printing",
        (r"\belectrohydrodynamic\b", r"\behd printing\b", r"\behd\b"),
    ),
    # --- Metrics (Batch A) ---
    ("metric:electric_field", (r"\belectric fields?\b", r"\bE[- ]field\b")),
    (
        "metric:dielectric_constant",
        (r"\bdielectric constant\b", r"\bdielectric permittivity\b", r"\brelative permittivity\b"),
    ),
    ("metric:breakdown_strength", (r"\bbreakdown strength\b", r"\bdielectric strength\b")),
    ("metric:energy_density", (r"\benergy density\b",)),
    (
        "metric:transmittance",
        (r"\boptical transmittance\b", r"\blight transmittance\b", r"\btransparency\b"),
    ),
    ("metric:response_time", (r"\bresponse time\b", r"\bswitching time\b", r"\bactuation speed\b")),
    (
        "metric:power_consumption",
        (r"\bpower consumption\b", r"\bpower density\b", r"\bpower output\b"),
    ),
    ("metric:current", (r"\bcurrent density\b", r"\bleakage current\b")),
    ("metric:work_density", (r"\bwork density\b", r"\bspecific work\b")),
    (
        "metric:temperature",
        (r"\btransition temperature\b", r"\bglass transition\b", r"\bTg\b", r"\bmelting point\b"),
    ),
    (
        "metric:conductivity",
        (r"\bconductivity\b", r"\belectrical conductivity\b", r"\bionic conductivity\b"),
    ),
    # --- Mechanisms (Batch A) ---
    ("mechanism:cross_linking", (r"\bcross[- ]link\w*\b", r"\bcrosslink\w*\b")),
    ("mechanism:pull_in_instability", (r"\bpull[- ]in\b", r"\belectromechanical instability\b")),
    ("mechanism:electrowetting", (r"\belectrowetting\b", r"\bEWOD\b")),
    ("mechanism:self_assembly", (r"\bself[- ]assembl\w*\b", r"\bself[- ]organiz\w*\b")),
    ("mechanism:swelling", (r"\bswelling\b", r"\bosmotic pressure\b", r"\bhydration\b")),
    (
        "mechanism:magnetic_actuation",
        (r"\bmagnetic field\b", r"\bmagnetostrictive\b", r"\bmagneto[- ]responsive\b"),
    ),
    ("mechanism:joule_heating", (r"\bjoule heating\b", r"\bohmic heating\b")),
    ("mechanism:charge_injection", (r"\bcharge injection\b", r"\bspace charge injection\b")),
    (
        "mechanism:ferroelectric_switching",
        (r"\bferroelectric switching\b", r"\bpolarization switching\b", r"\bdomain switching\b"),
    ),
    ("mechanism:electroosmosis", (r"\belectroosmosis\b", r"\belectro[- ]osmotic\b")),
    # --- Batch B (narrowed regex to avoid bare-word noise) ---
    (
        "metric:frequency",
        (
            r"\bresonant frequency\b",
            r"\boperating frequency\b",
            r"\bswitching frequency\b",
            r"\bcutoff frequency\b",
        ),
    ),
    (
        "metric:pressure",
        (
            r"\b\d+\.?\d*\s?(?:kPa|MPa|GPa)\b",
            r"\bactuation pressure\b",
            r"\bhydrostatic pressure\b",
        ),
    ),
    (
        "metric:stress",
        (
            r"\btensile stress\b",
            r"\bmechanical stress\b",
            r"\belectromechanical stress\b",
            r"\btensile strength\b",
            r"\byield strength\b",
        ),
    ),
    (
        "metric:young_modulus",
        (
            r"\bYoung'?s? modulus\b",
            r"\belastic modulus\b",
            r"\bflexural modulus\b",
            r"\btensile modulus\b",
        ),
    ),
    (
        "mechanism:diffusion",
        (
            r"\bmass diffusion\b",
            r"\bion diffusion\b",
            r"\bcarrier diffusion\b",
            r"\bthermal diffusion\b",
            r"\bmass transport\b",
            r"\bion transport\b",
        ),
    ),
    (
        "mechanism:phase_transition",
        (
            r"\bphase transition\b",
            r"\bnematic[- ]isotropic\b",
            r"\bsmectic[- ]nematic\b",
            r"\bfirst[- ]order phase\b",
        ),
    ),
    (
        "mechanism:wettability",
        (
            r"\bwettability\b",
            r"\bcontact angle\b",
            r"\bhydrophobic surface\b",
            r"\bhydrophilic surface\b",
        ),
    ),
    (
        "mechanism:buckling",
        (r"\bbuckling\b", r"\bwrinkling\b", r"\bsurface buckling\b"),
    ),
]

_ACTIVE_PATTERNS: list[tuple[str, tuple[str, ...]]] | None = None


def build_concept_patterns(wiki_root: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Scan wiki/concepts/*.md, build augmenting patterns from frontmatter aliases + stems.

    Each concept .md becomes a (target_id, patterns) tuple where target_id is
    f"concept:{stem}". ASCII terms get \b boundaries; non-ASCII (Korean) terms are
    matched literally and only included when length >= 3 to avoid substring noise.
    """
    concepts_dir = wiki_root / "concepts"
    if not concepts_dir.is_dir():
        return []

    patterns: list[tuple[str, tuple[str, ...]]] = []
    for path in sorted(concepts_dir.glob("*.md")):
        stem = path.stem
        target_id = f"concept:{stem}"
        text = path.read_text(encoding="utf-8")
        frontmatter, _ = parse_frontmatter(text)
        aliases = frontmatter.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]

        terms = {stem.replace("_", " ")}
        for alias in aliases:
            term = str(alias).strip()
            if len(term) >= 3:
                terms.add(term)

        regexes: list[str] = []
        for term in sorted(terms, key=len, reverse=True):
            escaped = re.escape(term)
            if term.isascii():
                regexes.append(rf"\b{escaped}s?\b")
            else:
                regexes.append(escaped)

        if regexes:
            patterns.append((target_id, tuple(regexes)))
    return patterns


def _merge_patterns(
    base: list[tuple[str, tuple[str, ...]]],
    augment: list[tuple[str, tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...]]]:
    """Union patterns by target_id; preserve base order, append new target_ids."""
    merged: dict[str, list[str]] = {tid: list(pats) for tid, pats in base}
    order = [tid for tid, _ in base]
    for target_id, pats in augment:
        if target_id in merged:
            for pat in pats:
                if pat not in merged[target_id]:
                    merged[target_id].append(pat)
        else:
            merged[target_id] = list(pats)
            order.append(target_id)
    return [(tid, tuple(merged[tid])) for tid in order]


def load_active_patterns(wiki_root: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Build and cache the active pattern list for this wiki_root."""
    global _ACTIVE_PATTERNS
    augment = build_concept_patterns(wiki_root)
    _ACTIVE_PATTERNS = _merge_patterns(TARGET_PATTERNS, augment)
    return _ACTIVE_PATTERNS


def extract_section_map(body: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1).strip()] = body[start:end].strip()
    return sections


_NOISE_URL = re.compile(
    r"https?://|doi\.org|ip:\s*\d+|org/content/terms|download to ip", re.IGNORECASE
)
_NOISE_WATERMARK = re.compile(
    r"downloaded via|downloaded from|all rights reserved|copyright\s*©", re.IGNORECASE
)
_NOISE_NUMBER = re.compile(r"^\s*\d+\.?\s*$|^\s*[A-Z]\.\s*$")
_LLM_REJECTION = re.compile(
    r"I appreciate you testing|"
    r"cannot summarize|"
    r"outside my scope|"
    r"appreciate your interest|"
    r"provide summaries",
    re.I,
)
_NOISE_LLM_REJECTION = re.compile(
    r"i appreciate|cannot summarize|outside my scope|"
    r"as an ai|i'm unable to|cannot provide|unable to provide|"
    r"apologies|i don't have|i cannot|not appropriate|"
    r"vaccine.*trial|suspension.*fund",
    re.I,
)


def _is_noise_unit(text: str) -> bool:
    if len(text) < 10:
        return True
    if _NOISE_NUMBER.match(text):
        return True
    if _NOISE_URL.search(text):
        return True
    if _NOISE_WATERMARK.search(text):
        return True
    if _LLM_REJECTION.search(text):
        return True
    if _NOISE_LLM_REJECTION.search(text):
        return True
    return False


def clean_unit(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def split_fact_units(section_text: str) -> list[str]:
    bullet_units = []
    for line in section_text.splitlines():
        match = BULLET_RE.match(line)
        if match:
            unit = clean_unit(match.group(1))
            if unit and not _is_noise_unit(unit):
                bullet_units.append(unit)
    if bullet_units:
        return bullet_units

    units = []
    start = 0
    text = section_text.strip()
    for index, char in enumerate(text):
        if char not in ".!?":
            continue
        previous_char = text[index - 1] if index > 0 else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if previous_char.isdigit() and next_char.isdigit():
            continue
        unit = clean_unit(text[start : index + 1])
        if unit and not _is_noise_unit(unit):
            units.append(unit)
        start = index + 1
    tail = clean_unit(text[start:])
    if tail and not _is_noise_unit(tail):
        units.append(tail)
    return units


def infer_targets(text: str) -> list[str]:
    active = _ACTIVE_PATTERNS if _ACTIVE_PATTERNS is not None else TARGET_PATTERNS
    targets = []
    for target, patterns in active:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            targets.append(target)
    return targets


def target_score(targets: list[str]) -> int:
    score = 0
    for target in targets:
        if target.startswith("metric:"):
            score += 3
        else:
            score += 1
    return score


def section_fact_type(section_name: str, unit: str) -> str:
    if section_name == "장점·한계":
        lower = unit.lower()
        if "한계" in unit or "limitation" in lower:
            return "Limitation"
        return ""
    return SECTION_FACT_TYPES.get(section_name, "")


def fact_id_for(
    paper_id: str, fact_type: str, section_name: str, text: str, occurrence: int
) -> str:
    stem = paper_id.removeprefix("paper:")
    normalized = re.sub(r"\s+", " ", f"{section_name} {text} {occurrence}".strip().lower())
    digest = hashlib.blake2s(normalized.encode("utf-8"), digest_size=8).hexdigest()
    return f"fact:{stem}:{fact_type.lower()}:{digest}"


def extract_facts_from_note(
    path: Path,
    wiki_root: Path,
    limit_per_paper: int = 15,
) -> list[SemanticFact]:
    note = parse_note(path, wiki_root)
    if note.kind != "Paper":
        return []
    if note.status == "irrelevant":
        return []

    sections = extract_section_map(note.body)
    facts = []
    occurrences: dict[tuple[str, str, str], int] = {}
    for section_name in ["Abstract", "핵심 메커니즘", "주요 기여", "방법론", "장점·한계"]:
        candidates = []
        for order, unit in enumerate(split_fact_units(sections.get(section_name, ""))):
            fact_type = section_fact_type(section_name, unit)
            if not fact_type:
                continue
            targets = infer_targets(unit)
            candidates.append((-target_score(targets), order, unit, fact_type, targets))
        for _, _, unit, fact_type, targets in sorted(candidates):
            occurrence_key = (section_name, fact_type, re.sub(r"\s+", " ", unit.strip().lower()))
            occurrences[occurrence_key] = occurrences.get(occurrence_key, 0) + 1
            facts.append(
                SemanticFact(
                    paper_id=note.node_id,
                    fact_id=fact_id_for(
                        note.node_id,
                        fact_type,
                        section_name,
                        unit,
                        occurrences[occurrence_key],
                    ),
                    fact_type=fact_type,
                    text=unit,
                    evidence=f"{section_name}: {unit}",
                    confidence=SECTION_CONFIDENCE[section_name],
                    targets=targets,
                )
            )
            if len(facts) >= limit_per_paper:
                return facts
    return facts


def extract_facts(
    wiki_dir: Path,
    paper_ids: list[str] | None = None,
    limit_per_paper: int = 15,
) -> list[SemanticFact]:
    wiki_root = resolve_wiki_root(wiki_dir)
    load_active_patterns(wiki_root)
    selected = set(paper_ids or [])
    facts = []
    for path in iter_markdown_files(wiki_root):
        note = parse_note(path, wiki_root)
        if note.kind != "Paper":
            continue
        if selected and note.node_id not in selected:
            continue
        facts.extend(extract_facts_from_note(path, wiki_root, limit_per_paper=limit_per_paper))
    return facts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract draft semantic facts from wiki notes.")
    parser.add_argument("--wiki-dir", default="wiki")
    parser.add_argument("--output", default=".graph_index/facts.jsonl")
    parser.add_argument("--paper", action="append", default=[])
    parser.add_argument("--limit-per-paper", type=int, default=15)
    args = parser.parse_args(argv)

    facts = extract_facts(
        Path(args.wiki_dir),
        paper_ids=args.paper or None,
        limit_per_paper=args.limit_per_paper,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_facts_jsonl(output, facts)
    print(f"facts_written: {output}")
    print(f"facts: {len(facts)}")


if __name__ == "__main__":
    main()
