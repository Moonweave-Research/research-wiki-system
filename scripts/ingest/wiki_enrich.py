import anthropic
import argparse
import re
import time
from pathlib import Path

VAULT_DIR = Path(__file__).parent.parent
RESEARCH_DIR = VAULT_DIR / "wiki" / "research"
MODEL = "claude-haiku-4-5"
ENRICH_MARKER = "## 핵심 메커니즘"

SYSTEM_PROMPT = """\
You are a research assistant for a PhD student in materials science. The student's research \
centers on electroactive soft actuators and related sensing systems. Their primary domains:

**Dielectric Elastomer Actuators (DEA)**
Soft actuators that deform under high electric fields via Maxwell stress (electrostatic pressure \
between compliant electrodes compresses the dielectric film, generating in-plane strain). Key \
figures of merit: actuation strain (%), energy density (J/kg or J/cm³), dielectric permittivity \
(ε'), dielectric loss (tan δ), breakdown field (MV/m), Young's modulus. Common dielectrics: VHB \
acrylic tape, PDMS, polyurethane, silicone, PVDF. Electrodes: carbon grease, carbon black, CNT \
networks, PEDOT:PSS, ionic hydrogels.

**Liquid Crystal Elastomers (LCE)**
Stimuli-responsive polymer networks that contract or bend upon the LC order–disorder phase \
transition. Stimuli: heat (Joule heating, photothermal via gold nanoparticles or carbon black), \
light (azobenzene or spiropyran mesogens, 365 nm UV for trans→cis, 450 nm visible for cis→trans), \
humidity, solvent swelling. Key parameters: mesogen alignment direction, crosslink density, \
actuation stroke (%), response time (s), transition temperature (T_NI). Synthesis routes: \
thiol-ene two-stage crosslinking, aza-Michael addition, main-chain vs. side-chain LCE.

**Ionic Polymer-Metal Composites (IPMC)**
Bending actuators operated at low voltage (1–5 V) via asymmetric ion/water migration through \
an ionomer membrane (Nafion, Flemion) plated with platinum or gold electrodes. Key metrics: \
tip displacement (mm), blocking force (mN), water uptake, ion exchange capacity.

**Adjacent domains the student monitors:** soft robotics (pneumatic, hydraulic, cable-driven, \
origami/kirigami), piezoelectric and triboelectric nanogenerators, capacitive/resistive/optical \
flexible sensors, 2D materials (graphene, MoS₂, h-BN) for flexible electronics, in-sensor \
computing, hydrogel-based actuators and ionotronics, photonic sensors, chiral materials, \
van der Waals heterostructures, neuromorphic devices.

**Fabrication methods encountered:** electrospinning, 3D/4D printing, soft lithography, \
dip coating, spin coating, physical/chemical vapor deposition, screen printing, inkjet printing, \
atomic layer deposition (ALD), magnetron sputtering, laser ablation, self-assembly, \
photolithography, nanoimprint lithography.

**Key performance metrics to recognize across domains:**
- Actuators: strain (%), stress (kPa or MPa), energy density (J/kg), power density (W/kg), \
  bandwidth (Hz), cycle lifetime (cycles to failure), response time (ms or s)
- Sensors: sensitivity (output/input unit), detection limit, linear range, gauge factor (GF), \
  response/recovery time, selectivity, stability (drift over cycles)
- Electrical: sheet resistance (Ω/sq), transmittance (%), carrier mobility (cm²/V·s), \
  on/off ratio, subthreshold swing
- Optical: reflectance, absorbance, photoluminescence quantum yield, dissymmetry factor (g)

**How to handle papers far from the core domain:**
If a paper is about, e.g., in-sensor computing with 2D heterostructures and liquid crystals, \
identify the one mechanistic or fabrication insight that could transfer: e.g., how the chiral \
LC layer selectively reflects circularly polarized light, and whether that principle could apply \
to photoresponsive LCE design. Keep the "내 연구와의 연결" section honest — if the connection \
is weak, say so briefly rather than forcing relevance.

---

Given a paper's title, authors, year, and abstract, generate a structured analysis in Korean. \
Technical terms, material names, chemical abbreviations, and proper nouns must stay in their \
original English/Latin form (e.g., write "DEA", "LCE", "Maxwell stress", "azobenzene", \
"PEDOT:PSS", not Korean transliterations).

Output exactly these 5 sections, beginning immediately with "## 핵심 메커니즘" — no preamble, \
no "Here is the analysis:", no closing remarks.

## 핵심 메커니즘
What physical, chemical, or biological mechanism drives the paper's core result? \
Write 2–4 concise bullets. Prefer mechanistic specificity over vague summary. \
Good example: "Maxwell 압력으로 VHB 필름이 두께 방향으로 압축되어 면내 변형률 발생". \
Bad example: "전기 에너지를 기계 에너지로 변환하는 메커니즘 사용".

## 주요 기여
What does this paper add beyond prior work? State each advance concretely; include numbers \
where the abstract provides them. Write 2–4 bullets. \
Good example: "polyurethane 유전층으로 기존 VHB 대비 구동 전압 40% 감소 (3 kV → 1.8 kV)". \
Bad example: "새로운 소재로 성능을 크게 향상시킴".

## 방법론
Fabrication, characterization, and measurement methods. Include key process parameters \
(rpm, temperature, thickness, voltage) where available. Write 2–4 bullets.

## 장점·한계
Use "**장점**:" and "**한계**:" prefixes. Strengths should cite demonstrated metrics; \
limitations may be acknowledged in the paper or apparent from the scope. Write 2–3 bullets.

## 내 연구와의 연결
Specific relevance to DEA / LCE / IPMC / soft actuator or sensor research. If the paper is \
outside the core domain, identify the one technique or result that transfers. Write 1–3 bullets.\
"""


def extract_paper_info(md_path: Path) -> dict[str, str] | None:
    text = md_path.read_text(encoding="utf-8")
    abstract_match = re.search(r"## Abstract\n\n?(.+?)(?:\n\n##|\Z)", text, re.DOTALL)
    if not abstract_match:
        return None
    abstract = abstract_match.group(1).strip()
    if not abstract:
        return None

    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else md_path.stem

    authors_match = re.search(r"\*\*Authors:\*\* (.+)$", text, re.MULTILINE)
    authors = authors_match.group(1).strip() if authors_match else ""

    year_match = re.search(r"\*\*Year:\*\* (\d{4})", text)
    year = year_match.group(1) if year_match else ""

    return {"title": title, "authors": authors, "year": year, "abstract": abstract}


def already_enriched(md_path: Path) -> bool:
    return ENRICH_MARKER in md_path.read_text(encoding="utf-8")


def strip_existing_analysis(text: str) -> str:
    idx = text.find(ENRICH_MARKER)
    if idx == -1:
        return text
    return text[:idx].rstrip()


def call_api(client: anthropic.Anthropic, info: dict[str, str]) -> str:
    user_content = (
        f"Title: {info['title']}\n"
        f"Authors: {info['authors']}\n"
        f"Year: {info['year']}\n\n"
        f"Abstract:\n{info['abstract']}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text.strip()


def enrich_file(
    md_path: Path,
    client: anthropic.Anthropic,
    overwrite: bool,
    dry_run: bool,
) -> str:
    if already_enriched(md_path) and not overwrite:
        return "skip:already-enriched"

    info = extract_paper_info(md_path)
    if info is None:
        return "skip:no-abstract"

    if dry_run:
        return "would-enrich"

    analysis = call_api(client, info)

    text = md_path.read_text(encoding="utf-8")
    text = strip_existing_analysis(text)
    text = text.rstrip() + "\n\n" + analysis + "\n"
    md_path.write_text(text, encoding="utf-8")
    return "enriched"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich research wiki notes with Claude-generated analysis."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("file", nargs="?", type=Path, help="Single .md file to enrich")
    group.add_argument("--all", action="store_true", help="Enrich all files in wiki/research/")
    parser.add_argument("--overwrite", action="store_true", help="Re-generate existing analysis")
    parser.add_argument("--dry-run", action="store_true", help="Count without calling the API")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    if not args.file and not args.all:
        parser.error("Specify a file or --all")

    if args.all:
        targets = sorted(RESEARCH_DIR.glob("*.md"))
    else:
        targets = [args.file]

    client = anthropic.Anthropic() if not args.dry_run else None  # type: ignore[assignment]

    counts: dict[str, int] = {}
    for md_path in targets:
        result = enrich_file(md_path, client, args.overwrite, args.dry_run)
        counts[result] = counts.get(result, 0) + 1
        label = result.upper()
        print(f"[{label}] {md_path.name}")
        if result == "enriched" and args.delay > 0:
            time.sleep(args.delay)

    print(
        f"\nDone: {counts.get('enriched', 0)} enriched, "
        f"{counts.get('would-enrich', 0)} would-enrich, "
        f"{counts.get('skip:already-enriched', 0)} already-enriched, "
        f"{counts.get('skip:no-abstract', 0)} no-abstract"
    )


if __name__ == "__main__":
    main()
