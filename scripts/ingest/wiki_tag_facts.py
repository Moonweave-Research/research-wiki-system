"""LLM-based fact tagger.

Reads untagged facts from .graph_index/facts.jsonl, asks claude -p to assign
controlled-vocabulary tags from wiki/concepts/_tagging_vocabulary.json, and
appends results to wiki/_fact_tags_overlay.jsonl (sidecar overlay).

Resumable: skips facts already in the overlay file.

Recursion-guarded: refuses to nest if invoked from inside a claude -p call.
"""

import os
import sys

if os.environ.get("CLAUDE_FACT_TAGGING_CHILD") == "1":
    sys.stderr.write("recursion guard: refusing to nest fact-tagging subprocess\n")
    sys.exit(0)

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from scripts.core.wiki_llm_provider import provider_call

DEFAULT_FACTS_PATH = Path(".graph_index/facts.jsonl")
DEFAULT_VOCAB_PATH = Path("wiki/concepts/_tagging_vocabulary.json")
DEFAULT_OVERLAY_PATH = Path("wiki/_fact_tags_overlay.jsonl")


def load_vocabulary(path: Path) -> tuple[list[str], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids: list[str] = []
    lines: list[str] = []
    lines.append("CONCEPTS (use the most specific applicable):")
    for c in data["concepts"]:
        ids.append(c["id"])
        aliases = c.get("aliases") or []
        alias_str = f" [{', '.join(aliases[:5])}]" if aliases else ""
        lines.append(f"  {c['id']}{alias_str}")
    lines.append("")
    lines.append("METRICS (quantitative measurements):")
    for m in data["metrics"]:
        ids.append(m["id"])
        lines.append(f"  {m['id']}")
    lines.append("")
    lines.append("MECHANISMS (physical processes):")
    for m in data["mechanisms"]:
        ids.append(m["id"])
        lines.append(f"  {m['id']}")
    return ids, "\n".join(lines)


def load_overlay_seen(overlay_path: Path) -> set[str]:
    if not overlay_path.exists():
        return set()
    seen: set[str] = set()
    with overlay_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = rec.get("fact_id")
            if fid:
                seen.add(fid)
    return seen


def select_facts_to_tag(
    facts_path: Path,
    seen: set[str],
    only_ids: set[str] | None,
    only_untagged: bool,
) -> list[dict]:
    selected: list[dict] = []
    with facts_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fid = rec.get("fact_id")
            if not fid or fid in seen:
                continue
            if only_ids is not None:
                if fid not in only_ids:
                    continue
            elif only_untagged:
                if rec.get("targets"):
                    continue
            selected.append(rec)
    return selected


def build_prompt(vocabulary_block: str, batch: list[dict]) -> str:
    fact_lines = []
    for f in batch:
        evidence = (f.get("evidence") or "").replace("\n", " ").strip()
        if len(evidence) > 350:
            evidence = evidence[:350] + "..."
        fact_lines.append(f'  - fact_id: "{f["fact_id"]}"\n    evidence: "{evidence}"')
    facts_block = "\n".join(fact_lines)

    return (
        "You are a materials-science fact tagger. For each fact below, assign tags "
        "from the controlled vocabulary.\n\n"
        f"VOCABULARY:\n{vocabulary_block}\n\n"
        f"FACTS TO TAG ({len(batch)}):\n{facts_block}\n\n"
        "RULES:\n"
        "1. Output ONLY IDs from the vocabulary above. Never invent new IDs.\n"
        "2. Tag based on EVIDENCE TEXT ONLY. Do not infer from fact_id or paper context.\n"
        "3. Pick the MOST SPECIFIC applicable concept. The IS_A hierarchy will surface "
        "ancestors automatically. Do NOT tag both Dielectric_Elastomer_Actuator and Soft_Actuator.\n"
        "4. If evidence mentions a quantitative measurement (voltage, strain, etc.), "
        "include the metric tag.\n"
        "5. Empty list [] is the correct answer when evidence is generic introduction, "
        "pure citation text, copyright notice, generic methodology comment, or otherwise "
        "lacks vocabulary-mappable content. Do NOT force-fit.\n"
        "6. Negation ('data 부재', 'no measurement') means the metric was NOT measured — "
        "do not tag it.\n"
        "7. Maximum 4 tags per fact (typical 0-2).\n\n"
        "OUTPUT FORMAT — JSON array, one object per fact, no preamble, no markdown fences:\n"
        "[\n"
        '  {"fact_id": "fact:xxx", "targets": ["concept:Dielectric_Elastomer_Actuator", "metric:voltage"]},\n'
        '  {"fact_id": "fact:yyy", "targets": []},\n'
        "  ...\n"
        "]\n\n"
        "Return tags for ALL facts in order. Do not skip any."
    )


def call_claude(prompt: str, timeout: int = 120) -> str:
    """Run the configured LLM provider for fact-tag output.

    Function name kept for backwards-compat; underlying provider is selected
    by the LLM_PROVIDER env var (claude default, codex alternate).
    """
    output = provider_call(prompt, timeout_s=timeout)
    if output.startswith("```"):
        lines = [line for line in output.splitlines() if not line.startswith("```")]
        output = "\n".join(lines).strip()
    return output


def parse_response(output: str, valid_ids: set[str], expected_fact_ids: list[str]) -> list[dict]:
    """Parse claude response. Validate IDs against vocab. Return list of {fact_id, targets, rejected}."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        # Try to find JSON array in output
        match = re.search(r"\[\s*\{.*?\}\s*\]", output, re.DOTALL)
        if not match:
            raise RuntimeError(f"no JSON array found in output:\n{output[:500]}")
        data = json.loads(match.group(0))

    if not isinstance(data, list):
        raise RuntimeError(f"expected JSON array, got {type(data).__name__}")

    by_id = {item.get("fact_id"): item for item in data if isinstance(item, dict)}
    results: list[dict] = []
    for fid in expected_fact_ids:
        item = by_id.get(fid)
        if item is None:
            results.append({"fact_id": fid, "targets": [], "rejected": [], "missing": True})
            continue
        raw_targets = item.get("targets") or []
        accepted = [t for t in raw_targets if t in valid_ids]
        rejected = [t for t in raw_targets if t not in valid_ids]
        results.append(
            {"fact_id": fid, "targets": accepted, "rejected": rejected, "missing": False}
        )
    return results


def append_overlay(overlay_path: Path, results: list[dict], source: str) -> None:
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    with overlay_path.open("a", encoding="utf-8") as fh:
        for r in results:
            rec = {
                "fact_id": r["fact_id"],
                "targets": r["targets"],
                "source": source,
            }
            if r.get("rejected"):
                rec["rejected"] = r["rejected"]
            if r.get("missing"):
                rec["missing"] = True
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tag untagged facts via claude -p.")
    parser.add_argument("--facts", default=str(DEFAULT_FACTS_PATH))
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH))
    parser.add_argument("--overlay", default=str(DEFAULT_OVERLAY_PATH))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap on number of batches (default: unlimited)",
    )
    parser.add_argument(
        "--only-fact-ids-file",
        default=None,
        help="Path to file with fact_ids (one per line). Restricts processing to these.",
    )
    parser.add_argument(
        "--source-tag", default="claude-p", help="Source label written into overlay records"
    )
    parser.add_argument(
        "--rate-limit-sec", type=float, default=0.0, help="Sleep between batches (seconds)"
    )
    args = parser.parse_args(argv)

    facts_path = Path(args.facts)
    vocab_path = Path(args.vocab)
    overlay_path = Path(args.overlay)

    valid_ids_list, vocab_block = load_vocabulary(vocab_path)
    valid_ids = set(valid_ids_list)
    sys.stderr.write(f"Vocabulary: {len(valid_ids)} valid IDs\n")

    seen = load_overlay_seen(overlay_path)
    sys.stderr.write(f"Already in overlay: {len(seen)} fact_ids (will skip)\n")

    only_ids: set[str] | None = None
    only_untagged = True
    if args.only_fact_ids_file:
        only_ids = set()
        with open(args.only_fact_ids_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    only_ids.add(line)
        only_untagged = False
        sys.stderr.write(f"Restricted to {len(only_ids)} fact_ids from file\n")

    facts = select_facts_to_tag(facts_path, seen, only_ids, only_untagged)
    sys.stderr.write(f"Selected {len(facts)} facts to tag\n")
    if not facts:
        sys.stderr.write("Nothing to do.\n")
        return

    n_batches = (len(facts) + args.batch_size - 1) // args.batch_size
    if args.max_batches:
        n_batches = min(n_batches, args.max_batches)

    total_tagged = 0
    total_rejected = 0
    total_missing = 0
    failures: list[str] = []
    start = time.time()
    for batch_idx in range(n_batches):
        batch = facts[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        prompt = build_prompt(vocab_block, batch)
        expected_ids = [f["fact_id"] for f in batch]
        try:
            output = call_claude(prompt)
            results = parse_response(output, valid_ids, expected_ids)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            sys.stderr.write(f"  [batch {batch_idx + 1}/{n_batches}] FAILED: {exc}\n")
            failures.append(f"batch {batch_idx + 1}: {exc}")
            continue
        append_overlay(overlay_path, results, args.source_tag)
        n_tags = sum(len(r["targets"]) for r in results)
        n_rej = sum(len(r["rejected"]) for r in results)
        n_miss = sum(1 for r in results if r["missing"])
        total_tagged += n_tags
        total_rejected += n_rej
        total_missing += n_miss
        elapsed = time.time() - start
        eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
        sys.stderr.write(
            f"  [batch {batch_idx + 1}/{n_batches}] {len(batch)} facts, "
            f"{n_tags} tags ({n_rej} rejected, {n_miss} missing) — "
            f"elapsed {elapsed:.0f}s, eta {eta:.0f}s\n"
        )
        if args.rate_limit_sec > 0:
            time.sleep(args.rate_limit_sec)

    print()
    print(f"Done. Processed {n_batches} batches.")
    print(f"  Total tags written: {total_tagged}")
    print(f"  Vocab-rejected tags: {total_rejected}")
    print(f"  Missing fact responses: {total_missing}")
    print(f"  Batch failures: {len(failures)}")
    if failures:
        for f in failures:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
