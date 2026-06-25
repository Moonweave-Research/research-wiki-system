import argparse
import json
import logging
import signal
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from scripts.ingest.arxiv_ingest import ATOM_NS, canonical_arxiv_id
except ModuleNotFoundError:
    from arxiv_ingest import ATOM_NS, canonical_arxiv_id  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def _fetch_atom(query: str, max_results: int) -> bytes:
    encoded_query = urllib.parse.quote(query, safe="")
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query={encoded_query}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
    )
    req = urllib.request.Request(
        url, headers={"User-Agent": "ResearchWiki/1.0 (mailto:research@example.com)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _parse_ids_from_atom(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    ids = []
    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        id_el = entry.find(f"{{{ATOM_NS}}}id")
        if id_el is None or not id_el.text:
            continue
        raw = id_el.text.strip()
        # Strip URL prefix: http://arxiv.org/abs/2401.04088v2
        if "/" in raw:
            raw = raw.rsplit("/", 1)[-1]
        ids.append(canonical_arxiv_id(raw))
    return ids


def _load_state(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return set(data.get("seen_ids", []))
    except (json.JSONDecodeError, OSError) as exc:
        # Move the corrupt file aside so the failure is visible and recoverable
        # rather than silently re-ingesting the whole arXiv query window.
        backup = state_file.with_suffix(state_file.suffix + ".corrupt")
        try:
            state_file.rename(backup)
            logger.error(
                f"State file {state_file} unreadable ({exc}); moved to {backup}. "
                "Treating as empty for THIS poll only — inspect the backup before next run."
            )
        except OSError:
            logger.error(f"State file {state_file} unreadable ({exc}) and cannot be moved aside.")
        return set()


def _save_state(state_file: Path, seen_ids: set[str]) -> None:
    """Atomic write: tmp file + os.replace. SIGKILL mid-write would otherwise
    leave the state file truncated/blank, causing the next poll to re-ingest
    every ID in the arXiv query window (rate-limit risk against arXiv).
    """
    import os

    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps({"seen_ids": sorted(seen_ids)}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, state_file)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _invoke_auto_ingest(
    arxiv_id: str,
    vault: Path,
    cache_dir: Path,
    merge: bool,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "scripts.ingest.auto_ingest",
        "--arxiv-ids",
        arxiv_id,
        "--vault",
        str(vault),
        "--cache-dir",
        str(cache_dir),
    ]
    if merge:
        cmd.append("--merge")
    if dry_run:
        cmd.append("--dry-run")

    # Must run from REPO_ROOT (not vault) so `python -m scripts.ingest.auto_ingest` resolves
    # when --vault points outside this repository. Same constraint as auto_ingest._run_stage*.
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    for line in result.stdout.splitlines():
        logger.info(f"[{arxiv_id}] {line}")
    for line in result.stderr.splitlines():
        logger.warning(f"[{arxiv_id}] {line}")
    return result.returncode


def _poll_once(args: argparse.Namespace, seen_ids: set[str]) -> set[str]:
    logger.info(f"Polling arXiv for query: {args.query}")
    try:
        xml_bytes = _fetch_atom(args.query, args.max_per_poll * 3)
    except Exception as exc:
        logger.error(f"arXiv fetch failed: {exc}")
        return seen_ids

    try:
        all_ids = _parse_ids_from_atom(xml_bytes)
    except ET.ParseError as exc:
        # A malformed Atom response otherwise propagates as an uncaught exception
        # and kills the daemon between polls. Skip and retry next interval.
        logger.error(f"arXiv Atom parse failed: {exc}")
        return seen_ids
    new_ids = [aid for aid in all_ids if aid not in seen_ids]
    new_ids = new_ids[: args.max_per_poll]

    if not new_ids:
        logger.info("No new IDs found.")
        return seen_ids

    logger.info(f"Found {len(new_ids)} new ID(s): {new_ids}")

    attempted: list[str] = []
    for arxiv_id in new_ids:
        rc = _invoke_auto_ingest(
            arxiv_id=arxiv_id,
            vault=args.vault,
            cache_dir=args.cache_dir,
            merge=args.merge,
            dry_run=args.dry_run,
        )
        if rc != 0:
            logger.warning(f"auto_ingest exited {rc} for {arxiv_id}")
        # Mark seen on attempt regardless of outcome to avoid hammering arXiv on retries.
        attempted.append(arxiv_id)

    updated = seen_ids | set(attempted)
    _save_state(args.state_file, updated)
    logger.info(f"State updated: {len(attempted)} new ID(s) marked seen.")
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Poll arXiv RSS and ingest new papers into the wiki."
    )
    parser.add_argument(
        "--query", required=True, help="arXiv search query (e.g. cat:cond-mat.soft)"
    )
    parser.add_argument("--interval", type=int, default=3600, help="Poll interval in seconds")
    parser.add_argument("--vault", type=Path, default=Path("."))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "research-wiki" / "arxiv",
    )
    parser.add_argument("--max-per-poll", type=int, default=10)
    parser.add_argument("--state-file", type=Path, default=Path(".arxiv_watcher_state.json"))
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(argv)

    args.vault = args.vault.resolve()
    if not args.state_file.is_absolute():
        args.state_file = args.vault / args.state_file
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    seen_ids = _load_state(args.state_file)
    logger.info(f"Loaded {len(seen_ids)} seen ID(s) from state file.")

    if args.once:
        _poll_once(args, seen_ids)
        return 0

    stop_event = threading.Event()

    def _shutdown(signum: int, frame) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while not stop_event.is_set():
        seen_ids = _poll_once(args, seen_ids)
        if stop_event.wait(timeout=args.interval):
            break

    logger.info("Watcher stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
