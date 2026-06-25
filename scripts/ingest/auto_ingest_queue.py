import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.ingest import auto_ingest
except ModuleNotFoundError:
    import auto_ingest  # type: ignore[no-redef]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a batch of arXiv IDs from a JSON queue file."
    )
    parser.add_argument("--queue", type=Path, required=True, help="Path to queue JSON file")
    parser.add_argument("--vault", type=Path, default=Path("."))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "research-wiki" / "arxiv",
    )
    args = parser.parse_args(argv)

    queue_path = args.queue
    if not queue_path.exists():
        print(f"Error: queue file not found: {queue_path}", file=sys.stderr)
        return 2

    try:
        data = json.loads(queue_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error: failed to read queue file: {exc}", file=sys.stderr)
        return 2

    arxiv_ids: list[str] = data.get("arxiv_ids", [])
    if not arxiv_ids:
        print("Error: no arxiv_ids in queue file", file=sys.stderr)
        return 2

    options: dict = data.get("options", {})

    ingest_argv = [
        "--arxiv-ids",
        ",".join(arxiv_ids),
        "--vault",
        str(args.vault),
        "--cache-dir",
        str(args.cache_dir),
    ]
    if options.get("merge"):
        ingest_argv.append("--merge")
    if options.get("skip_facts"):
        ingest_argv.append("--skip-facts")
    if options.get("no_download"):
        ingest_argv.append("--no-download")

    return auto_ingest.main(ingest_argv)


if __name__ == "__main__":
    sys.exit(main())
