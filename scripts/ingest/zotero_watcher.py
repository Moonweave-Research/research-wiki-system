import argparse
import hashlib
import logging
import signal
import subprocess
import sys
import threading
import unicodedata
from datetime import datetime
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


class SyncHandler(FileSystemEventHandler):
    def __init__(
        self,
        target_path: Path,
        sync_args: list[str],
        vault_dir: Path,
        debounce_seconds: float,
    ):
        self._resolved_target = target_path.resolve()
        self._target_name = target_path.name
        self._sync_args = sync_args
        self._vault_dir = vault_dir
        self._debounce_seconds = debounce_seconds
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._running = False
        # Cache the export-file hash so Better BibTeX rewrites that don't
        # change content (the common case — BBT touches the file on every
        # refresh interval) skip the sync + log-line that follows.
        self._last_content_hash: str | None = None

    def on_modified(self, event) -> None:
        self._schedule_sync(event.src_path)

    def on_created(self, event) -> None:
        self._schedule_sync(event.src_path)

    def on_moved(self, event) -> None:
        # BBT writes atomically (temp → final rename); dest_path is the real target.
        self._schedule_sync(event.dest_path)

    def _schedule_sync(self, path: str) -> None:
        # macOS FSEvents uses NFD; Python Path.resolve() returns NFC.
        # Normalize both to NFC before comparing to handle Korean filenames.
        event_path = unicodedata.normalize("NFC", str(Path(path).resolve()))
        target_path = unicodedata.normalize("NFC", str(self._resolved_target))
        if event_path != target_path:
            return
        # Skip when content is byte-identical to the last sync — BBT
        # periodically rewrites the export even when no library item changed,
        # which used to produce 7+ redundant watcher entries per hour.
        current_hash = self._compute_hash()
        if current_hash is not None and current_hash == self._last_content_hash:
            logger.debug("Export content unchanged (hash match); skipping sync")
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._run_sync)
            self._timer.start()
        logger.info(f"Change detected, sync scheduled in {self._debounce_seconds}s")

    def _compute_hash(self) -> str | None:
        # 64KB prefix is sufficient — BBT exports are large but the manifest
        # header changes whenever any item is added/edited/removed.
        try:
            with open(self._resolved_target, "rb") as f:
                return hashlib.sha256(f.read(65536)).hexdigest()
        except OSError:
            return None

    def _run_sync(self) -> None:
        with self._lock:
            if self._running:
                logger.info("Sync in progress, retry scheduled...")
                self._timer = threading.Timer(self._debounce_seconds, self._run_sync)
                self._timer.start()
                return
            self._running = True
        try:
            logger.info("Running sync...")
            result = subprocess.run(
                self._sync_args,
                cwd=self._vault_dir,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                logger.info(line)
            for line in result.stderr.splitlines():
                logger.warning(line)

            if result.returncode != 0:
                logger.error(f"Sync failed (exit {result.returncode})")
                return

            # Auto-enrich any newly created notes (skips already-enriched ones).
            enrich_args = [
                sys.executable,
                str(self._vault_dir / "scripts" / "wiki_enrich.py"),
                "--all",
                "--delay",
                "0",
            ]
            enrich_result = subprocess.run(
                enrich_args,
                cwd=self._vault_dir,
                capture_output=True,
                text=True,
            )
            for line in enrich_result.stdout.splitlines():
                if not line.startswith("[SKIP"):
                    logger.info(f"enrich: {line}")
            if enrich_result.returncode != 0:
                logger.error(f"Enrich failed (exit {enrich_result.returncode})")

            log_path = self._vault_dir / "wiki" / "log.md"
            if log_path.exists():
                ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n## [{ts}] watcher | triggered sync for {self._target_name}\n")
            # Remember the content we just synced so the next BBT rewrite
            # with identical bytes short-circuits at _schedule_sync.
            self._last_content_hash = self._compute_hash()
        finally:
            with self._lock:
                self._running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch Zotero export and sync to wiki on change.")
    parser.add_argument("export_file", type=Path, help="Path to Better BibTeX auto-export JSON")
    parser.add_argument("--domain", default="research", choices=["research", "general"])
    parser.add_argument("--vault", type=Path, default=Path.cwd())
    parser.add_argument("--debounce", type=float, default=3.0)
    parser.add_argument("--fetch-abstracts", action="store_true")
    args = parser.parse_args()

    export_file = args.export_file.resolve()
    vault_dir = args.vault.resolve()

    if not export_file.exists():
        logger.warning(f"{export_file} not found — will watch for its creation")

    sync_args = [
        sys.executable,
        str(vault_dir / "scripts" / "zotero_sync.py"),
        str(export_file),
        "--domain",
        args.domain,
        "--merge",
    ]
    if args.fetch_abstracts:
        sync_args.append("--fetch-abstracts")

    handler = SyncHandler(export_file, sync_args, vault_dir, args.debounce)
    observer = Observer()
    # Watch the parent directory (watchdog cannot watch a single file directly).
    observer.schedule(handler, str(export_file.parent), recursive=False)

    stop_event = threading.Event()

    def _shutdown(signum: int, frame) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    observer.start()
    print(f"Watching {export_file}. Vault: {vault_dir}. Press Ctrl+C to stop.")
    stop_event.wait()
    observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
