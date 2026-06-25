#!/usr/bin/env python3
from __future__ import annotations

import unicodedata
import zipfile
from pathlib import Path
from sync_prompts import main as sync_prompts


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "obsidian-paper-recall-kit.zip"
INCLUDE = [
    "README.md",
    "START_HERE.md",
    "LICENSE.md",
    "guides",
    "templates",
    "sample-vault",
    "prompts",
]


def iter_files() -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE:
        path = ROOT / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            raise FileNotFoundError(path)
    return files


def main() -> None:
    sync_prompts()
    files = iter_files()
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, unicodedata.normalize("NFC", relative))
    print(f"wrote {OUTPUT} ({len(files)} files)")


if __name__ == "__main__":
    main()
