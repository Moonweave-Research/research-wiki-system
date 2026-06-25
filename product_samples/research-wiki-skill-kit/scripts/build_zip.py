from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
DOWNLOADS = ROOT / "downloads"
ZIP_PATH = DOWNLOADS / "obsidian-research-wiki-skill-prompt-kit.zip"
PACKAGE = "obsidian-research-wiki-skill-prompt-kit"
FILES = [
    "PROMPT_KIT.md",
    "00_README.md",
    "01_main_prompt.md",
    "02_qa_prompt.md",
    "03_test_prompt.md",
    "04_distribution_copy.md",
    "LICENSE.md",
]


def main() -> None:
    DOWNLOADS.mkdir(exist_ok=True)
    (DOWNLOADS / "PROMPT_KIT.md").write_text(
        (SOURCE / "PROMPT_KIT.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (ROOT / "LICENSE.md").write_text(
        (SOURCE / "LICENSE.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{PACKAGE}/", "")
        for name in FILES:
            archive.write(SOURCE / name, f"{PACKAGE}/{name}")


if __name__ == "__main__":
    main()
