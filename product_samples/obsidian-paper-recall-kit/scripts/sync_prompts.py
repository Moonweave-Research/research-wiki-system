#!/usr/bin/env python3
from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE = ROOT / "prompts" / "research-wiki-design-prompt.md"
INDEX_FILE = ROOT / "index.html"
PROMPT_IDS = [
    "full-prompt",
    "prompt-step-1",
    "prompt-step-2",
    "prompt-step-3",
    "prompt-step-4",
    "prompt-step-5",
]


def read_prompt_blocks() -> list[str]:
    markdown = PROMPT_FILE.read_text(encoding="utf-8")
    blocks = [
        match.group(1).strip()
        for match in re.finditer(r"```text\n([\s\S]*?)```", markdown)
    ]
    if len(blocks) != len(PROMPT_IDS):
        raise ValueError(
            f"expected {len(PROMPT_IDS)} prompt blocks, found {len(blocks)}"
        )
    return blocks


def replace_pre(html_text: str, prompt_id: str, content: str) -> str:
    pattern = re.compile(
        rf'(<pre id="{re.escape(prompt_id)}">)([\s\S]*?)(</pre>)'
    )
    replacement = rf"\1{html.escape(content, quote=False)}\3"
    html_text, count = pattern.subn(replacement, html_text, count=1)
    if count != 1:
        raise ValueError(f"could not update pre block: {prompt_id}")
    return html_text


def main() -> None:
    blocks = read_prompt_blocks()
    html_text = INDEX_FILE.read_text(encoding="utf-8")
    for prompt_id, block in zip(PROMPT_IDS, blocks, strict=True):
        html_text = replace_pre(html_text, prompt_id, block)
    INDEX_FILE.write_text(html_text, encoding="utf-8")
    print(f"synced {len(blocks)} prompt blocks into {INDEX_FILE}")


if __name__ == "__main__":
    main()
