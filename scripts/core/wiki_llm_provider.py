"""LLM provider adapter — claude or codex CLI subprocess.

Selected via the LLM_PROVIDER env var (default: claude). Each adapter returns
the raw response *text*, stripped of any envelope or banner. Callers parse
domain-specific structure (e.g., JSON-in-text) on top.

claude path: `claude --print --output-format json --model <model> <prompt>`,
parses the envelope and returns `result`.

codex path: `codex exec --skip-git-repo-check --ephemeral <prompt>`. The
banner and token-count footer go to stderr; stdout is the bare response.
The model is selected via codex's `~/.codex/config.toml`; override with
the `model` argument if non-None.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Final

DEFAULT_PROVIDER: Final = "claude"
DEFAULT_TIMEOUT_S: Final = 180
DEFAULT_CLAUDE_MODEL: Final = "sonnet"


class ProviderError(RuntimeError):
    pass


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _claude_result_from_envelope(envelope) -> str:
    if isinstance(envelope, dict):
        if envelope.get("is_error"):
            raise ProviderError(f"claude API error: {envelope.get('api_error_status')}")
        if "result" in envelope:
            return str(envelope.get("result", "")).strip()
        if "message" in envelope and isinstance(envelope["message"], dict):
            return _content_text(envelope["message"].get("content")).strip()
        return _content_text(envelope.get("content")).strip()
    if isinstance(envelope, list):
        parts = []
        for item in envelope:
            text = _claude_result_from_envelope(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    raise ProviderError(f"unexpected claude JSON envelope type: {type(envelope).__name__}")


def _claude_call(prompt: str, system: str | None, model: str, timeout_s: int) -> str:
    args = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
    ]
    if system:
        args.extend(["--append-system-prompt", system])
    args.append(prompt)
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        raise ProviderError(f"claude CLI rc={result.returncode}: {result.stderr[:300]}")
    try:
        envelope = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON envelope: {result.stdout[:200]}") from exc
    return _claude_result_from_envelope(envelope)


def _codex_call(prompt: str, system: str | None, model: str | None, timeout_s: int) -> str:
    args = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-rules",
    ]
    if model:
        args.extend(["-m", model])
    # codex has no equivalent of --append-system-prompt for `exec`; bundle the
    # system instructions into the prompt body. Both providers thus see the
    # same context, just delivered differently.
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    args.append(full_prompt)
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        raise ProviderError(f"codex CLI rc={result.returncode}: {result.stderr[:300]}")
    return result.stdout.strip()


def provider_call(
    prompt: str,
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> str:
    """Dispatch to the configured LLM provider and return the response text.

    `system` is the system prompt: claude uses --append-system-prompt; codex
    has no exec-mode equivalent so it gets prepended to the user prompt.
    `provider` overrides the env var; `model` is forwarded to the underlying
    CLI (claude defaults to sonnet, codex uses its config default unless set).
    """
    chosen = provider or os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    if chosen == "claude":
        return _claude_call(
            prompt, system=system, model=model or DEFAULT_CLAUDE_MODEL, timeout_s=timeout_s
        )
    if chosen == "codex":
        return _codex_call(prompt, system=system, model=model, timeout_s=timeout_s)
    raise ProviderError(f"unknown LLM_PROVIDER={chosen!r}, expected 'claude' or 'codex'")
