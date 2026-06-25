import json
from types import SimpleNamespace

from scripts.core import wiki_llm_provider as mod


def test_claude_call_accepts_legacy_result_envelope(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"is_error": False, "result": "legacy text"}),
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    assert mod._claude_call("prompt", system=None, model="sonnet", timeout_s=1) == "legacy text"


def test_claude_call_accepts_assistant_message_list_envelope(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "text", "text": "new text"},
                            ]
                        },
                    }
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    assert mod._claude_call("prompt", system=None, model="sonnet", timeout_s=1) == "new text"
