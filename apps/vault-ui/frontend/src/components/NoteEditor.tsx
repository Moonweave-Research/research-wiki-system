import { useEffect, useRef, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { markdown } from "@codemirror/lang-markdown";
import { EditorView } from "@codemirror/view";

import { api, type NoteResponse } from "../api.ts";

// Dark, token-aligned CodeMirror theme. Neon stays on the caret/selection only.
const darkTheme = EditorView.theme(
  {
    "&": { backgroundColor: "transparent", color: "#cfd2d9", fontSize: "15px" },
    ".cm-content": {
      fontFamily: "var(--mono)",
      caretColor: "var(--neon)",
      lineHeight: "1.7",
      padding: "16px 0",
    },
    ".cm-gutters": {
      backgroundColor: "transparent",
      color: "var(--dim)",
      border: "none",
    },
    "&.cm-focused .cm-cursor": { borderLeftColor: "var(--neon)" },
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": {
      backgroundColor: "var(--neon-soft)",
    },
    ".cm-activeLine": { backgroundColor: "rgba(255,255,255,0.02)" },
    ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--muted)" },
  },
  { dark: true },
);

export default function NoteEditor({
  note,
  onSaved,
  onCancel,
}: {
  note: NoteResponse;
  onSaved: () => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(note.markdown);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dirty = value !== note.markdown;

  function save() {
    if (saving || value === note.markdown) return;
    setSaving(true);
    setError(null);
    api
      .saveNote(note.path, value, note.mtime)
      .then(() => onSaved())
      .catch((err) => {
        const message = String(err.message ?? err);
        setError(
          message.startsWith("409")
            ? "This note changed on disk since you opened it — reload to get the latest, then re-apply your edits."
            : message,
        );
      })
      .finally(() => setSaving(false));
  }

  // Keep ⌘S/Ctrl+S bound to the latest save closure without rebinding per keystroke.
  const saveRef = useRef(save);
  saveRef.current = save;
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && (event.key === "s" || event.key === "S")) {
        event.preventDefault();
        saveRef.current();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <article className="article editor">
      <div className="editor-bar">
        <span className="meta" translate="no">
          {note.path}
        </span>
        <div className="editor-actions">
          <button type="button" className="ed-btn" onClick={onCancel} disabled={saving}>
            Cancel
          </button>
          <button
            type="button"
            className="ed-btn save"
            onClick={save}
            disabled={saving || !dirty}
            title="Save (⌘S)"
          >
            {saving ? "Saving…" : "Save ⌘S"}
          </button>
        </div>
      </div>
      {error && <p className="editor-error">Save failed: {error}</p>}
      <CodeMirror
        value={value}
        theme={darkTheme}
        extensions={[markdown(), EditorView.lineWrapping]}
        onChange={setValue}
        basicSetup={{ lineNumbers: true, foldGutter: false, highlightActiveLine: true }}
      />
    </article>
  );
}
