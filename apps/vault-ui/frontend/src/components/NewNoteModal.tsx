import { useEffect, useRef, useState } from "react";

export default function NewNoteModal({
  open,
  error,
  onClose,
  onCreate,
}: {
  open: boolean;
  error: string | null;
  onClose: () => void;
  onCreate: (title: string) => void;
}) {
  const [title, setTitle] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setTitle("");
    const tid = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(tid);
  }, [open]);

  if (!open) return null;

  function submit() {
    const trimmed = title.trim();
    if (trimmed) onCreate(trimmed);
  }

  return (
    <div
      className="search-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Create note"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="newnote-panel">
        <input
          ref={inputRef}
          className="search-input"
          placeholder="New note title…"
          value={title}
          aria-label="New note title"
          autoComplete="off"
          spellCheck={false}
          onChange={(event) => setTitle(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              submit();
            } else if (event.key === "Escape") {
              event.preventDefault();
              onClose();
            }
          }}
        />
        {error && <p className="editor-error">{error}</p>}
        <div className="newnote-actions">
          <button type="button" className="ed-btn" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="ed-btn save" onClick={submit} disabled={!title.trim()}>
            Create
          </button>
        </div>
      </div>
    </div>
  );
}
