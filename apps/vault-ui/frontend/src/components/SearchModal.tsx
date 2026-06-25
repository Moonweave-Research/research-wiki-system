import { useEffect, useRef, useState } from "react";

import { api, type SearchHit } from "../api.ts";

export default function SearchModal({
  open,
  onClose,
  onNavigate,
}: {
  open: boolean;
  onClose: () => void;
  onNavigate: (path: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchHit[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState(false);
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchSeqRef = useRef(0);

  // Reset and focus when opening.
  useEffect(() => {
    if (!open) return;
    setQuery("");
    setResults([]);
    setActiveIdx(0);
    // Defer focus so the modal is rendered before we attempt focus.
    const tid = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(tid);
  }, [open]);

  // Debounced search — cancel stale responses.
  useEffect(() => {
    if (!open) {
      searchSeqRef.current += 1;
      return;
    }
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const trimmed = query.trim();
    const seq = ++searchSeqRef.current;
    if (!trimmed) {
      setResults([]);
      setActiveIdx(0);
      setSearching(false);
      setSearchError(false);
      return;
    }
    setSearching(true);
    setSearchError(false);
    debounceRef.current = setTimeout(() => {
      api
        .search(trimmed)
        .then((data) => {
          if (searchSeqRef.current === seq) {
            setResults(data.results);
            setActiveIdx(0);
          }
        })
        .catch(() => {
          if (searchSeqRef.current === seq) {
            setResults([]);
            setSearchError(true);
          }
        })
        .finally(() => {
          if (searchSeqRef.current === seq) setSearching(false);
        });
    }, 120);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query, open]);

  // Scroll active row into view.
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const item = list.children[activeIdx] as HTMLElement | undefined;
    if (item) item.scrollIntoView({ block: "nearest" });
  }, [activeIdx]);

  function confirm(path: string) {
    onNavigate(path);
    onClose();
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx((i) => Math.min(i + 1, results.length - 1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx((i) => Math.max(i - 1, 0));
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      const hit = results[activeIdx];
      if (hit) confirm(hit.path);
      return;
    }
    if (e.key === "Tab") {
      // Trap Tab inside the panel — cycle between input and list items.
      e.preventDefault();
      const focusables = getFocusables();
      if (focusables.length === 0) return;
      const cur = document.activeElement;
      const idx = focusables.indexOf(cur as HTMLElement);
      const next = e.shiftKey
        ? focusables[(idx - 1 + focusables.length) % focusables.length]
        : focusables[(idx + 1) % focusables.length];
      next?.focus();
    }
  }

  function getFocusables(): HTMLElement[] {
    const panel = document.getElementById("search-modal-panel");
    if (!panel) return [];
    return Array.from(panel.querySelectorAll<HTMLElement>("input, button, [role=option]"));
  }

  if (!open) return null;

  return (
    <div
      className="search-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Search notes"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div id="search-modal-panel" className="search-panel" onKeyDown={handleKeyDown}>
        <input
          ref={inputRef}
          type="search"
          className="search-input"
          placeholder="Search notes…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search query"
          aria-autocomplete="list"
          aria-controls="search-results"
          aria-activedescendant={results[activeIdx] ? `sr-${activeIdx}` : undefined}
          autoComplete="off"
          spellCheck={false}
        />
        {results.length > 0 && (
          <ul
            ref={listRef}
            id="search-results"
            className="search-results"
            role="listbox"
            aria-label="Search results"
          >
            {results.map((hit, i) => (
              <li
                key={hit.path}
                id={`sr-${i}`}
                role="option"
                aria-selected={i === activeIdx}
                className={`search-hit${i === activeIdx ? " active" : ""}`}
                tabIndex={-1}
                onMouseEnter={() => setActiveIdx(i)}
                onClick={() => confirm(hit.path)}
              >
                <span className="search-hit-title">{hit.title}</span>
                <span className="search-hit-path">{hit.path}</span>
              </li>
            ))}
          </ul>
        )}
        {query.trim() && results.length === 0 && (
          <div className={`search-empty${searchError ? " err" : ""}`}>
            {searching
              ? "Searching…"
              : searchError
                ? "Search failed — try again."
                : `No results for "${query.trim()}"`}
          </div>
        )}
      </div>
    </div>
  );
}
