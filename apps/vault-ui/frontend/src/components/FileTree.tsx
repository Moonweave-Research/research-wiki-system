import { useEffect, useRef, useState } from "react";

import type { TreeNode } from "../api.ts";

function countNotes(node: TreeNode): number {
  let total = node.notes.length;
  for (const child of node.children) total += countNotes(child);
  return total;
}

function Folder({
  node,
  depth,
  accPath,
  current,
  onSelect,
  focusedPath,
  setFocusedPath,
}: {
  node: TreeNode;
  depth: number;
  accPath: string;
  current: string | null;
  onSelect: (path: string) => void;
  focusedPath: string | null;
  setFocusedPath: (p: string | null) => void;
}) {
  const folderKey = accPath || "__root__";
  const [open, setOpen] = useState(depth < 1);
  const activeRef = useRef<HTMLAnchorElement | null>(null);

  // Auto-expand when current note lives inside this folder.
  useEffect(() => {
    if (current && accPath && current.startsWith(accPath + "/")) {
      setOpen(true);
    }
  }, [current, accPath]);

  // Scroll active leaf into view when it becomes active.
  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({ block: "nearest" });
    }
  }, [current]);

  const total = node.name ? countNotes(node) : 0;

  return (
    <div>
      {node.name && (
        <button
          type="button"
          className="tree-btn folder"
          role="treeitem"
          aria-expanded={open}
          tabIndex={focusedPath === folderKey ? 0 : -1}
          data-treepath={folderKey}
          onFocus={() => setFocusedPath(folderKey)}
          onClick={() => setOpen((v) => !v)}
          onKeyDown={(e) => {
            if (e.key === "ArrowRight") { e.preventDefault(); setOpen(true); }
            if (e.key === "ArrowLeft") { e.preventDefault(); setOpen(false); }
          }}
        >
          <span className="chev" aria-hidden="true">
            {open ? "▾" : "▸"}
          </span>
          {node.name}
          {total > 0 && <span className="count tnum" aria-hidden="true">{total}</span>}
        </button>
      )}
      {open && (
        <div role="group">
          {node.children.map((child) => {
            const childPath = accPath ? `${accPath}/${child.name}` : child.name;
            return (
              <Folder
                key={child.name}
                node={child}
                depth={depth + 1}
                accPath={childPath}
                current={current}
                onSelect={onSelect}
                focusedPath={focusedPath}
                setFocusedPath={setFocusedPath}
              />
            );
          })}
          {node.notes.map((note) => {
            const isActive = note.path === current;
            return (
              <a
                key={note.path}
                ref={isActive ? activeRef : null}
                href={`#/${note.path}`}
                role="treeitem"
                className={`tree-btn leaf${isActive ? " active" : ""}`}
                aria-current={isActive ? "page" : undefined}
                tabIndex={focusedPath === note.path ? 0 : -1}
                data-treepath={note.path}
                title={note.title}
                translate="no"
                onFocus={() => setFocusedPath(note.path)}
                onClick={(event) => {
                  event.preventDefault();
                  onSelect(note.path);
                }}
              >
                {note.title}
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}

function flattenTree(node: TreeNode): { path: string; title: string }[] {
  const results: { path: string; title: string }[] = [];
  for (const note of node.notes) results.push(note);
  for (const child of node.children) results.push(...flattenTree(child));
  return results;
}

export default function FileTree({
  tree,
  current,
  onSelect,
}: {
  tree: TreeNode | null;
  current: string | null;
  onSelect: (path: string) => void;
}) {
  const [filter, setFilter] = useState("");
  const [focusedPath, setFocusedPath] = useState<string | null>(null);
  const navRef = useRef<HTMLElement | null>(null);

  // Collect all treeitem paths in DOM order for roving navigation.
  function getItems(): HTMLElement[] {
    if (!navRef.current) return [];
    return Array.from(navRef.current.querySelectorAll<HTMLElement>("[role=treeitem]"));
  }

  function handleNavKeyDown(e: React.KeyboardEvent) {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    const items = getItems();
    const idx = items.findIndex((el) => el === document.activeElement);
    if (idx === -1) return;
    e.preventDefault();
    const next = e.key === "ArrowDown" ? items[idx + 1] : items[idx - 1];
    if (next) (next as HTMLElement).focus();
  }

  if (!tree) return <div className="empty">Loading vault…</div>;
  if (tree.children.length === 0 && tree.notes.length === 0) {
    return <div className="empty">No notes found.</div>;
  }

  const trimmed = filter.trim().toLowerCase();
  const flat = flattenTree(tree);
  const filtered = trimmed
    ? flat.filter(
        (n) => n.title.toLowerCase().includes(trimmed) || n.path.toLowerCase().includes(trimmed),
      )
    : [];
  // Roving tabindex: exactly one treeitem must be Tab-reachable on load (APG).
  // Fall back to the active note, else the first item, until the user focuses one.
  const effectiveUnfiltered = focusedPath ?? current ?? flat[0]?.path ?? null;
  const effectiveFiltered =
    focusedPath && filtered.some((n) => n.path === focusedPath)
      ? focusedPath
      : (filtered[0]?.path ?? null);

  return (
    <nav ref={navRef} role="tree" aria-label="Vault files" onKeyDown={handleNavKeyDown}>
      <div className="tree-filter-wrap">
        <input
          type="search"
          className="tree-filter"
          placeholder="Filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Filter notes"
        />
      </div>
      {trimmed ? (
        <div role="group">
          {filtered.map((note) => {
              const isActive = note.path === current;
              return (
                <a
                  key={note.path}
                  href={`#/${note.path}`}
                  role="treeitem"
                  className={`tree-btn leaf${isActive ? " active" : ""}`}
                  aria-current={isActive ? "page" : undefined}
                  tabIndex={effectiveFiltered === note.path ? 0 : -1}
                  data-treepath={note.path}
                  title={note.title}
                  translate="no"
                  onFocus={() => setFocusedPath(note.path)}
                  onClick={(event) => {
                    event.preventDefault();
                    onSelect(note.path);
                  }}
                >
                  {note.title}
                </a>
              );
            })}
        </div>
      ) : (
        <Folder
          node={tree}
          depth={0}
          accPath=""
          current={current}
          onSelect={onSelect}
          focusedPath={effectiveUnfiltered}
          setFocusedPath={setFocusedPath}
        />
      )}
    </nav>
  );
}
