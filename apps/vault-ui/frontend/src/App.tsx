import { useCallback, useEffect, useRef, useState } from "react";

import {
  api,
  type BacklinksResponse,
  type ConceptAuditResponse,
  type GraphResponse,
  type Health,
  type NoteResponse,
  type RelatedResponse,
  type SearchHit,
  type TagCount,
  type TreeResponse,
} from "./api.ts";
import BacklinksPanel from "./components/BacklinksPanel.tsx";
import RelatedPanel from "./components/RelatedPanel.tsx";
import FileTree from "./components/FileTree.tsx";
import TagFacet from "./components/TagFacet.tsx";
import ConceptAuditPanel from "./components/ConceptAuditPanel.tsx";
import GraphView from "./components/GraphView.tsx";
import NoteViewer from "./components/NoteViewer.tsx";
import NoteEditor from "./components/NoteEditor.tsx";
import NewNoteModal from "./components/NewNoteModal.tsx";
import SearchModal from "./components/SearchModal.tsx";

type Mode = "reader" | "graph";
type CaptureStep = "idle" | "global-bloom" | "core-collapse" | "local-expand" | "similar-burst" | "final-map";

function pathFromHash(): string | null {
  const hash = window.location.hash;
  return hash.startsWith("#/") ? decodeURIComponent(hash.slice(2)) : null;
}

export default function App() {
  const [captureMode, setCaptureModeState] = useState(
    () => new URLSearchParams(window.location.search).get("capture") === "1",
  );
  const [tree, setTree] = useState<TreeResponse | null>(null);
  const [current, setCurrent] = useState<string | null>(pathFromHash());
  const [note, setNote] = useState<NoteResponse | null>(null);
  const [noteLoading, setNoteLoading] = useState(false);
  const [noteError, setNoteError] = useState<string | null>(null);
  const [backlinks, setBacklinks] = useState<BacklinksResponse | null>(null);
  const [blLoading, setBlLoading] = useState(false);
  const [related, setRelated] = useState<RelatedResponse | null>(null);
  const [relLoading, setRelLoading] = useState(false);
  const [treeOpen, setTreeOpen] = useState(false);
  const [backOpen, setBackOpen] = useState(false);
  const [mode, setMode] = useState<Mode>("reader");
  const [editing, setEditing] = useState(false);
  const [reloadNonce, setReloadNonce] = useState(0);
  const [backlinksError, setBacklinksError] = useState(false);
  const [relatedError, setRelatedError] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphDepth, setGraphDepth] = useState(1);
  const [graphScope, setGraphScope] = useState<"local" | "global">("local");
  const [graphSimilar, setGraphSimilar] = useState(false);
  const [captureStep, setCaptureStep] = useState<CaptureStep>("idle");
  const [audit, setAudit] = useState<ConceptAuditResponse | null>(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const copyTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const searchTriggerRef = useRef<HTMLElement | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [tags, setTags] = useState<TagCount[]>([]);
  const [tagsOpen, setTagsOpen] = useState(false);
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const [tagNotes, setTagNotes] = useState<SearchHit[]>([]);
  const [newNoteOpen, setNewNoteOpen] = useState(false);
  const [newNoteError, setNewNoteError] = useState<string | null>(null);

  const setCaptureMode = useCallback((enabled: boolean) => {
    const url = new URL(window.location.href);
    if (enabled) {
      url.searchParams.set("capture", "1");
    } else {
      url.searchParams.delete("capture");
    }
    window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
    setCaptureModeState(enabled);
  }, []);

  const refreshTree = useCallback(
    () =>
      api
        .tree()
        .then(setTree)
        .catch(() => setTree({ stale: true, tree: { name: "", children: [], notes: [] } })),
    [],
  );

  useEffect(() => {
    refreshTree();
  }, [refreshTree]);

  useEffect(() => {
    if (!captureMode) return;
    const focusPath = "concepts/Actuator.md";
    setCurrent(focusPath);
    setMode("graph");
    setEditing(false);
    setCaptureStep("global-bloom");
    setGraphScope("global");
    setGraphDepth(1);
    setGraphSimilar(false);
    const timers = [
      window.setTimeout(() => {
        setCaptureStep("core-collapse");
        setGraphScope("global");
        setGraphDepth(1);
        setGraphSimilar(false);
      }, 4000),
      window.setTimeout(() => {
        setCaptureStep("local-expand");
        setGraphScope("local");
        setGraphDepth(2);
        setGraphSimilar(false);
      }, 8000),
      window.setTimeout(() => {
        setCaptureStep("similar-burst");
        setGraphScope("local");
        setGraphDepth(2);
        setGraphSimilar(true);
      }, 12000),
      window.setTimeout(() => {
        setCaptureStep("final-map");
        setGraphScope("global");
        setGraphDepth(2);
        setGraphSimilar(true);
      }, 16000),
    ];
    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [captureMode]);

  useEffect(() => {
    if (!captureMode) {
      setCaptureStep("idle");
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setCaptureMode(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [captureMode, setCaptureMode]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, [reloadNonce]);

  useEffect(() => {
    api.tags().then((res) => setTags(res.tags)).catch(() => setTags([]));
  }, [reloadNonce]);

  useEffect(() => {
    if (!activeTag) {
      setTagNotes([]);
      return;
    }
    let cancelled = false;
    api
      .notesByTag(activeTag)
      .then((res) => !cancelled && setTagNotes(res.notes))
      .catch(() => !cancelled && setTagNotes([]));
    return () => {
      cancelled = true;
    };
  }, [activeTag, reloadNonce]);

  useEffect(() => {
    const onHash = () => setCurrent(pathFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        searchTriggerRef.current = document.activeElement as HTMLElement;
        setSearchOpen(true);
        return;
      }
      if (e.key === "g" && !e.metaKey && !e.ctrlKey && !e.altKey && !searchOpen) {
        const tag = (document.activeElement as HTMLElement).tagName.toLowerCase();
        const editable = (document.activeElement as HTMLElement).isContentEditable;
        if (tag === "input" || tag === "textarea" || editable) return;
        setMode((m) => (m === "graph" ? "reader" : "graph"));
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [searchOpen]);

  const closeSearch = useCallback(() => {
    setSearchOpen(false);
    // Restore focus to the element that triggered the modal.
    const trigger = searchTriggerRef.current;
    if (trigger && typeof trigger.focus === "function") {
      setTimeout(() => trigger.focus(), 0);
    }
  }, []);

  const copyPath = useCallback(() => {
    if (!current) return;
    navigator.clipboard.writeText(current).then(() => {
      setCopied(true);
      if (copyTimeout.current) clearTimeout(copyTimeout.current);
      copyTimeout.current = setTimeout(() => setCopied(false), 1500);
    });
  }, [current]);

  const navigate = useCallback((path: string) => {
    window.location.hash = `#/${path}`;
    setCurrent(path);
    setTreeOpen(false);
  }, []);

  const changeStatus = useCallback(
    (status: string) => {
      if (!current || !note) return;
      const prevNote = note;
      setStatusError(null);
      setNote({ ...note, frontmatter: { ...note.frontmatter, status } });
      api
        .setStatus(current, status)
        .then(() => setReloadNonce((nonce) => nonce + 1))
        .catch((err) => {
          setNote(prevNote);
          setStatusError(String(err.message ?? err));
        });
    },
    [current, note],
  );

  const submitNewNote = useCallback(
    (title: string) => {
      setNewNoteError(null);
      api
        .createNote(title)
        .then((res) => {
          setNewNoteOpen(false);
          refreshTree();
          navigate(res.path);
        })
        .catch((err) => setNewNoteError(String(err.message ?? err)));
    },
    [refreshTree, navigate],
  );

  useEffect(() => {
    if (!current) {
      setNote(null);
      setBacklinks(null);
      setRelated(null);
      return;
    }
    document.title = `${current.split("/").pop()} · research·vault`;
    let cancelled = false;
    setNoteLoading(true);
    setNoteError(null);
    api
      .note(current)
      .then((data) => !cancelled && setNote(data))
      .catch((err) => !cancelled && setNoteError(String(err.message ?? err)))
      .finally(() => !cancelled && setNoteLoading(false));
    setBlLoading(true);
    setBacklinksError(false);
    api
      .backlinks(current)
      .then((data) => !cancelled && setBacklinks(data))
      .catch(() => {
        if (cancelled) return;
        setBacklinks({ stale: false, backlinks: [] });
        setBacklinksError(true);
      })
      .finally(() => !cancelled && setBlLoading(false));
    setRelLoading(true);
    setRelatedError(false);
    api
      .related(current)
      .then((data) => !cancelled && setRelated(data))
      .catch(() => {
        if (cancelled) return;
        setRelated({ stale: false, related: [] });
        setRelatedError(true);
      })
      .finally(() => !cancelled && setRelLoading(false));
    return () => {
      cancelled = true;
    };
  }, [current, reloadNonce]);

  // Leaving the note (or going to graph) drops out of edit mode.
  useEffect(() => {
    setEditing(false);
    setStatusError(null);
  }, [current, mode]);

  useEffect(() => {
    if (mode !== "graph") return;
    if (graphScope === "local" && !current) return;
    let cancelled = false;
    setGraphLoading(true);
    const request =
      graphScope === "global"
        ? api.graphGlobal(current, graphSimilar)
        : api.graph(current as string, graphDepth, graphSimilar);
    request
      .then((data) => !cancelled && setGraph(data))
      .catch(() => !cancelled && setGraph({ stale: false, nodes: [], links: [] }))
      .finally(() => !cancelled && setGraphLoading(false));
    return () => {
      cancelled = true;
    };
  }, [mode, current, graphDepth, graphScope, graphSimilar]);

  useEffect(() => {
    if (mode !== "graph") return;
    let cancelled = false;
    setAuditLoading(true);
    api
      .conceptAudit()
      .then((data) => !cancelled && setAudit(data))
      .catch(() => !cancelled && setAudit(null))
      .finally(() => !cancelled && setAuditLoading(false));
    return () => {
      cancelled = true;
    };
  }, [mode, reloadNonce]);

  const stale = (tree?.stale ?? false) || (health?.stale ?? false);
  const blCount = backlinks?.backlinks.length ?? 0;

  return (
    <div className={`app${captureMode ? " capture-mode" : ""}`} data-capture-step={captureStep}>
      <aside className={`col tree${treeOpen ? " open" : ""}`}>
        <div className="head">
          <span className="brand">
            research<span className="dot">·</span>vault
          </span>
          <button
            type="button"
            className="newnote-btn"
            aria-label="Create note"
            onClick={() => {
              setNewNoteError(null);
              setNewNoteOpen(true);
            }}
          >
            + New
          </button>
          <button
            type="button"
            className="iconbtn tree-close"
            aria-label="Close file tree"
            onClick={() => setTreeOpen(false)}
          >
            ✕
          </button>
        </div>
        {stale && (
          <div className="banner">
            Link-graph index missing or stale. Run <code>python -m scripts.core.linkgraph build</code>.
          </div>
        )}
        <div className="scroll">
          <button
            type="button"
            className="tags-toggle"
            aria-expanded={tagsOpen || activeTag !== null}
            onClick={() => setTagsOpen((v) => !v)}
          >
            Tags {tagsOpen || activeTag ? "▾" : "▸"}
            <span className="count tnum">{tags.length}</span>
          </button>
          {(tagsOpen || activeTag) && (
            <TagFacet
              tags={tags}
              activeTag={activeTag}
              notes={tagNotes}
              current={current}
              onSelectTag={(tag) => {
                setActiveTag(tag);
                setTreeOpen(false);
              }}
              onClear={() => setActiveTag(null)}
              onNavigate={navigate}
            />
          )}
          {!activeTag && (
            <>
              <div className="seclabel tnum">
                Vault{tree ? ` · ${countNotes(tree.tree)} notes` : ""}
              </div>
              <FileTree tree={tree?.tree ?? null} current={current} onSelect={navigate} />
            </>
          )}
        </div>
      </aside>

      <main className="col reader">
        <div className="head">
          <button
            type="button"
            className="iconbtn tree-toggle"
            aria-label="Toggle file tree"
            onClick={() => setTreeOpen((v) => !v)}
          >
            ☰
          </button>
          <span className="crumb" translate="no">
            {current ?? "no note selected"}
          </span>
          {current && (
            <button
              type="button"
              className="iconbtn copy-path-btn"
              aria-label="Copy note path"
              onClick={copyPath}
            >
              {copied ? "✓" : "⎘"}
            </button>
          )}
          <div className="modeseg" role="group" aria-label="View mode">
            <button
              type="button"
              className={mode === "reader" ? "on" : ""}
              aria-pressed={mode === "reader"}
              onClick={() => setMode("reader")}
            >
              Read
            </button>
            <button
              type="button"
              className={mode === "graph" ? "on" : ""}
              aria-pressed={mode === "graph"}
              onClick={() => setMode("graph")}
            >
              Graph
            </button>
          </div>
          {mode === "reader" && note && (
            <button
              type="button"
              className={`ed-toggle${editing ? " on" : ""}`}
              aria-pressed={editing}
              onClick={() => setEditing((v) => !v)}
            >
              {editing ? "Reading" : "Edit"}
            </button>
          )}
          <button
            type="button"
            className="kbd"
            aria-label="Open search"
            onClick={() => {
              searchTriggerRef.current = document.activeElement as HTMLElement;
              setSearchOpen(true);
            }}
          >
            ⌘K
          </button>
          <button
            type="button"
            className="iconbtn back-toggle"
            aria-label="Toggle backlinks"
            onClick={() => setBackOpen((v) => !v)}
          >
            ⤺
          </button>
        </div>
        {mode === "graph" ? (
          <div className="graph-shell">
            <div className="graph-stage">
              <GraphView
                data={graph}
                loading={graphLoading && (graphScope === "global" || current !== null)}
                onNavigate={navigate}
                depth={graphDepth}
                onDepthChange={setGraphDepth}
                scope={graphScope}
                onScopeChange={setGraphScope}
                similar={graphSimilar}
                onSimilarChange={setGraphSimilar}
                captureMode={captureMode}
                onCaptureModeChange={setCaptureMode}
                captureStep={captureStep}
                captureStats={{ notes: health?.notes, links: health?.links }}
              />
            </div>
            <ConceptAuditPanel audit={audit} loading={auditLoading} onNavigate={navigate} />
          </div>
        ) : editing && note ? (
          <div className="scroll">
            <NoteEditor
              note={note}
              onSaved={() => {
                setEditing(false);
                setReloadNonce((nonce) => nonce + 1);
              }}
              onCancel={() => setEditing(false)}
            />
          </div>
        ) : (
          <div className="scroll">
            <NoteViewer
              note={note}
              loading={noteLoading}
              error={noteError}
              statusError={statusError}
              onNavigate={navigate}
              onStatusChange={changeStatus}
            />
          </div>
        )}
      </main>

      <SearchModal open={searchOpen} onClose={closeSearch} onNavigate={navigate} />

      <NewNoteModal
        open={newNoteOpen}
        error={newNoteError}
        onClose={() => setNewNoteOpen(false)}
        onCreate={submitNewNote}
      />

      <aside className={`col back${backOpen ? " open" : ""}`}>
        <div className="head">
          <span className="seclabel" style={{ margin: 0 }}>
            Backlinks <span className="count tnum">{blCount}</span>
          </span>
          <button
            type="button"
            className="iconbtn back-close"
            aria-label="Close backlinks"
            onClick={() => setBackOpen(false)}
          >
            ✕
          </button>
        </div>
        <div className="scroll">
          <BacklinksPanel
            backlinks={backlinks?.backlinks ?? []}
            loading={blLoading && current !== null}
            error={backlinksError}
            onNavigate={navigate}
          />
          {current && ((related?.related.length ?? 0) > 0 || relatedError) && (
            <>
              <div className="seclabel related-head">
                Related <span className="count tnum">{related?.related.length ?? 0}</span>
              </div>
              <RelatedPanel
                related={related?.related ?? []}
                loading={relLoading && current !== null}
                error={relatedError}
                onNavigate={navigate}
              />
            </>
          )}
        </div>
      </aside>
    </div>
  );
}

function countNotes(node: { children: { name: string }[]; notes: unknown[] }): number {
  let total = node.notes.length;
  for (const child of node.children as never[]) total += countNotes(child);
  return total;
}
