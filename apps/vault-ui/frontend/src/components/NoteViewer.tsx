import { useMemo } from "react";
import Markdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

import type { NoteResponse } from "../api.ts";

const WIKILINK_SCHEME = "wikilink:";

/** Rewrite [[target]], [[target|label]], [[target#h]] into markdown links with a
 *  custom scheme so the <a> renderer can resolve them against link_resolution. */
function rewriteWikilinks(markdown: string): string {
  return markdown.replace(/\[\[([^\]]+)\]\]/g, (_match, inner: string) => {
    const key = inner.split("|")[0].split("#")[0].trim();
    const hasLabel = inner.includes("|");
    const label = hasLabel ? inner.split("|")[1].trim() : key.replace(/_/g, " ");
    return `[${label}](${WIKILINK_SCHEME}${encodeURIComponent(key)})`;
  });
}

const STATUS_OPTIONS = ["unread", "reading", "done", "archived", "irrelevant"];

function frontmatterChips(fm: Record<string, unknown>) {
  const chips: { cls: string; text: string }[] = [];
  if (typeof fm.category === "string") chips.push({ cls: "", text: fm.category });
  const tags = Array.isArray(fm.tags) ? (fm.tags as unknown[]).slice(0, 6) : [];
  for (const tag of tags) chips.push({ cls: "tag", text: String(tag) });
  return chips;
}

export default function NoteViewer({
  note,
  loading,
  error,
  onNavigate,
  onStatusChange,
  statusError,
}: {
  note: NoteResponse | null;
  loading: boolean;
  error: string | null;
  onNavigate: (path: string) => void;
  onStatusChange?: (status: string) => void;
  statusError?: string | null;
}) {
  const rewritten = useMemo(() => (note ? rewriteWikilinks(note.markdown) : ""), [note]);

  if (loading) {
    return (
      <article className="article" aria-busy="true">
        <div className="skel title" />
        <div className="skel" />
        <div className="skel" style={{ width: "90%" }} />
        <div className="skel" style={{ width: "80%" }} />
      </article>
    );
  }
  if (error) {
    return (
      <article className="article">
        <p className="empty">Could not load note: {error}</p>
      </article>
    );
  }
  if (!note) {
    return (
      <article className="article">
        <p className="empty">Select a note from the tree to start reading.</p>
      </article>
    );
  }

  return (
    <article className="article">
      <div className="chips">
        {(() => {
          const status =
            typeof note.frontmatter.status === "string" ? note.frontmatter.status : null;
          if (onStatusChange) {
            return (
              <>
                <select
                  className="status-select"
                  value={status ?? "unread"}
                  aria-label="Reading status"
                  onChange={(event) => onStatusChange(event.target.value)}
                >
                  {STATUS_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
                {statusError && (
                  <span className="status-err" role="alert" title={statusError}>
                    save failed
                  </span>
                )}
              </>
            );
          }
          return status ? <span className="chip status">{status}</span> : null;
        })()}
        {frontmatterChips(note.frontmatter).map((chip, index) => (
          <span key={index} className={`chip ${chip.cls}`}>
            {chip.text}
          </span>
        ))}
      </div>
      <h1 className="title">{note.title}</h1>
      <div className="meta" translate="no">
        {note.path}
      </div>
      <div className="body">
        <Markdown
          remarkPlugins={[remarkGfm]}
          urlTransform={(url) =>
            url.startsWith(WIKILINK_SCHEME) ? url : defaultUrlTransform(url)
          }
          components={{
            a({ href, children }) {
              if (href && href.startsWith(WIKILINK_SCHEME)) {
                const key = decodeURIComponent(href.slice(WIKILINK_SCHEME.length));
                const target = note.link_resolution[key] ?? null;
                if (target) {
                  return (
                    <a
                      className="wl"
                      href={`#/${target}`}
                      onClick={(event) => {
                        event.preventDefault();
                        onNavigate(target);
                      }}
                    >
                      {children}
                    </a>
                  );
                }
                return (
                  <span className="wl broken" title={`unresolved: ${key}`}>
                    {children}
                  </span>
                );
              }
              return (
                <a href={href} target="_blank" rel="noopener noreferrer">
                  {children}
                </a>
              );
            },
          }}
        >
          {rewritten}
        </Markdown>
      </div>
    </article>
  );
}
