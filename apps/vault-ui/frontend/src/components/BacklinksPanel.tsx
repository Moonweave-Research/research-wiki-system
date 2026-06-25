import type { Backlink } from "../api.ts";

export default function BacklinksPanel({
  backlinks,
  loading,
  error,
  onNavigate,
}: {
  backlinks: Backlink[];
  loading: boolean;
  error?: boolean;
  onNavigate: (path: string) => void;
}) {
  if (loading) return <div className="empty">Loading backlinks…</div>;
  if (error) return <div className="empty err">Couldn't load backlinks.</div>;
  if (backlinks.length === 0) return <div className="empty">No backlinks to this note.</div>;
  return (
    <ul role="list" style={{ listStyle: "none" }}>
      {backlinks.map((bl) => (
        <li key={bl.src_path}>
          <a
            href={`#/${bl.src_path}`}
            className="bl"
            onClick={(event) => {
              event.preventDefault();
              onNavigate(bl.src_path);
            }}
          >
            <div className="t">{bl.src_title}</div>
            <div className="p" translate="no">
              {bl.src_path}
            </div>
          </a>
        </li>
      ))}
    </ul>
  );
}
