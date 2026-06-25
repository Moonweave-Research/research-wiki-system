import type { RelatedHit } from "../api.ts";

export default function RelatedPanel({
  related,
  loading,
  error,
  onNavigate,
}: {
  related: RelatedHit[];
  loading: boolean;
  error?: boolean;
  onNavigate: (path: string) => void;
}) {
  if (loading) return <div className="empty">Finding related…</div>;
  if (error) return <div className="empty err">Couldn't load related papers.</div>;
  if (related.length === 0) return null; // concept/non-paper notes have no related layer
  return (
    <ul role="list" style={{ listStyle: "none" }}>
      {related.map((rel) => (
        <li key={rel.path}>
          <a
            href={`#/${rel.path}`}
            className="bl related"
            onClick={(event) => {
              event.preventDefault();
              onNavigate(rel.path);
            }}
          >
            <div className="t">{rel.title}</div>
            <div className="p" translate="no">
              {rel.path}
            </div>
          </a>
        </li>
      ))}
    </ul>
  );
}
