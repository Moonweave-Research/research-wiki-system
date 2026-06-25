import type { SearchHit, TagCount } from "../api.ts";

export default function TagFacet({
  tags,
  activeTag,
  notes,
  current,
  onSelectTag,
  onClear,
  onNavigate,
}: {
  tags: TagCount[];
  activeTag: string | null;
  notes: SearchHit[];
  current: string | null;
  onSelectTag: (tag: string) => void;
  onClear: () => void;
  onNavigate: (path: string) => void;
}) {
  if (activeTag) {
    return (
      <div className="tag-filter">
        <button type="button" className="tag-clear" onClick={onClear}>
          ← all notes
        </button>
        <div className="seclabel">
          <span className="tag-chip on">{activeTag}</span>
          <span className="count tnum">{notes.length}</span>
        </div>
        <ul className="tag-notes">
          {notes.map((note) => (
            <li key={note.path}>
              <button
                type="button"
                className={`tag-note${note.path === current ? " on" : ""}`}
                onClick={() => onNavigate(note.path)}
                title={note.path}
              >
                {note.title}
              </button>
            </li>
          ))}
        </ul>
      </div>
    );
  }
  return (
    <div className="tag-cloud">
      {tags.slice(0, 40).map((tag) => (
        <button
          type="button"
          key={tag.tag}
          className="tag-chip"
          onClick={() => onSelectTag(tag.tag)}
          title={`${tag.count} notes`}
        >
          {tag.tag} <span className="tag-count tnum">{tag.count}</span>
        </button>
      ))}
    </div>
  );
}
