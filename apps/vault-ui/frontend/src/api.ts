export interface TreeNode {
  name: string;
  children: TreeNode[];
  notes: { path: string; title: string }[];
}
export interface TreeResponse {
  stale: boolean;
  tree: TreeNode;
}
export interface NoteResponse {
  path: string;
  title: string;
  frontmatter: Record<string, unknown>;
  markdown: string;
  link_resolution: Record<string, string | null>;
  mtime: number;
}
export interface Backlink {
  src_path: string;
  src_title: string;
}
export interface BacklinksResponse {
  stale: boolean;
  backlinks: Backlink[];
}
export interface Health {
  status: string;
  stale: boolean;
  notes: number;
  links: number;
}
export interface RelatedHit {
  path: string;
  title: string;
  score: number;
}
export interface RelatedResponse {
  stale: boolean;
  related: RelatedHit[];
}
export interface GraphNode {
  id: string;
  title: string;
  folder: string;
  current: boolean;
  degree?: number;
  community?: number | null;
  x?: number;
  y?: number;
}
export interface GraphLink {
  source: string;
  target: string;
  kind: "link" | "similar";
}
export interface GraphResponse {
  stale: boolean;
  nodes: GraphNode[];
  links: GraphLink[];
  truncated?: boolean;
  layout?: { algorithm: string; deterministic: boolean };
}
export interface ConceptAuditSummary {
  nodes: number;
  research: number;
  concept_linked: number;
  zero_concept: number;
  isolated: number;
  isolation_percent: number;
}
export interface ConceptAuditOverlap {
  concept_paper_rows: number;
  unique_edges: number;
  duplicate_rows: number;
  row_avg_concepts_per_paper: number;
  unique_avg_concepts_per_paper: number;
  row_multi_concept_percent: number;
  unique_multi_concept_percent: number;
}
export interface ConceptAuditCommunities {
  wikilink_communities: number;
  wikilink_modularity: number | null;
  overlay_communities: number;
  overlay_modularity: number | null;
  communities_db_count: number;
  largest_community: number;
}
export interface ConceptAuditRule {
  target_slug: string;
  target: string;
  required_tags: string[];
  suppressed_parent_slugs: string[];
  qualifying_count: number;
  routed_count: number;
  target_reused_or_created: string;
  suppressed_parent_counts: Record<string, number>;
}
export interface ConceptAuditPaperHotspot {
  path: string;
  title: string;
  row_concepts: number;
  unique_concepts: number;
  concepts: string[];
}
export interface ConceptAuditTagPair {
  tags: string[];
  normalized_tags: string[];
  count: number;
  samples: SearchHit[];
}
export interface ConceptAuditTopConcept {
  concept: string;
  title: string;
  papers: number;
}
export interface ConceptAuditResponse {
  summary: ConceptAuditSummary;
  overlap: ConceptAuditOverlap;
  communities: ConceptAuditCommunities;
  rules: ConceptAuditRule[];
  hotspots: {
    top_concepts: ConceptAuditTopConcept[];
    top_overlapping_papers: ConceptAuditPaperHotspot[];
    uncovered_tag_pairs: ConceptAuditTagPair[];
  };
}

export interface SearchHit {
  path: string;
  title: string;
}
export interface SearchResponse {
  stale: boolean;
  results: SearchHit[];
}
export interface TagCount {
  tag: string;
  count: number;
}
export interface TagsResponse {
  stale: boolean;
  tags: TagCount[];
}
export interface NotesByTagResponse {
  stale: boolean;
  notes: SearchHit[];
}

async function getJSON<T>(url: string): Promise<T> {
  // no-store: the vault mutates (edits, status flips, external syncs), so a cached
  // GET would show stale content/mtime and silently bypass the save conflict guard.
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const api = {
  health: () => getJSON<Health>("/api/health"),
  tree: () => getJSON<TreeResponse>("/api/tree"),
  note: (path: string) => getJSON<NoteResponse>(`/api/note?path=${encodeURIComponent(path)}`),
  saveNote: (path: string, body: string, baseMtime?: number) =>
    postJSON<{ path: string; ok: boolean; index_stale: boolean }>("/api/note", {
      path,
      body,
      base_mtime: baseMtime,
    }),
  setStatus: (path: string, status: string) =>
    postJSON<{ path: string; status: string; ok: boolean; index_stale: boolean }>(
      "/api/note/status",
      { path, status },
    ),
  createNote: (title: string, folder = "general") =>
    postJSON<{ path: string; ok: boolean; index_stale: boolean }>("/api/note/create", {
      title,
      folder,
    }),
  backlinks: (path: string) =>
    getJSON<BacklinksResponse>(`/api/backlinks?path=${encodeURIComponent(path)}`),
  graph: (path: string, depth: number, includeSimilar: boolean) =>
    getJSON<GraphResponse>(
      `/api/graph?path=${encodeURIComponent(path)}&depth=${depth}&include_similar=${includeSimilar}`,
    ),
  graphGlobal: (current: string | null, includeSimilar: boolean) => {
    const params = new URLSearchParams();
    if (current) params.set("current", current);
    params.set("include_similar", String(includeSimilar));
    return getJSON<GraphResponse>(`/api/graph/global?${params.toString()}`);
  },
  conceptAudit: () => getJSON<ConceptAuditResponse>("/api/concept-audit"),
  search: (q: string) =>
    getJSON<SearchResponse>(`/api/search?q=${encodeURIComponent(q)}`),
  related: (path: string) =>
    getJSON<RelatedResponse>(`/api/related?path=${encodeURIComponent(path)}`),
  tags: () => getJSON<TagsResponse>("/api/tags"),
  notesByTag: (tag: string) =>
    getJSON<NotesByTagResponse>(`/api/notes-by-tag?tag=${encodeURIComponent(tag)}`),
};
