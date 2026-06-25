import type {
  ConceptAuditPaperHotspot,
  ConceptAuditResponse,
  ConceptAuditTagPair,
} from "../api.ts";

function fixed(value: number | null, digits = 2): string {
  return value == null ? "n/a" : value.toFixed(digits);
}

function pct(value: number): string {
  if (Number.isInteger(value)) return `${value}%`;
  return `${value.toFixed(value < 10 && value > 0 ? 1 : 0)}%`;
}

function paperLabel(paper: ConceptAuditPaperHotspot): string {
  return paper.title || paper.path;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

function hasNumber(record: Record<string, unknown>, key: string): boolean {
  return typeof record[key] === "number";
}

function isRenderableAudit(audit: unknown): audit is ConceptAuditResponse {
  if (!isRecord(audit)) return false;
  if (!isRecord(audit.summary) || !isRecord(audit.overlap)) return false;
  if (!isRecord(audit.communities) || !isRecord(audit.hotspots)) return false;
  const { summary, overlap, communities, hotspots } = audit;
  return (
    ["nodes", "isolated", "isolation_percent", "zero_concept", "concept_linked"].every((key) =>
      hasNumber(summary, key),
    ) &&
    ["unique_avg_concepts_per_paper", "unique_multi_concept_percent"].every((key) =>
      hasNumber(overlap, key),
    ) &&
    hasNumber(communities, "wikilink_communities") &&
    Array.isArray(hotspots.top_concepts) &&
    Array.isArray(hotspots.top_overlapping_papers) &&
    Array.isArray(hotspots.uncovered_tag_pairs)
  );
}

function AuditState({ children }: { children: string }) {
  return (
    <aside className="audit-panel empty" aria-label="Concept graph audit">
      {children}
    </aside>
  );
}

export default function ConceptAuditPanel({
  audit,
  loading,
  onNavigate,
}: {
  audit: ConceptAuditResponse | null;
  loading: boolean;
  onNavigate: (path: string) => void;
}) {
  if (loading) return <AuditState>Loading audit…</AuditState>;
  if (!isRenderableAudit(audit)) return <AuditState>Audit unavailable.</AuditState>;

  const overlapping = audit.hotspots.top_overlapping_papers.slice(0, 5);
  const pairs = audit.hotspots.uncovered_tag_pairs.slice(0, 4);
  const concepts = audit.hotspots.top_concepts.slice(0, 4);

  return (
    <aside className="audit-panel" aria-label="Concept graph audit">
      <div className="audit-head">
        <div>
          <div className="seclabel">Concept Audit</div>
          <div className="audit-sub">wikilinks only</div>
        </div>
        <span className="count tnum">{audit.summary.nodes}</span>
      </div>

      <div className="audit-metrics">
        <div>
          <span>Isolated</span>
          <strong>{pct(audit.summary.isolation_percent)}</strong>
          <em className="tnum">{audit.summary.isolated} nodes</em>
        </div>
        <div>
          <span>Avg / paper</span>
          <strong>{fixed(audit.overlap.unique_avg_concepts_per_paper)}</strong>
          <em className="tnum">{pct(audit.overlap.unique_multi_concept_percent)} multi</em>
        </div>
        <div>
          <span>Modularity</span>
          <strong>{fixed(audit.communities.wikilink_modularity)}</strong>
          <em className="tnum">{audit.communities.wikilink_communities} groups</em>
        </div>
        <div>
          <span>Zero concept</span>
          <strong>{audit.summary.zero_concept}</strong>
          <em className="tnum">{audit.summary.concept_linked} linked</em>
        </div>
      </div>

      <section className="audit-section">
        <h2>Broad Concepts</h2>
        <ul role="list" className="audit-list">
          {concepts.map((concept) => (
            <li key={concept.concept}>
              <span className="audit-row-title">{concept.title}</span>
              <span className="audit-row-meta tnum">{concept.papers}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="audit-section">
        <h2>Overlap</h2>
        {overlapping.length === 0 ? (
          <div className="empty">No overlap hotspots.</div>
        ) : (
          <ul role="list" className="audit-list">
            {overlapping.map((paper) => (
              <li key={paper.path}>
                <a
                  href={`#/${paper.path}`}
                  className="audit-link"
                  onClick={(event) => {
                    event.preventDefault();
                    onNavigate(paper.path);
                  }}
                >
                  <span className="audit-row-title">{paperLabel(paper)}</span>
                  <span className="audit-row-meta tnum">{paper.unique_concepts}</span>
                </a>
                <div className="audit-tags">{paper.concepts.slice(0, 4).join(" · ")}</div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="audit-section">
        <h2>Candidate Pairs</h2>
        {pairs.length === 0 ? (
          <div className="empty">No candidate pairs.</div>
        ) : (
          <ul role="list" className="audit-list">
            {pairs.map((pair) => (
              <CandidatePair key={pair.normalized_tags.join("+")} pair={pair} onNavigate={onNavigate} />
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}

function CandidatePair({
  pair,
  onNavigate,
}: {
  pair: ConceptAuditTagPair;
  onNavigate: (path: string) => void;
}) {
  const sample = pair.samples[0];
  return (
    <li>
      {sample ? (
        <a
          href={`#/${sample.path}`}
          className="audit-link"
          onClick={(event) => {
            event.preventDefault();
            onNavigate(sample.path);
          }}
        >
          <span className="audit-row-title">{pair.tags.join(" + ")}</span>
          <span className="audit-row-meta tnum">{pair.count}</span>
        </a>
      ) : (
        <div className="audit-row">
          <span className="audit-row-title">{pair.tags.join(" + ")}</span>
          <span className="audit-row-meta tnum">{pair.count}</span>
        </div>
      )}
      {sample && (
        <div className="audit-tags" translate="no">
          {sample.title}
        </div>
      )}
    </li>
  );
}
