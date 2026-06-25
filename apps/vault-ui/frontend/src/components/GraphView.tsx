import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";

import type { GraphResponse } from "../api.ts";

const NEON_DIM = "rgba(94,234,212,0.5)";
const EDGE = "rgba(127,199,240,0.18)";
const EDGE_DIM = "rgba(127,199,240,0.05)";
// Similar (relatedness) edges: --neon-2 (56,189,248) at lower opacity, dashed.
const SIMILAR_EDGE = "rgba(56,189,248,0.14)";
const SIMILAR_EDGE_DIM = "rgba(56,189,248,0.04)";
// Global has ~1169 edges; at normal opacity they stack into a cyan wash that hides
// the community-colored nodes, so global edges are a faint web (hover still lights).
const GLOBAL_EDGE = "rgba(127,199,240,0.06)";
const GLOBAL_SIMILAR_EDGE = "rgba(56,189,248,0.05)";
const CURRENT = "#aef7e8";
const DIM_NODE = "rgba(255,255,255,0.08)";

// Read CSS token values once from :root so we never hardcode hex outside styles.css.
function readToken(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

type FGNode = GraphResponse["nodes"][number] & {
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number;
  fy?: number;
};
type FGLink = GraphResponse["links"][number] & { source: unknown; target: unknown };
type CaptureStep = "idle" | "global-bloom" | "core-collapse" | "local-expand" | "similar-burst" | "final-map";
type CaptureStats = { notes?: number; links?: number };

// Custom d3 force that pulls each node toward a fixed ring anchor for its community,
// so the major Leiden clusters spread into separate lobes (the default charge+link
// layout collapses them into one ball). Only the largest communities are anchored;
// the long tail of tiny communities floats by charge/link. Dependency-free.
function clusterForce(anchors: Map<number, { x: number; y: number }>, strength: number) {
  let nodes: FGNode[] = [];
  const force = (alpha: number) => {
    for (const n of nodes) {
      const a = n.community == null ? undefined : anchors.get(n.community);
      if (!a) continue;
      n.vx = (n.vx ?? 0) + (a.x - (n.x ?? 0)) * strength * alpha;
      n.vy = (n.vy ?? 0) + (a.y - (n.y ?? 0)) * strength * alpha;
    }
  };
  force.initialize = (input: FGNode[]) => {
    nodes = input;
  };
  return force;
}

// Gentle gravity toward the origin — keeps the composition cohesive and centered so
// nodes don't drift into a lopsided sprawl. Dependency-free.
function gravityForce(strength: number) {
  let nodes: FGNode[] = [];
  const force = (alpha: number) => {
    for (const n of nodes) {
      n.vx = (n.vx ?? 0) - (n.x ?? 0) * strength * alpha;
      n.vy = (n.vy ?? 0) - (n.y ?? 0) * strength * alpha;
    }
  };
  force.initialize = (input: FGNode[]) => {
    nodes = input;
  };
  return force;
}

function nodeVisualRadius(scope: "local" | "global", n: FGNode): number {
  if (scope === "global") return Math.min(4 + Math.sqrt(n.degree ?? 0) * 1.3, 12);
  return n.current ? 6 : 4;
}

function nodeCollisionRadius(scope: "local" | "global", n: FGNode): number {
  return nodeVisualRadius(scope, n) + (scope === "global" ? 7 : 4);
}

function collisionForce(radius: (node: FGNode) => number, strength: number) {
  let nodes: FGNode[] = [];
  let radii = new Map<FGNode, number>();
  let cellSize = 1;

  const force = (alpha: number) => {
    const grid = new Map<string, FGNode[]>();
    const key = (x: number, y: number) => `${Math.floor(x / cellSize)}:${Math.floor(y / cellSize)}`;

    nodes.forEach((n, index) => {
      let x = n.x ?? 0;
      let y = n.y ?? 0;
      const cx = Math.floor(x / cellSize);
      const cy = Math.floor(y / cellSize);
      const r = radii.get(n) ?? radius(n);

      for (let gx = cx - 1; gx <= cx + 1; gx += 1) {
        for (let gy = cy - 1; gy <= cy + 1; gy += 1) {
          for (const other of grid.get(`${gx}:${gy}`) ?? []) {
            let dx = x - (other.x ?? 0);
            let dy = y - (other.y ?? 0);
            if (dx === 0 && dy === 0) {
              const angle = (index + 1) * 2.399963;
              dx = Math.cos(angle) * 0.01;
              dy = Math.sin(angle) * 0.01;
            }
            const minDistance = r + (radii.get(other) ?? radius(other));
            const distanceSq = dx * dx + dy * dy;
            if (distanceSq >= minDistance * minDistance) continue;

            const distance = Math.sqrt(distanceSq);
            const push = ((minDistance - distance) / distance) * strength * alpha * 0.5;
            const vx = dx * push;
            const vy = dy * push;
            n.vx = (n.vx ?? 0) + vx;
            n.vy = (n.vy ?? 0) + vy;
            other.vx = (other.vx ?? 0) - vx;
            other.vy = (other.vy ?? 0) - vy;
            x += vx;
            y += vy;
          }
        }
      }

      const bucket = key(x, y);
      if (!grid.has(bucket)) grid.set(bucket, []);
      grid.get(bucket)!.push(n);
    });
  };
  force.initialize = (input: FGNode[]) => {
    nodes = input;
    radii = new Map(nodes.map((node) => [node, radius(node)]));
    const maxRadius = radii.size > 0 ? Math.max(...radii.values()) : 1;
    cellSize = Math.max(1, maxRadius * 2);
  };
  return force;
}

function linkKind(link: unknown): "link" | "similar" {
  return (link as FGLink).kind === "similar" ? "similar" : "link";
}

// Normalize a force-graph endpoint which may have been mutated to an object.
function nodeId(endpoint: unknown): string {
  if (endpoint && typeof endpoint === "object" && "id" in endpoint) {
    return (endpoint as FGNode).id;
  }
  return endpoint as string;
}

function captureSpotlightPulse(): number {
  return (Date.now() % 1800) / 1800;
}

function stableLinkOffset(link: unknown): number {
  const key = `${nodeId((link as FGLink).source)}:${nodeId((link as FGLink).target)}:${linkKind(link)}`;
  let hash = 0;
  for (let index = 0; index < key.length; index += 1) {
    hash = (hash * 31 + key.charCodeAt(index)) >>> 0;
  }
  return (hash % 1000) / 1000;
}

function captureRelayPhase(duration = 3200): number {
  return (Date.now() % duration) / duration;
}

function captureLinkProgress(link: unknown, step: CaptureStep, currentNodeId: string | null): number {
  const src = nodeId((link as FGLink).source);
  const dst = nodeId((link as FGLink).target);
  const touchesCurrent = currentNodeId != null && (src === currentNodeId || dst === currentNodeId);
  const isSimilar = linkKind(link) === "similar";

  if (step === "final-map" && isSimilar) return 0.2;
  if (step !== "local-expand" && step !== "similar-burst") return 0;
  if (step === "local-expand" && !touchesCurrent) return 0;
  if (step === "similar-burst" && !isSimilar) return 0;

  const width = step === "similar-burst" ? 0.3 : 0.38;
  const offsetSpan = step === "similar-burst" ? 0.62 : 0.72;
  const phase = captureRelayPhase(step === "similar-burst" ? 2600 : 3400);
  const relay = (phase - stableLinkOffset(link) * offsetSpan + 1) % 1;
  if (relay > width) return 0;
  const centerDistance = Math.abs(relay / width - 0.5) * 2;
  return Math.max(0, 1 - centerDistance);
}

function drawCaptureSpotlight(ctx: CanvasRenderingContext2D, node: FGNode, radius: number, globalScale: number) {
  const x = node.x ?? 0;
  const y = node.y ?? 0;
  const phase = captureSpotlightPulse();
  const safeScale = Math.max(globalScale, 0.5);
  const rings = [
    { progress: phase, alpha: 0.36 },
    { progress: (phase + 0.48) % 1, alpha: 0.22 },
  ];
  ctx.save();
  ctx.shadowBlur = 22 / safeScale;
  ctx.shadowColor = "rgba(174,247,232,0.55)";
  for (const ring of rings) {
    const ringRadius = radius * (2.6 + ring.progress * 3.8);
    ctx.beginPath();
    ctx.arc(x, y, ringRadius, 0, 2 * Math.PI);
    ctx.strokeStyle = `rgba(174,247,232,${ring.alpha * (1 - ring.progress)})`;
    ctx.lineWidth = 1.4 / safeScale;
    ctx.stroke();
  }
  ctx.restore();
}

function formatCount(value: number | undefined): string {
  return typeof value === "number" ? value.toLocaleString() : "—";
}

function captureLabel(step: CaptureStep): { kicker: string; title: string; detail: string } {
  switch (step) {
    case "global-bloom":
      return {
        kicker: "GLOBAL MAP",
        title: "A research vault becomes a living map.",
        detail: "Every cluster is a topic family.",
      };
    case "core-collapse":
      return {
        kicker: "CORE VIEW",
        title: "Noise collapses into structure.",
        detail: "Low-degree satellites fade so the backbone reads.",
      };
    case "local-expand":
      return {
        kicker: "ACTUATOR",
        title: "One concept opens its paper neighborhood.",
        detail: "semantic relay traces the surrounding research trail",
      };
    case "similar-burst":
      return {
        kicker: "SEMANTIC EDGE BURST",
        title: "Similar papers light up beyond manual links.",
        detail: "semantic edge burst",
      };
    case "final-map":
      return {
        kicker: "LIVING INDEX",
        title: "Turn scattered papers into a working research map.",
        detail: "search, notes, backlinks, and semantic edges stay connected",
      };
    default:
      return {
        kicker: "RESEARCH VAULT",
        title: "Knowledge graph capture mode.",
        detail: "Ready to record.",
      };
  }
}

export default function GraphView({
  data,
  loading,
  onNavigate,
  depth,
  onDepthChange,
  scope,
  onScopeChange,
  similar,
  onSimilarChange,
  captureMode = false,
  onCaptureModeChange,
  captureStep = "idle",
  captureStats,
}: {
  data: GraphResponse | null;
  loading: boolean;
  onNavigate: (path: string) => void;
  depth: number;
  onDepthChange: (d: number) => void;
  scope: "local" | "global";
  onScopeChange: (s: "local" | "global") => void;
  similar: boolean;
  onSimilarChange: (v: boolean) => void;
  captureMode?: boolean;
  onCaptureModeChange?: (enabled: boolean) => void;
  captureStep?: CaptureStep;
  captureStats?: CaptureStats;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fg = useRef<ForceGraphMethods<any, any> | undefined>(undefined);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState<string | null>(null);
  const [coreOnly, setCoreOnly] = useState(false);

  // CSS token values (read once, stable across renders).
  const tokenNeon = useMemo(() => readToken("--neon"), []);
  const tokenNeon2 = useMemo(() => readToken("--neon-2"), []);
  const tokenGreen = useMemo(() => readToken("--green"), []);
  const tokenBlue = useMemo(() => readToken("--blue"), []);

  const fitGraph = useCallback(
    (duration = 500, padding = 80) => {
      fg.current?.zoomToFit(
        duration,
        padding,
        (node) => scope !== "global" || ((node as FGNode).degree ?? 0) > 1,
      );
    },
    [scope],
  );

  useEffect(() => {
    if (!wrap.current) return;
    const el = wrap.current;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (scope !== "global" && coreOnly) setCoreOnly(false);
  }, [coreOnly, scope]);

  useEffect(() => {
    setCoreOnly(captureMode && (captureStep === "core-collapse" || captureStep === "final-map"));
  }, [captureMode, captureStep]);

  useEffect(() => {
    setHover(null);
  }, [scope, depth, similar, coreOnly]);

  const graphData = useMemo(() => {
    const nodes = (data?.nodes ?? []).map((node) => {
      return { ...node } as FGNode;
    });
    const visibleNodes =
      scope === "global" && coreOnly ? nodes.filter((node) => (node.degree ?? 0) > 1) : nodes;
    const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
    const links =
      scope === "global" && coreOnly
        ? (data?.links ?? []).filter(
            (link) => visibleNodeIds.has(nodeId(link.source)) && visibleNodeIds.has(nodeId(link.target)),
          )
        : (data?.links ?? []);

    return {
      nodes: visibleNodes,
      links,
    };
  }, [coreOnly, data, scope]);

  useEffect(() => {
    const inst = fg.current;
    if (!inst) return;
    inst
      .d3Force("link")
      ?.distance(scope === "global" ? 62 : 52)
      .strength(scope === "global" ? 0.04 : 0.16);
    // Gentle gravity keeps the whole composition centered and cohesive.
    inst.d3Force("gravity", gravityForce(scope === "global" ? 0.02 : 0.09));
    inst.d3Force(
      "collide",
      collisionForce((node) => nodeCollisionRadius(scope, node), scope === "global" ? 1.2 : 0.55),
    );

    const nodes = graphData.nodes as FGNode[];
    const sizes = new Map<number, number>();
    for (const n of nodes) {
      if (n.community != null) sizes.set(n.community, (sizes.get(n.community) ?? 0) + 1);
    }
    const major = [...sizes.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12);
    if (scope === "global" && major.length > 1) {
      // Spread the major communities onto a ring; each lobe's members are pulled to
      // its anchor so same-color clusters separate instead of forming one ball.
      const radius = 24 * Math.sqrt(nodes.length);
      const anchors = new Map<number, { x: number; y: number }>();
      major.forEach(([community], index) => {
        const angle = (index / major.length) * 2 * Math.PI;
        anchors.set(community, { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius });
      });
      inst.d3Force("charge")?.strength(-380);
      inst.d3Force("cluster", clusterForce(anchors, 0.13));
    } else {
      inst.d3Force("charge")?.strength(-180);
      inst.d3Force("cluster", null);
    }
    inst.d3ReheatSimulation?.();
  }, [graphData.nodes, scope]);

  useEffect(() => {
    if (!data?.layout?.deterministic || graphData.nodes.length === 0) return;
    const timer = window.setTimeout(() => fitGraph(500, 80), 100);
    return () => window.clearTimeout(timer);
  }, [data?.layout?.deterministic, fitGraph, graphData.nodes.length, scope, size.h, size.w]);

  // Build adjacency map from raw data, normalizing mutated endpoints.
  const adjacency = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const link of graphData.links) {
      const src = nodeId(link.source);
      const dst = nodeId(link.target);
      if (!map.has(src)) map.set(src, new Set());
      if (!map.has(dst)) map.set(dst, new Set());
      map.get(src)!.add(dst);
      map.get(dst)!.add(src);
    }
    return map;
  }, [graphData]);

  // Active set: hovered node + its neighbors.
  const activeSet = useMemo(() => {
    if (!hover) return null;
    const s = new Set<string>([hover]);
    for (const nb of adjacency.get(hover) ?? []) s.add(nb);
    return s;
  }, [hover, adjacency]);

  const currentNodeId = useMemo(() => graphData.nodes.find((node) => node.current)?.id ?? null, [graphData.nodes]);

  function touchesCurrentNode(link: unknown): boolean {
    if (!currentNodeId) return false;
    const src = nodeId((link as FGLink).source);
    const dst = nodeId((link as FGLink).target);
    return src === currentNodeId || dst === currentNodeId;
  }

  function captureLinkParticleCount(link: unknown): number {
    if (!captureMode) return 0;
    const progress = captureLinkProgress(link, captureStep, currentNodeId);
    if (captureStep === "local-expand" && touchesCurrentNode(link)) return progress > 0.08 ? 2 : 1;
    if (captureStep === "similar-burst" && linkKind(link) === "similar") return progress > 0.08 ? 2 : 1;
    if (captureStep === "final-map" && linkKind(link) === "similar") return progress > 0.18 ? 1 : 0;
    return 0;
  }

  // Map top-level folder name -> tint color (using CSS tokens).
  function folderColor(folder: string, isCurrent: boolean): string {
    if (isCurrent) return CURRENT;
    const palette: Record<string, string> = {
      research: tokenNeon,
      concepts: tokenNeon2,
      general: tokenGreen,
      meta: tokenBlue,
    };
    return palette[folder] ?? NEON_DIM;
  }

  // Global graph colors by Leiden community so clusters are legible; golden-angle
  // hue spread keeps adjacent communities distinct. Local stays folder-tinted.
  function communityColor(community: number | null | undefined, isCurrent: boolean): string {
    if (isCurrent) return CURRENT;
    if (community == null) return NEON_DIM;
    return `hsl(${Math.round((community * 137.508) % 360)}, 85%, 67%)`;
  }

  function nodeColor(n: FGNode): string {
    return scope === "global" ? communityColor(n.community, n.current) : folderColor(n.folder, n.current);
  }

  return (
    <div
      ref={wrap}
      className={`graphwrap${captureMode ? ` capture-graph capture-step-${captureStep}` : ""}`}
    >
      {loading && <div className="empty graph-empty">Loading graph…</div>}
      {!loading && graphData.nodes.length === 0 && (
        <div className="empty graph-empty">No connections to graph for this note.</div>
      )}
      {!loading && graphData.nodes.length > 0 && size.w > 0 && (
        <ForceGraph2D
          ref={fg}
          width={size.w}
          height={size.h}
          graphData={graphData}
          backgroundColor="rgba(0,0,0,0)"
          nodeRelSize={5}
          linkColor={(link) => {
            const isSimilar = linkKind(link) === "similar";
            const relayProgress = captureMode ? captureLinkProgress(link, captureStep, currentNodeId) : 0;
            if (relayProgress > 0 && captureStep === "local-expand") {
              return `rgba(174,247,232,${0.18 + relayProgress * 0.36})`;
            }
            if (relayProgress > 0 && (captureStep === "similar-burst" || captureStep === "final-map") && isSimilar) {
              return `rgba(56,189,248,${0.12 + relayProgress * 0.34})`;
            }
            const lit =
              scope === "global"
                ? isSimilar
                  ? GLOBAL_SIMILAR_EDGE
                  : GLOBAL_EDGE
                : isSimilar
                  ? SIMILAR_EDGE
                  : EDGE;
            const dim = isSimilar ? SIMILAR_EDGE_DIM : EDGE_DIM;
            if (!activeSet) return lit;
            const src = nodeId(link.source);
            const dst = nodeId(link.target);
            return activeSet.has(src) && activeSet.has(dst) ? lit : dim;
          }}
          linkLineDash={(link) => (linkKind(link) === "similar" ? [4, 3] : null)}
          linkWidth={(link) =>
            captureMode ? 0.8 + captureLinkProgress(link, captureStep, currentNodeId) * 0.8 : 1
          }
          linkDirectionalParticles={captureLinkParticleCount}
          linkDirectionalParticleSpeed={(link) =>
            captureStep === "similar-burst" && linkKind(link) === "similar" ? 0.011 : 0.007
          }
          linkDirectionalParticleWidth={(link) =>
            1.4 + (captureMode ? captureLinkProgress(link, captureStep, currentNodeId) * 1.6 : 0)
          }
          linkDirectionalParticleColor={(link) =>
            linkKind(link) === "similar" ? "rgba(56,189,248,0.82)" : "rgba(174,247,232,0.72)"
          }
          // Aesthetic physics: more friction (smoother, less jittery) + slower cooling
          // (a graceful settle/bloom rather than a snap), and a longer run before rest.
          d3VelocityDecay={0.5}
          d3AlphaDecay={0.0185}
          cooldownTicks={280}
          onEngineStop={() => fitGraph(700, 80)}
          onNodeHover={(n) => setHover(n ? (n as FGNode).id : null)}
          onNodeClick={(n) => onNavigate((n as FGNode).id)}
          onNodeDragEnd={(node) => {
            const n = node as FGNode;
            n.fx = n.x;
            n.fy = n.y;
          }}
          nodePointerAreaPaint={(node, color, ctx) => {
            const n = node as FGNode;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(n.x ?? 0, n.y ?? 0, n.current ? 10 : 7, 0, 2 * Math.PI);
            ctx.fill();
          }}
          nodeCanvasObject={(node, ctx, globalScale) => {
            const n = node as FGNode;
            const isActive = !activeSet || activeSet.has(n.id);
            const showLabel = captureMode
              ? n.current || n.id === hover || (scope === "global" && (n.degree ?? 0) >= 22)
              : scope === "local"
                ? n.current || n.id === hover
                : n.id === hover;
            const r = nodeVisualRadius(scope, n);
            const color = nodeColor(n);
            ctx.globalAlpha = isActive ? 1 : 0.15;
            ctx.shadowColor = color;
            ctx.shadowBlur = n.current ? 20 : n.id === hover ? 16 : 7;
            if (captureMode && n.current) {
              drawCaptureSpotlight(ctx, n, r, globalScale);
            }
            ctx.beginPath();
            ctx.arc(n.x ?? 0, n.y ?? 0, r, 0, 2 * Math.PI);
            ctx.fillStyle = isActive ? color : DIM_NODE;
            ctx.fill();
            ctx.shadowBlur = 0;
            if (showLabel) {
              const label = n.title.length > 32 ? `${n.title.slice(0, 31)}…` : n.title;
              const safeScale = Math.max(globalScale, 0.25);
              const fontSize = (n.current ? 12 : 11) / safeScale;
              const maxLabelWidth = (scope === "local" ? 180 : 150) / safeScale;
              ctx.font = `${fontSize}px 'Hanken Grotesk', sans-serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              ctx.fillStyle = n.current ? CURRENT : "rgba(233,234,238,0.92)";
              ctx.fillText(label, n.x ?? 0, (n.y ?? 0) + r + 4 / safeScale, maxLabelWidth);
            }
            ctx.globalAlpha = 1;
          }}
        />
      )}
      <div className="graph-controls">
        <div className="graph-depth-seg" role="group" aria-label="Graph scope">
          {(["local", "global"] as const).map((s) => (
            <button
              key={s}
              type="button"
              className={scope === s ? "on" : ""}
              aria-pressed={scope === s}
              onClick={() => onScopeChange(s)}
            >
              {s === "local" ? "Local" : "Global"}
            </button>
          ))}
        </div>
        {scope === "local" && (
          <div className="graph-depth-seg" role="group" aria-label="Graph depth">
            {([1, 2, 3] as const).map((d) => (
              <button
                key={d}
                type="button"
                className={depth === d ? "on" : ""}
                aria-pressed={depth === d}
                onClick={() => onDepthChange(d)}
              >
                {d}
              </button>
            ))}
          </div>
        )}
        {scope === "global" && (
          <button
            type="button"
            className={`graph-similar-btn graph-core-btn${coreOnly ? " on" : ""}`}
            aria-pressed={coreOnly}
            onClick={() => setCoreOnly(!coreOnly)}
          >
            Core
          </button>
        )}
        <button
          type="button"
          className={`graph-similar-btn${similar ? " on" : ""}`}
          aria-pressed={similar}
          onClick={() => onSimilarChange(!similar)}
        >
          Similar
        </button>
        <button
          type="button"
          className="graph-fit-btn"
          aria-label="Zoom to fit"
          onClick={() => fitGraph(400, 70)}
        >
          ⊡
        </button>
        <button
          type="button"
          className="graph-similar-btn graph-capture-btn"
          aria-label="Enter capture mode"
          onClick={() => onCaptureModeChange?.(true)}
        >
          Capture
        </button>
      </div>
      <div className="graph-legend" aria-hidden="true">
        {scope === "global" ? (
          <>
            <span style={{ color: tokenNeon }}>● clusters</span>
            {coreOnly && <span style={{ color: tokenNeon2 }}>● core view</span>}
          </>
        ) : (
          <>
            <span style={{ color: tokenNeon }}>● research</span>
            <span style={{ color: tokenNeon2 }}>● concepts</span>
          </>
        )}
        {similar && <span style={{ color: tokenNeon2 }}>┄ similar</span>}
      </div>
      {captureMode && (
        <>
          <div key={`hud-${captureStep}`} className="reels-hud" aria-hidden="true">
            <span>{captureLabel(captureStep).kicker}</span>
            <strong>{formatCount(captureStats?.notes)} notes</strong>
            <strong>{formatCount(captureStats?.links)} links</strong>
            <em>
              {graphData.nodes.length.toLocaleString()} shown · {graphData.links.length.toLocaleString()} edges
            </em>
          </div>
          <div key={`caption-${captureStep}`} className="reels-caption" aria-hidden="true">
            <span>{captureLabel(captureStep).title}</span>
            <em>{captureLabel(captureStep).detail}</em>
          </div>
          <div className="reels-progress" aria-hidden="true">
            {["global-bloom", "core-collapse", "local-expand", "similar-burst", "final-map"].map(
              (step) => (
                <span key={step} className={captureStep === step ? "on" : ""} />
              ),
            )}
          </div>
          <button type="button" className="reels-exit" onClick={() => onCaptureModeChange?.(false)}>
            Exit
          </button>
        </>
      )}
    </div>
  );
}
