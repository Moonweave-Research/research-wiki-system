import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const source = readFileSync(join(root, "src/components/GraphView.tsx"), "utf8");
const appSource = readFileSync(join(root, "src/App.tsx"), "utf8");
const searchSource = readFileSync(join(root, "src/components/SearchModal.tsx"), "utf8");
const styles = readFileSync(join(root, "src/styles.css"), "utf8");

assert(
  !/if\s*\(\s*data\?\.layout\?\.deterministic\s*&&[\s\S]{0,220}next\.fx\s*=\s*next\.x[\s\S]{0,80}next\.fy\s*=\s*next\.y/.test(
    source,
  ),
  "deterministic graph coordinates must seed the simulation, not auto-pin global nodes with fx/fy",
);

assert(
  !/scope\s*===\s*["']local["'][\s\S]{0,220}next\.fx\s*=\s*next\.x[\s\S]{0,80}next\.fy\s*=\s*next\.y/.test(
    source,
  ),
  "local graph coordinates must also seed the simulation instead of pinning fx/fy",
);

assert(
  /d3Force\s*\(\s*["']collide["']/.test(source),
  "global graph physics must include a collision force so force motion cannot collapse nodes into overlaps",
);

assert(
  /coreOnly/.test(source) &&
    /visibleNodeIds/.test(source) &&
    /\(node\.degree \?\? 0\) > 1/.test(source) &&
    />\s*Core\s*</.test(source),
  "global graph must expose a Core toggle that filters low-degree satellite nodes without changing the default view",
);

assert(
  /setHover\(null\)/.test(source) &&
    /\[scope,\s*depth,\s*similar,\s*coreOnly\]/.test(source),
  "graph view state changes must clear stale hover highlights across Local/Global/Depth/Similar/Core transitions",
);

assert(
  /coreOnly\s*&&\s*<span/.test(source) && /core view/.test(source),
  "global Core filter state must be reflected in the graph legend for view-state consistency",
);

assert(
  /captureMode/.test(appSource) &&
    /URLSearchParams/.test(appSource) &&
    /set\(["']capture["'],\s*["']1["']\)/.test(appSource) &&
    /setCaptureMode/.test(appSource) &&
    /history\.replaceState/.test(appSource) &&
    /Escape/.test(appSource) &&
    /setGraphScope\(["']global["']\)/.test(appSource) &&
    /setGraphScope\(["']local["']\)/.test(appSource) &&
    /setGraphDepth\(2\)/.test(appSource) &&
    /setGraphSimilar\(true\)/.test(appSource) &&
    /16000/.test(appSource),
  "App must expose an isolated ?capture=1 Reels Capture Mode that auto-stages Global, Local depth 2, Similar states, and a recording-safe final hold",
);

assert(
  /tree-close/.test(appSource) &&
    /aria-label=["']Close file tree["']/.test(appSource) &&
    /setTreeOpen\(false\)/.test(appSource) &&
    /\.tree-close/.test(styles) &&
    /\.tree-close[\s\S]{0,120}display:\s*inline-flex/.test(styles),
  "mobile file tree drawer must have an internal close button because the open drawer covers the header toggle",
);

assert(
  /captureStep/.test(source) &&
    /onCaptureModeChange/.test(source) &&
    /Capture/.test(source) &&
    /Exit/.test(source) &&
    /captureSpotlightPulse/.test(source) &&
    /drawCaptureSpotlight/.test(source) &&
    /captureRelayPhase/.test(source) &&
    /captureLinkProgress/.test(source) &&
    /touchesCurrentNode/.test(source) &&
    /captureLinkParticleCount/.test(source) &&
    /linkDirectionalParticles/.test(source) &&
    /linkDirectionalParticleSpeed/.test(source) &&
    /linkDirectionalParticleColor/.test(source) &&
    /semantic relay/.test(source) &&
    /working research map/.test(source) &&
    /0\.2/.test(source) &&
    /1\.6/.test(source) &&
    /captureMode\s*\?\s*0\.8\s*\+\s*captureLinkProgress[\s\S]{0,80}:\s*1/.test(source) &&
    /reels-hud/.test(source) &&
    /reels-caption/.test(source) &&
    /semantic edge burst/.test(source) &&
    /setCoreOnly\(captureMode &&/.test(source),
  "GraphView must render capture-mode HUD/captions, concept spotlight, semantic relay timing, and meaning-based link motion without changing normal graph controls or normal link width",
);

assert(
  /\.app\.capture-mode/.test(styles) &&
    /\.reels-hud/.test(styles) &&
    /\.reels-caption/.test(styles) &&
    /captureTitleIn/.test(styles) &&
    /captureMetricIn/.test(styles) &&
    /pointer-events:\s*none/.test(styles) &&
    !/captureSweep/.test(styles) &&
    !/\.capture-graph::before[\s\S]{0,520}animation:/.test(styles) &&
    /@keyframes capturePulse/.test(styles),
  "capture mode must keep staged HUD/caption motion but avoid screen-sweeping scan overlays",
);

assert(
  /searchSeqRef/.test(searchSource) &&
    /\+\+searchSeqRef\.current/.test(searchSource) &&
    /searchSeqRef\.current\s*===\s*seq/.test(searchSource) &&
    !/cancelRef/.test(searchSource),
  "SearchModal must guard debounced search with a per-request sequence so stale responses cannot overwrite the latest query",
);
