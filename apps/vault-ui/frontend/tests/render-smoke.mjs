// Build-artifact smoke: assert `npm run build` produced a dist bundle that
// references the app entry. Run after `npm run build`.
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const dist = join(root, "dist");

function fail(msg) {
  console.error(`[smoke] FAIL: ${msg}`);
  process.exit(1);
}

if (!existsSync(join(dist, "index.html"))) fail("dist/index.html missing — run `npm run build`");
const html = readFileSync(join(dist, "index.html"), "utf8");
if (!html.includes("research")) fail("dist/index.html missing app title");
const assets = join(dist, "assets");
if (!existsSync(assets) || readdirSync(assets).filter((f) => f.endsWith(".js")).length === 0) {
  fail("no JS bundle emitted in dist/assets");
}
console.log("[smoke] OK — dist bundle present with JS assets");
