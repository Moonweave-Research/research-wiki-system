import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const source = readFileSync(join(root, "src/components/NoteViewer.tsx"), "utf8");

assert(
  !/urlTransform=\{\s*\(\s*url\s*\)\s*=>\s*url\s*\}/.test(source),
  "NoteViewer must not disable react-markdown's default URL sanitizer with an identity transform",
);

assert(
  /defaultUrlTransform/.test(source) &&
    /url\.startsWith\(WIKILINK_SCHEME\)/.test(source) &&
    /defaultUrlTransform\(url\)/.test(source),
  "NoteViewer must preserve wikilink: URLs while delegating normal Markdown URLs to defaultUrlTransform",
);

console.log("[note-viewer-contract] OK — Markdown URLs keep the default sanitizer outside wikilinks");
