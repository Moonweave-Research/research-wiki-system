# research-wiki-system export candidate

This is a local dry-run export of the reusable research-wiki system code.

It intentionally excludes the private vault corpus:

- `wiki/`
- Zotero/BetterBibTeX exports
- local PDF/library attachments
- personal Obsidian state
- generated indexes such as `.graph_index/`
- local runtime/deploy caches such as `.omx/`, `.run/`, and `.wrangler/`

Use `tests/fixtures/sample-vault/` as the public-safe sample vault while testing
system behavior outside the private observatory repository.
