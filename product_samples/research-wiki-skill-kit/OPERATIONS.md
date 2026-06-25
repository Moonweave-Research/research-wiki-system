# Research Wiki Skill Kit Operations

This directory is the durable source for the DM free distribution page:

https://research-wiki-skill-kit.pages.dev/

## Rebuild package

```bash
python3 scripts/build_zip.py
unzip -l downloads/obsidian-research-wiki-skill-prompt-kit.zip
```

Expected ZIP contents include `LICENSE.md` and must not include `.git`, `.env`, `.wrangler`, `scripts`, or `source`.

## Deploy

```bash
stage=$(mktemp -d /tmp/research-wiki-skill-pages.XXXXXX)
rsync -a --delete \
  --exclude '.wrangler' \
  --exclude '.DS_Store' \
  --exclude '.git' \
  --exclude 'OPERATIONS.md' \
  --exclude 'scripts' \
  --exclude 'source' \
  ./ "$stage"/
npx wrangler pages deploy "$stage" --project-name research-wiki-skill-kit --branch main --commit-dirty=true
trash "$stage"
```

## Verify

```bash
curl -I https://research-wiki-skill-kit.pages.dev/
curl -I https://research-wiki-skill-kit.pages.dev/downloads/PROMPT_KIT.md
curl -I https://research-wiki-skill-kit.pages.dev/downloads/obsidian-research-wiki-skill-prompt-kit.zip
curl -I https://research-wiki-skill-kit.pages.dev/LICENSE.md
curl -I https://research-wiki-skill-kit.pages.dev/.wrangler/cache/pages.json
curl -I https://research-wiki-skill-kit.pages.dev/.git/config
curl -I https://research-wiki-skill-kit.pages.dev/.env
curl -I https://research-wiki-skill-kit.pages.dev/_headers
curl -I https://research-wiki-skill-kit.pages.dev/_redirects
```

Required live checks:

- Public DM page returns 200.
- ZIP returns `Cache-Control: no-store` and `Content-Disposition: attachment`.
- Global pages return `X-Robots-Tag: noindex, nofollow, noarchive`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and `X-Content-Type-Options: nosniff`.
- Hidden or operational files return 404.
