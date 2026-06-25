# Paper Recall Operations

## Public URLs

- DM landing page: `https://paper-recall-kit.pages.dev`
- Prompt file: `https://paper-recall-kit.pages.dev/prompts/research-wiki-design-prompt.md`
- Sample kit zip: `https://paper-recall-kit.pages.dev/obsidian-paper-recall-kit.zip`
- Optional support link: `https://ko-fi.com/phdailog`

Use only the landing page URL in HeyDM.

## Cloudflare

- Account: configured in the local Cloudflare profile. Do not commit account
  emails, account IDs, tokens, or `.wrangler/` cache files.
- Pages project: `paper-recall-kit`
- Production domain: `paper-recall-kit.pages.dev`
- Deployment type: Direct Upload, no Git provider.

## Update Flow

1. Edit `prompts/research-wiki-design-prompt.md`.
2. Sync the landing page copy blocks from the prompt source:

   ```bash
   python3 scripts/sync_prompts.py
   ```

3. Rebuild the downloadable zip with UTF-8/NFC filenames:

   ```bash
   python3 scripts/build_zip.py
   ```

4. Deploy:

   ```bash
stage=$(mktemp -d /tmp/paper-recall-pages.XXXXXX)
rsync -a --delete \
  --exclude '.wrangler' \
  --exclude '.DS_Store' \
  --exclude '.git' \
  --exclude 'OPERATIONS.md' \
  --exclude 'scripts' \
  ./ "$stage"/
npx wrangler pages deploy "$stage" --project-name paper-recall-kit --branch main --commit-dirty=true
trash "$stage"
   ```

5. Verify:

   ```bash
   curl -I https://paper-recall-kit.pages.dev/
   curl -I https://paper-recall-kit.pages.dev/obsidian-paper-recall-kit.zip
   curl -I https://paper-recall-kit.pages.dev/prompts/research-wiki-design-prompt.md
   curl -I https://paper-recall-kit.pages.dev/.wrangler/cache/pages.json
   ```

## Link Privacy

This is an unlisted public link, not an authenticated private download. Search exposure is reduced with:

- `robots.txt`
- `<meta name="robots" content="noindex, nofollow">`
- `_headers` with `X-Robots-Tag: noindex, nofollow, noarchive`

Anyone with the link can still open it. If true access control becomes necessary, use Cloudflare Access, one-time token links, or an email/DM gate. Do not add that friction for the current free Instagram DM flow unless abuse appears.
