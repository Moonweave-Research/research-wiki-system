const blockedPrefixes = ["/scripts/", "/source/", "/.wrangler/", "/.git/"];
const blockedPaths = new Set(["/OPERATIONS.md", "/.env", "/.DS_Store"]);

export async function onRequest(context) {
  const url = new URL(context.request.url);
  if (
    blockedPaths.has(url.pathname) ||
    blockedPrefixes.some((prefix) => url.pathname.startsWith(prefix))
  ) {
    return new Response("Not found", {
      status: 404,
      headers: {
        "cache-control": "no-store",
        "x-robots-tag": "noindex, nofollow, noarchive",
      },
    });
  }

  return context.next();
}
