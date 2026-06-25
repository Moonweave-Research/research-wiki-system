const BLOCKED_PREFIXES = [
  "/scripts/",
  "/.wrangler/",
  "/.git/",
];

const BLOCKED_PATHS = new Set([
  "/OPERATIONS.md",
  "/.DS_Store",
]);

export async function onRequest(context) {
  const { pathname } = new URL(context.request.url);
  const blocked =
    BLOCKED_PATHS.has(pathname) ||
    BLOCKED_PREFIXES.some((prefix) => pathname.startsWith(prefix));

  if (blocked) {
    return new Response("Not found", {
      status: 404,
      headers: {
        "Cache-Control": "no-store",
        "Content-Type": "text/plain; charset=utf-8",
        "X-Robots-Tag": "noindex, nofollow, noarchive",
      },
    });
  }

  return context.next();
}
