import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.responses import Response
from fastapi.staticfiles import StaticFiles

from .routes import router


class VaultStaticFiles(StaticFiles):
    async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1000})
            return
        await super().__call__(scope, receive, send)

    async def get_response(self, path: str, scope) -> Response:  # type: ignore[no-untyped-def]
        response = await super().get_response(path, scope)
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("text/html"):
            response.headers["cache-control"] = "no-cache"
        return response


def mount_spa(app: FastAPI, static_dir: str | None) -> bool:
    """Serve the built frontend bundle at / when static_dir is a real directory,
    so a single uvicorn process serves both the SPA and /api (no dev server / proxy
    needed in the always-on launchd deployment). Returns whether it mounted."""
    if static_dir and Path(static_dir).is_dir():
        app.mount("/", VaultStaticFiles(directory=static_dir, html=True), name="spa")
        return True
    return False


app = FastAPI(title="research-wiki vault-ui")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4180",
        "http://127.0.0.1:4180",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")
# Mount the SPA LAST so /api routes win; no-op in dev (env unset -> Vite serves it).
mount_spa(app, os.getenv("VAULT_UI_STATIC_DIR"))
