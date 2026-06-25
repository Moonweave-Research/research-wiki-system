"""vault-ui API routes (read-only graph + a guarded note-body write)."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from . import services

router = APIRouter()


class NoteSave(BaseModel):
    path: str
    body: str
    base_mtime: float | None = None


class NoteStatus(BaseModel):
    path: str
    status: str


class NoteCreate(BaseModel):
    title: str
    folder: str = "general"


@router.get("/health")
def health() -> dict:
    return services.health()


@router.get("/tree")
def tree() -> dict:
    return services.build_tree()


@router.get("/note")
def note(path: str = Query(..., description="vault-relative path, e.g. research/foo.md")) -> dict:
    result = services.get_note(path)
    if result is None:
        raise HTTPException(status_code=404, detail=f"note not found: {path}")
    return result


@router.post("/note")
def save_note(payload: NoteSave) -> dict:
    try:
        result = services.save_note(payload.path, payload.body, base_mtime=payload.base_mtime)
    except services.ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"note not found: {payload.path}")
    return result


@router.post("/note/status")
def set_status(payload: NoteStatus) -> dict:
    try:
        result = services.set_status(payload.path, payload.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"note not found: {payload.path}")
    return result


@router.post("/note/create")
def create_note(payload: NoteCreate) -> dict:
    try:
        return services.create_note(payload.title, payload.folder)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/backlinks")
def backlinks(path: str = Query(..., description="vault-relative note path")) -> dict:
    return services.get_backlinks(path)


@router.get("/search")
def search(q: str = Query(""), limit: int = Query(30, ge=1, le=100)) -> dict:
    return services.search_notes(q, limit)


@router.get("/tags")
def tags() -> dict:
    return services.list_tags()


@router.get("/notes-by-tag")
def notes_by_tag(tag: str = Query(..., description="exact tag")) -> dict:
    return services.notes_by_tag(tag)


@router.get("/graph")
def graph(
    path: str = Query(..., description="vault-relative note path"),
    depth: int = Query(1, ge=1, le=3, description="neighborhood hops"),
    include_similar: bool = Query(False, description="add relatedness-index edges"),
) -> dict:
    return services.get_graph(path, depth, include_similar)


@router.get("/graph/global")
def graph_global(
    current: str | None = Query(None, description="note to flag as current"),
    include_similar: bool = Query(False, description="add relatedness-index edges"),
) -> dict:
    return services.get_global_graph(current, include_similar)


@router.get("/concept-audit")
def concept_audit() -> dict:
    return services.get_concept_audit()


@router.get("/related")
def related(
    path: str = Query(..., description="vault-relative note path"),
    limit: int = Query(8, ge=1, le=30),
) -> dict:
    return services.get_related(path, limit)
