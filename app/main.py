"""Research FastAPI service for mammogram BI-RADS proxy inference."""
import asyncio
import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mammo.config import MODELS
from mammo.inference import MammoPipeline

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 100 * 1024 * 1024
MAX_REQUEST_BODY = MAX_UPLOAD + 1024 * 1024
state: dict = {}
logger = logging.getLogger(__name__)


class RequestBodyTooLarge(Exception):
    pass


class MaxBodySizeMiddleware:
    """Reject oversized HTTP bodies before multipart parsing or spooling."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            content_length = 0
        if content_length > self.max_bytes:
            await PlainTextResponse("request body too large", status_code=413)(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await PlainTextResponse("request body too large", status_code=413)(scope, receive, send)


def safe_filename(filename: str | None) -> str:
    """Return only the final component for POSIX- or Windows-style paths."""
    return Path((filename or "upload").replace("\\", "/")).name or "upload"


async def read_upload_limited(file: UploadFile, max_bytes: int = MAX_UPLOAD) -> bytes:
    """Read at most one byte beyond the limit, without buffering an arbitrary upload."""
    data = await file.read(max_bytes + 1)
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > max_bytes:
        raise HTTPException(413, "file too large")
    return data


@asynccontextmanager
async def lifespan(_: FastAPI):
    required = (MODELS / "detector.pt", MODELS / "classifier.pt", MODELS / "screening.pt")
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"model weights not found in {MODELS}: {missing}. Train them first (see README).")
    state["pipeline"] = MammoPipeline(device=os.getenv("MAMMO_DEVICE") or None)
    state["inference_lock"] = asyncio.Lock()
    yield
    state.clear()


app = FastAPI(title="MammoAI", lifespan=lifespan)
app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_REQUEST_BODY)


@app.get("/api/health")
def health():
    return {"status": "ok", "device": state["pipeline"].device}


@app.post("/api/analyze")
async def analyze(
    file: Annotated[UploadFile, File()],
    conf: Annotated[float, Query(ge=0.01, le=0.99)] = 0.05,
):
    data = await read_upload_limited(file)
    try:
        async with state["inference_lock"]:
            result = await run_in_threadpool(state["pipeline"].analyze_bytes, data, conf)
    except Exception as e:
        logger.exception("mammogram analysis failed")
        raise HTTPException(400, "cannot process image") from e

    # Send the decoded image back so the browser can display DICOM too.
    image = result.pop("image")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "filename": safe_filename(file.filename),
        "width": image.width,
        "height": image.height,
        "image_png_base64": base64.b64encode(buf.getvalue()).decode(),
        **result,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
