"""FastAPI service: upload a mammogram, get detected lesions and their cancer-risk level."""
import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mammo.config import MODELS
from mammo.inference import MammoPipeline

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 100 * 1024 * 1024
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    missing = [p.name for p in (MODELS / "detector.pt", MODELS / "classifier.pt") if not p.exists()]
    if missing:
        raise RuntimeError(f"model weights not found in {MODELS}: {missing}. Train them first (see README).")
    state["pipeline"] = MammoPipeline()
    yield
    state.clear()


app = FastAPI(title="MammoAI", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"status": "ok", "device": state["pipeline"].device}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), conf: float = Query(0.05, ge=0.01, le=0.99)):
    data = await file.read()
    if not data or len(data) > MAX_UPLOAD:
        raise HTTPException(400, "empty or too large file")
    try:
        result = state["pipeline"].analyze_bytes(data, conf)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"cannot process image: {e}") from e

    # Send the decoded image back so the browser can display DICOM too.
    image = result.pop("image")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "filename": file.filename,
        "width": image.width,
        "height": image.height,
        "image_png_base64": base64.b64encode(buf.getvalue()).decode(),
        **result,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
