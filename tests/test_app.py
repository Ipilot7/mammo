import asyncio
import threading
import time
from pathlib import Path
from typing import Annotated

import pytest
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from app import main as app_main

ROOT = Path(__file__).resolve().parents[1]


class FakeUpload:
    def __init__(self, data=b"image", filename="test.png"):
        self.data = data
        self.filename = filename
        self.requested_size = None

    async def read(self, size=-1):
        self.requested_size = size
        return self.data if size < 0 else self.data[:size]


def test_upload_reader_limits_bytes_before_inference():
    upload = FakeUpload(b"123456")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_main.read_upload_limited(upload, max_bytes=5))

    assert exc.value.status_code == 413
    assert upload.requested_size == 6


def test_request_body_limit_rejects_multipart_before_endpoint_parsing():
    test_app = FastAPI()
    test_app.add_middleware(app_main.MaxBodySizeMiddleware, max_bytes=128)
    called = False

    @test_app.post("/upload")
    async def upload(file: Annotated[UploadFile, File()]):
        nonlocal called
        called = True
        return {"size": len(await file.read())}

    response = TestClient(test_app).post(
        "/upload",
        files={"file": ("large.bin", b"x" * 1024, "application/octet-stream")},
    )

    assert response.status_code == 413
    assert called is False


def test_request_body_limit_counts_stream_without_content_length():
    test_app = FastAPI()
    test_app.add_middleware(app_main.MaxBodySizeMiddleware, max_bytes=128)
    completed = False

    @test_app.post("/upload")
    async def upload(request: Request):
        nonlocal completed
        await request.body()
        completed = True
        return {"ok": True}

    response = TestClient(test_app).post(
        "/upload",
        content=(b"x" * 64 for _ in range(3)),
        headers={"content-type": "application/octet-stream"},
    )

    assert response.status_code == 413
    assert completed is False


def test_analyze_hides_internal_exception_details(monkeypatch):
    class BrokenPipeline:
        def analyze_bytes(self, _data, _conf):
            raise RuntimeError("secret path: /private/models/detector.pt")

    app_main.state["pipeline"] = BrokenPipeline()
    app_main.state["inference_lock"] = asyncio.Lock()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_main.analyze(FakeUpload(), conf=0.05))

    assert exc.value.status_code == 400
    assert exc.value.detail == "cannot process image"
    assert "/private/" not in exc.value.detail


def test_lifespan_requires_all_three_models(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "detector.pt").write_bytes(b"x")
    (models / "classifier.pt").write_bytes(b"x")
    monkeypatch.setattr(app_main, "MODELS", models)

    async def run():
        async with app_main.lifespan(app_main.app):
            pass

    with pytest.raises(RuntimeError, match="screening.pt"):
        asyncio.run(run())


def test_lifespan_honors_device_override_and_creates_lock(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    for name in ("detector.pt", "classifier.pt", "screening.pt"):
        (models / name).write_bytes(b"x")
    monkeypatch.setattr(app_main, "MODELS", models)
    monkeypatch.setenv("MAMMO_DEVICE", "cpu")
    created = {}

    class FakePipeline:
        def __init__(self, device=None):
            self.device = device
            created["device"] = device

    monkeypatch.setattr(app_main, "MammoPipeline", FakePipeline)

    async def run():
        async with app_main.lifespan(app_main.app):
            assert isinstance(app_main.state["inference_lock"], asyncio.Lock)
            assert app_main.state["pipeline"].device == "cpu"

    asyncio.run(run())
    assert created["device"] == "cpu"


def test_successful_analysis_returns_png_without_pipeline_image(monkeypatch):
    class Pipeline:
        def analyze_bytes(self, _data, _conf):
            return {
                "image": Image.new("L", (4, 3), 0),
                "findings": [],
                "screening": None,
                "breast_contour": [],
                "overall_birads": None,
                "overall_level": "none",
            }

    app_main.state["pipeline"] = Pipeline()
    app_main.state["inference_lock"] = asyncio.Lock()
    result = asyncio.run(app_main.analyze(FakeUpload(filename="../../sample.dcm"), conf=0.05))

    assert result["filename"] == "sample.dcm"
    assert result["width"] == 4
    assert result["height"] == 3
    assert result["findings"] == []
    assert "image" not in result
    assert result["image_png_base64"]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("../../sample.dcm", "sample.dcm"),
        (r"C:\Users\Alice\patient.dcm", "patient.dcm"),
        (r"..\..\patient.dcm", "patient.dcm"),
    ],
)
def test_safe_filename_removes_posix_and_windows_paths(filename, expected):
    assert app_main.safe_filename(filename) == expected


def test_viewer_declares_empty_favicon_to_avoid_console_404():
    html = (ROOT / "app" / "static" / "index.html").read_text()

    assert '<link rel="icon" href="data:,">' in html


def test_concurrent_requests_are_serialized():
    class Pipeline:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.guard = threading.Lock()

        def analyze_bytes(self, _data, _conf):
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            with self.guard:
                self.active -= 1
            return {
                "image": Image.new("L", (2, 2), 0),
                "findings": [],
                "screening": None,
                "breast_contour": [],
                "overall_birads": None,
                "overall_level": "none",
            }

    pipeline = Pipeline()

    async def run():
        app_main.state["pipeline"] = pipeline
        app_main.state["inference_lock"] = asyncio.Lock()
        await asyncio.gather(
            app_main.analyze(FakeUpload(filename="one.png"), conf=0.05),
            app_main.analyze(FakeUpload(filename="two.png"), conf=0.05),
        )

    asyncio.run(run())
    assert pipeline.max_active == 1
