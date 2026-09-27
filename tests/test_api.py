"""
A few focused tests covering the core happy path and the failure modes
worth demonstrating for a take-home: bad content-type, empty upload, and
a DB round-trip check. Not aiming for coverage -- aiming for evidence of
deliberate failure-mode thinking.

NOTE: adjust the import path (`app.main`) and the endpoint URL
(`/predict` below) if your actual route is named differently
(e.g. `/predict`) -- match whatever you actually wired in endpoints.py.
"""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# Point this at a real tile from your dataset so the happy-path test
# exercises the actual trained model, not a fake image.
SAMPLE_TILE_PATH = Path("data/eval_set/tile_001.png")


def _make_fake_png_bytes() -> bytes:
    """A tiny in-memory valid PNG, for tests that don't need a real tile."""
    img = Image.new("RGB", (64, 64), color=(120, 150, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_valid_tile_returns_prediction(client):
    with open(SAMPLE_TILE_PATH, "rb") as f:
        response = client.post(
            "/predict",
            files={"tile": ("tile_001.png", f, "image/png")},
        )

    assert response.status_code == 200
    body = response.json()

    assert body["predicted_label"] in {
        "AnnualCrop", "Forest", "Highway", "Industrial",
        "Residential", "River", "SeaLake",
    }
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["class_probs"], dict)
    assert len(body["class_probs"]) == 7
    assert body["model_version"]
    assert "prediction_id" in body


def test_predict_rejects_non_image_file(client):
    fake_file = io.BytesIO(b"this is not an image, just plain text bytes")
    response = client.post(
        "/predict",
        files={"tile": ("not_an_image.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400


def test_predict_rejects_corrupt_image_bytes(client):
    """Content-type header claims image/png, but the bytes are garbage --
    tests the decode-failure path inside ml/predict.py, not just the
    content-type check in the route."""
    corrupt_bytes = io.BytesIO(b"\x89PNG\r\n\x1a\nnot actually valid PNG data")
    response = client.post(
        "/predict",
        files={"tile": ("corrupt.png", corrupt_bytes, "image/png")},
    )
    assert response.status_code == 400


def test_predict_rejects_empty_file(client):
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/predict",
        files={"tile": ("empty.png", empty_file, "image/png")},
    )
    assert response.status_code == 400


def test_predictions_endpoint_reflects_stored_result(client):
    fake_bytes = _make_fake_png_bytes()
    predict_response = client.post(
        "/predict",
        files={"tile": ("roundtrip_test.png", io.BytesIO(fake_bytes), "image/png")},
    )
    assert predict_response.status_code == 200
    prediction_id = predict_response.json().get("prediction_id")

    list_response = client.get("/predictions")
    assert list_response.status_code == 200

    stored_ids = [row.get("prediction_id") for row in list_response.json()]
    assert prediction_id in stored_ids