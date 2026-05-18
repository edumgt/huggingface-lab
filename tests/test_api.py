from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_options() -> None:
    response = client.get("/api/options")
    assert response.status_code == 200
    payload = response.json()
    assert "models" in payload
    assert "image" in payload["output_types"]
    assert "video" in payload["output_types"]


def test_generate_image() -> None:
    response = client.post(
        "/api/generate",
        json={
            "model_id": "stabilityai/sdxl-turbo",
            "output_type": "image",
            "prompt": "a cozy cabin in snowy mountain",
            "width": 512,
            "height": 512,
            "video_size": "square",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    # media_type is "image/png" when diffusers is available, otherwise "image/svg+xml"
    assert payload["media_type"] in ("image/svg+xml", "image/png")
    assert payload["width"] == 512
    assert payload["height"] == 512
    assert payload["file_url"].startswith("/outputs/asset_")
    assert payload["data_url"] is not None
