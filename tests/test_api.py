from pathlib import Path
import sys

import pandas as pd
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.main import app, stock_service


client = TestClient(app)


def _fake_ohlcv(periods: int = 90) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = pd.Series(range(periods), index=idx, dtype="float64") + 100.0
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": pd.Series([1_000_000 + i * 1000 for i in range(periods)], index=idx),
        },
        index=idx,
    )


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


def test_generate_image_with_finance_poster_style() -> None:
    response = client.post(
        "/api/generate",
        json={
            "model_id": "stabilityai/sdxl-turbo",
            "output_type": "image",
            "prompt": "quarterly earnings report",
            "width": 512,
            "height": 512,
            "video_size": "square",
            "style_preset": "finance_poster",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    # the response echoes back the user's raw prompt, the preset only augments
    # the prompt actually sent to the renderer.
    assert payload["prompt"] == "quarterly earnings report"


def test_stock_options() -> None:
    response = client.get("/api/stock-options")
    assert response.status_code == 200
    payload = response.json()
    assert "candle" in payload["chart_types"]
    assert "infographic" in payload["chart_types"]
    assert "rsi" in payload["indicators"]


def test_generate_stock_chart_candle(monkeypatch) -> None:
    monkeypatch.setattr(stock_service, "_data_fetcher", lambda ticker, period, interval: _fake_ohlcv())

    response = client.post(
        "/api/stock-chart",
        json={
            "ticker": "aapl",
            "period": "3mo",
            "interval": "1d",
            "chart_type": "candle",
            "indicators": ["sma20", "rsi", "macd", "volume"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["media_type"] == "image/png"
    assert payload["file_url"].startswith("/outputs/stock_")
    assert payload["data_url"] is not None
    assert "last_close" in payload["summary"]


def test_generate_stock_chart_infographic(monkeypatch) -> None:
    monkeypatch.setattr(stock_service, "_data_fetcher", lambda ticker, period, interval: _fake_ohlcv())

    response = client.post(
        "/api/stock-chart",
        json={
            "ticker": "MSFT",
            "period": "6mo",
            "interval": "1d",
            "chart_type": "infographic",
            "indicators": ["sma20"],
        },
    )
    assert response.status_code == 200
    assert response.json()["chart_type"] == "infographic"


def test_stock_chart_unknown_ticker_returns_400(monkeypatch) -> None:
    def _raise(ticker, period, interval):
        raise ValueError(f"'{ticker}' 종목의 시세 데이터를 찾을 수 없습니다.")

    monkeypatch.setattr(stock_service, "_data_fetcher", _raise)

    response = client.post(
        "/api/stock-chart",
        json={"ticker": "NOPE", "period": "1mo", "interval": "1d"},
    )
    assert response.status_code == 400
