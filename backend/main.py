from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.app.generator import GenerateRequest, GenerationResult, GeneratorService
from backend.app.stock import StockChartRequest, StockChartResult, StockChartService

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
OUTPUT_DIR = BASE_DIR / "generated"


class HealthResponse(BaseModel):
    status: str


class FilesResponse(BaseModel):
    python_files: list[str]


class GenerationOptions(BaseModel):
    models: list[str]
    output_types: list[str]
    video_sizes: list[str]
    style_presets: list[str]


class StockOptions(BaseModel):
    periods: list[str]
    intervals: list[str]
    chart_types: list[str]
    indicators: list[str]


app = FastAPI(title="python-generate-image studio", version="2.0.0")
service = GeneratorService(output_dir=OUTPUT_DIR)
stock_service = StockChartService(output_dir=OUTPUT_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/files", response_model=FilesResponse)
def files() -> FilesResponse:
    python_files = sorted(path.name for path in BASE_DIR.glob("*.py"))
    return FilesResponse(python_files=python_files)


@app.get("/api/options", response_model=GenerationOptions)
def options() -> GenerationOptions:
    return GenerationOptions(
        models=service.available_models(),
        output_types=["image", "video"],
        video_sizes=list(service.VIDEO_SIZES.keys()),
        style_presets=service.available_style_presets(),
    )


@app.post("/api/generate", response_model=GenerationResult)
def generate(payload: GenerateRequest) -> GenerationResult:
    return service.generate(payload)


@app.get("/api/stock-options", response_model=StockOptions)
def stock_options() -> StockOptions:
    return StockOptions(
        periods=stock_service.available_periods(),
        intervals=stock_service.available_intervals(),
        chart_types=stock_service.available_chart_types(),
        indicators=stock_service.available_indicators(),
    )


@app.post("/api/stock-chart", response_model=StockChartResult)
def stock_chart(payload: StockChartRequest) -> StockChartResult:
    try:
        return stock_service.generate(payload)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
