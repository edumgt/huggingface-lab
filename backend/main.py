from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.app.generator import GenerateRequest, GenerationResult, GeneratorService

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


app = FastAPI(title="python-generate-image studio", version="2.0.0")
service = GeneratorService(output_dir=OUTPUT_DIR)

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
    )


@app.post("/api/generate", response_model=GenerationResult)
def generate(payload: GenerateRequest) -> GenerationResult:
    return service.generate(payload)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
