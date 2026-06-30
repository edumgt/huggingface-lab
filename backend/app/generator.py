from __future__ import annotations

import html
import base64
import io
import logging
import os
import textwrap
import uuid
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional AI back-end (diffusers / torch).  When the libraries are not
# installed the service falls back to an SVG placeholder so that the FastAPI
# app and tests still work without a GPU or a full model installation.
# ---------------------------------------------------------------------------
try:
    import torch
    from diffusers import StableDiffusionPipeline

    _DIFFUSERS_AVAILABLE = True
except ImportError:
    _DIFFUSERS_AVAILABLE = False
    logger.info("diffusers/torch not installed — using SVG preview fallback")


STYLE_PRESETS: dict[str, str] = {
    "none": "",
    "finance_poster": (
        "financial poster design, stock market growth chart background, bold modern "
        "headline typography, gold and navy color palette, glowing line graph, "
        "corporate fintech branding, high contrast, sleek, 4k"
    ),
    "infographic": (
        "clean infographic style, flat design icons, data dashboard layout, soft "
        "gradient background, minimal vector illustration, financial statistics "
        "panels, organized grid composition, crisp typography"
    ),
}


class GenerateRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str = Field(min_length=1)
    output_type: str = Field(pattern="^(image|video)$")
    prompt: str = Field(min_length=1, max_length=1000)
    width: int = Field(default=768, ge=256, le=1536)
    height: int = Field(default=768, ge=256, le=1536)
    video_size: str = Field(default="square")
    style_preset: str = Field(default="none")

    @field_validator("video_size")
    @classmethod
    def validate_video_size(cls, value: str) -> str:
        allowed = {"square", "landscape", "portrait"}
        if value not in allowed:
            raise ValueError(f"video_size must be one of {sorted(allowed)}")
        return value

    @field_validator("style_preset")
    @classmethod
    def validate_style_preset(cls, value: str) -> str:
        if value not in STYLE_PRESETS:
            raise ValueError(f"style_preset must be one of {sorted(STYLE_PRESETS)}")
        return value

    @property
    def styled_prompt(self) -> str:
        suffix = STYLE_PRESETS[self.style_preset]
        return f"{self.prompt}, {suffix}" if suffix else self.prompt


class GenerationResult(BaseModel):
    model_config = {"protected_namespaces": ()}
    file_url: str
    media_type: str
    width: int
    height: int
    model_id: str
    prompt: str
    data_url: str | None = None


class GeneratorService:
    VIDEO_SIZES = {
        "square": (768, 768),
        "landscape": (1024, 576),
        "portrait": (576, 1024),
    }

    # Cache loaded pipelines so they are not reloaded on every request.
    _pipeline_cache: dict[str, object] = {}

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def available_models() -> list[str]:
        return [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/sdxl-turbo",
        ]

    @staticmethod
    def available_style_presets() -> list[str]:
        return list(STYLE_PRESETS.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, request: GenerateRequest) -> GenerationResult:
        width, height = request.width, request.height

        if request.output_type == "video":
            width, height = self.VIDEO_SIZES[request.video_size]

        if _DIFFUSERS_AVAILABLE:
            return self._generate_ai(request, width, height)
        return self._generate_svg_preview(request, width, height)

    # ------------------------------------------------------------------
    # AI generation path (diffusers)
    # ------------------------------------------------------------------

    def _load_pipeline(self, model_id: str) -> "StableDiffusionPipeline":
        if model_id not in self._pipeline_cache:
            # HF_HOME is read automatically by the HuggingFace libraries.
            # Set TRANSFORMERS_OFFLINE=1 to prevent any internet access.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("Loading pipeline %s on %s …", model_id, device)
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)
            pipe.set_progress_bar_config(disable=True)
            self._pipeline_cache[model_id] = pipe
        return self._pipeline_cache[model_id]  # type: ignore[return-value]

    def _generate_ai(self, request: GenerateRequest, width: int, height: int) -> GenerationResult:
        pipe = self._load_pipeline(request.model_id)
        image = pipe(
            prompt=request.styled_prompt,
            width=width,
            height=height,
            num_inference_steps=20,
        ).images[0]

        file_name = f"asset_{uuid.uuid4().hex[:8]}.png"
        output_path = self.output_dir / file_name
        image.save(str(output_path), format="PNG")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")

        return GenerationResult(
            file_url=f"/outputs/{file_name}",
            media_type="image/png",
            width=width,
            height=height,
            model_id=request.model_id,
            prompt=request.prompt,
            data_url=f"data:image/png;base64,{encoded}",
        )

    # ------------------------------------------------------------------
    # SVG placeholder path (no diffusers installed)
    # ------------------------------------------------------------------

    def _generate_svg_preview(self, request: GenerateRequest, width: int, height: int) -> GenerationResult:
        label = "IMAGE" if request.output_type == "image" else f"VIDEO PREVIEW ({request.video_size})"
        output_path, svg_content = self._create_preview_svg(
            model_id=request.model_id,
            prompt=request.styled_prompt,
            width=width,
            height=height,
            label=label,
        )
        encoded_svg = base64.b64encode(svg_content.encode("utf-8")).decode("ascii")
        return GenerationResult(
            file_url=f"/outputs/{output_path.name}",
            media_type="image/svg+xml",
            width=width,
            height=height,
            model_id=request.model_id,
            prompt=request.prompt,
            data_url=f"data:image/svg+xml;base64,{encoded_svg}",
        )

    def _create_preview_svg(self, model_id: str, prompt: str, width: int, height: int, label: str) -> tuple[Path, str]:
        escaped_lines = textwrap.wrap(f"Prompt: {prompt}", width=60)
        text_lines = [
            label,
            f"Model: {model_id}",
            *escaped_lines,
        ]

        text_nodes = []
        y = 80
        for line in text_lines:
            safe_line = html.escape(line)
            text_nodes.append(
                f'<text x="36" y="{y}" fill="#e5e7eb" font-size="24" font-family="Arial">{safe_line}</text>'
            )
            y += 36

        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#0f172a" />
  <rect x="12" y="12" width="{width - 24}" height="{height - 24}" fill="none" stroke="#22d3ee" stroke-width="4" rx="10" />
  {"".join(text_nodes)}
</svg>
'''
        file_name = f"asset_{uuid.uuid4().hex[:8]}.svg"
        path = self.output_dir / file_name
        path.write_text(svg, encoding="utf-8")
        return path, svg
