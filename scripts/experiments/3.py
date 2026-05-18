import torch
from diffusers import StableDiffusionPipeline
import os

# ✅ Hugging Face 토큰이 필요 없는 공개 모델 사용 (runwayml)
# model_id = "Lykon/dreamshaper-8"
# model_id = "runwayml/stable-diffusion-v1-5"
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
HF_TOKEN = os.getenv("HF_TOKEN", "hf_***MASKED***")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=HF_TOKEN,  # ✅ 토큰 전달
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # CPU면 float32 권장
).to("cuda" if torch.cuda.is_available() else "cpu")

pipe.safety_checker = None
# ✅ 새로운 챔피언들
champions = {
    "3": {
    "trait": "A Korean martial artist in a modern training outfit, full body, 2:3 portrait, natural soft lighting, smooth skin",
    "pose": "Standing in a confident sparring stance"
  },
    "4": {
    "trait": "Two athletes preparing for a fitness photoshoot in a studio",
    "pose": "Posing back-to-back with energetic expressions"
  },
}

# ✅ 리얼리즘 강화 스타일
# style = (
#     "Robot style, full body,  2:3 portrait, desert asphalt plaza at 3pm, "
#     "natural lighting, 3D illustration, two-tone color scheme: dark gray and light brown, "
#     "high resolution, dynamic pose, volumetric lighting, fabric texture, "
#     "hyper detailed hands and face, cinematic shading, highly detailed armor"
# )

# style = "Robot style, full body,  2:3 portrait, desert asphalt plaza at 3pm, "
# style += "natural lighting, 3D illustration, two-tone color scheme: dark gray and light brown, fabric texture"

style = ""

# ✅ 출력 디렉토리 생성
os.makedirs("outputs", exist_ok=True)

# ✅ 챔피언 이미지 생성
for name, desc in champions.items():
    prompt = f"{name}, {desc['trait']}, {desc['pose']}, {style}"
    print(f"🎨 Generating {name}...")
    image = pipe(prompt, width=720, height=1280).images[0]
    image.save(f"outputs/{name}.png")

print("✅ New champion portraits saved in /outputs")
