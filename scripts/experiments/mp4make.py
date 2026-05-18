from diffusers import StableDiffusionImg2ImgPipeline
import torch
import os
import cv2
from PIL import Image
import numpy as np
import random

# === 설정 ===
output_dir = "frames_connected"
os.makedirs(output_dir, exist_ok=True)

num_frames = 300
fps = 30
width, height = 800, 600
guidance_scale = 7.5
strength = 0.75  # 고정된 strength

model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
HF_TOKEN = os.getenv("HF_TOKEN", "hf_***MASKED***")

# 프롬프트 설정
base_prompt = (
    "Full-body female humanoid robot, 25-year-old Japanese appearance, "
    "working as a nurse in a modern hospital, wearing a white and pink nurse uniform with a name tag and stethoscope, "
    "realistic chrome and ceramic body with soft joint covers, standing beside a hospital bed with medical equipment in the background, "
    "bright clinical lighting, soft focus, photorealistic"
)

# === 모델 로드 ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

# === 초기 이미지 준비 ===
initial_image = Image.new("RGB", (width, height), "white")

# === 프레임 생성 ===
for i in range(num_frames):
    # 동일한 프롬프트지만 살짝씩 random noise 주기 (seed 없이 입력의 다양성 확보)
    prompt = base_prompt + f", subtle variation {random.randint(1, 10000)}"

    result = pipe(
        prompt=prompt,
        image=initial_image,
        strength=strength,
        guidance_scale=guidance_scale
    ).images[0]

    frame_path = f"{output_dir}/frame_{i:03}.png"
    result.save(frame_path)
    print(f"🖼️ 프레임 {i:03} 저장 완료")

# === 영상으로 저장 ===
frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
sample_frame = cv2.imread(os.path.join(output_dir, frame_files[0]))
height, width, _ = sample_frame.shape

out = cv2.VideoWriter("output_connected.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
for file in frame_files:
    frame = cv2.imread(os.path.join(output_dir, file))
    out.write(frame)
out.release()
print("✅ 영상 저장 완료: output_connected.mp4")
