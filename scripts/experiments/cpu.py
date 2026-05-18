from diffusers import StableDiffusionImg2ImgPipeline
import torch
import os
import cv2
from PIL import Image
import numpy as np

# === 설정 ===
# prompt = "A full-body One Single Real humanoid mech inspired by LOL Champion  Overwatch Hanzo, standing alone , "
# prompt += "tactical armor plating , natural lighting at 3pm, desert asphalt plaza "
# prompt += "background, color scheme is just gray and brown, sharp shadows, very high detailed textures"


output_dir = "frames_connected"
os.makedirs(output_dir, exist_ok=True)

num_frames = 300
fps = 30
width, height = 800, 600
guidance_scale = 8.0
strength_decay = 0.95
min_strength = 0.2

# playgroundai/playground-v2-1024px-aesthetic
# SG161222/Realistic_Vision_V6.0_B1_noVAE
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
HF_TOKEN = os.getenv("HF_TOKEN", "hf_***MASKED***")

# === 모델 로드 ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

# === 첫 이미지 로드 ===
initial_image = Image.new("RGB", (width, height), "white")
image = pipe(prompt=prompt, image=initial_image, strength=1.0, guidance_scale=guidance_scale).images[0]
image.save(f"{output_dir}/frame_000.png")

# === 이어지는 프레임 생성 ===
for i in range(1, num_frames):
    strength = max(min_strength, strength_decay ** i)
    image = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance_scale).images[0]
    image.save(f"{output_dir}/frame_{i:03}.png")
    print(f"🖼️ 프레임 {i:03} 저장 완료 (strength={strength:.4f})")

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
