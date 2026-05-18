from diffusers import StableDiffusionImg2ImgPipeline
import torch
import os
import cv2
from PIL import Image
import numpy as np

# === 설정 ===
prompt = "Just two woman, natural lighting"
output_dir = "frames_connected"
os.makedirs(output_dir, exist_ok=True)

num_frames = 96               # 총 프레임 수 (4초짜리 영상 = 24fps x 4초)
fps = 24                      # 초당 프레임 수
width, height = 1080, 1920
guidance_scale = 7.5
strength_decay = 0.985        # 점진적 변화 비율
min_strength = 0.3            # 최소 변화도 (너무 낮으면 이미지 변화 없음)

# === 모델 로드 ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "Lykon/dreamshaper-8",
    torch_dtype=torch.float16
).to("cuda")

# === 첫 프레임 생성 ===
initial_image = Image.new("RGB", (width, height), "white")
image = pipe(prompt=prompt, image=initial_image, strength=1.0, guidance_scale=guidance_scale).images[0]
image.save(f"{output_dir}/frame_000.png")

# === 이어지는 프레임 생성 ===
for i in range(1, num_frames):
    strength = max(min_strength, strength_decay ** i)
    image = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance_scale).images[0]
    image.save(f"{output_dir}/frame_{i:03}.png")
    print(f"🖼️ 프레임 {i:03} 저장 완료 (strength={strength:.4f})")

# === OpenCV로 mp4 저장 ===
frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
sample_frame = cv2.imread(os.path.join(output_dir, frame_files[0]))
height, width, _ = sample_frame.shape

out = cv2.VideoWriter("output_connected.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for file in frame_files:
    frame = cv2.imread(os.path.join(output_dir, file))
    out.write(frame)

out.release()
print("✅ 24fps 영상 생성 완료: output_connected.mp4")
