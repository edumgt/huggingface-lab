from diffusers import StableDiffusionPipeline
import torch
import os
import cv2
from PIL import Image

# 모델 로드 (CPU 전용)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 폴더 준비
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# 프레임 생성
prompt = "Robot style, full body,  2:3 portrait, desert asphalt plaza at 3pm, "
prompt += "natural lighting, 3D illustration, two-tone color scheme: dark gray and light brown"
prompt += "high resolution, dynamic pose, volumetric lighting, fabric texture, "
prompt += "hyper detailed hands and face, cinematic shading, highly detailed armor"


for i in range(30):
    image = pipe(prompt, width=720, height=1280).images[0]
    image.save(f"{output_dir}/frame_{i:03}.png")

# OpenCV로 동영상 생성
frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
sample_frame = cv2.imread(os.path.join(output_dir, frame_files[0]))
height, width, _ = sample_frame.shape

out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
for file in frame_files:
    frame = cv2.imread(os.path.join(output_dir, file))
    out.write(frame)
out.release()
print("🎞️ 동영상 생성 완료: output_video.mp4")
