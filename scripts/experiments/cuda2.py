from diffusers import StableDiffusionImg2ImgPipeline
import torch
import os
import cv2
from PIL import Image
import subprocess
import time

# === 설정 ===
prompt = (
"A young woman wearing shorts and a light casual outfit, walking slowly along a riverside at 8pm, under soft moonlight and warm streetlights, with gentle reflections on the water, quiet and peaceful night atmosphere, cinematic lighting, photorealistic style"
)
output_dir = "jogging_frames"
os.makedirs(output_dir, exist_ok=True)

num_frames = 180   # 5초 × 30fps
fps = 30
target_fps = 60
width, height = 512, 768
guidance_scale = 7.5
strength = 0.35
generator = torch.Generator("cuda").manual_seed(12345)

mp3_file = "background.mp3"

# 파일명 정의
temp_video = "temp_no_audio.mp4"
interpolated_video = "interpolated_60fps.mp4"
final_video = "jogging_with_audio.mp4"

# === 모델 로드 ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    # "runwayml/stable-diffusion-v1-5",
    "Lykon/dreamshaper-8",
    torch_dtype=torch.float16
).to("cuda")
pipe.enable_attention_slicing()

# === 프레임 생성 ===
print(f"🚀 {num_frames}프레임 생성 시작...")
start_time = time.time()
prev_image = Image.new("RGB", (width, height), "white")

for i in range(num_frames):
    image = pipe(
        prompt=prompt,
        image=prev_image,
        strength=strength,
        guidance_scale=guidance_scale,
        generator = generator
    ).images[0]
    image.save(f"{output_dir}/frame_{i:03}.png")
    print(f"✅ frame_{i:03}.png 생성 완료")
    prev_image = image

print(f"🖼️ 전체 프레임 생성 완료! ⏱️ {time.time() - start_time:.2f}초")

# === 영상 생성 (OpenCV)
frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
sample_frame = cv2.imread(os.path.join(output_dir, frame_files[0]))
height, width, _ = sample_frame.shape
out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for f in frame_files:
    img = cv2.imread(os.path.join(output_dir, f))
    out.write(img)
out.release()
print(f"🎬 영상 생성 완료: {temp_video}")

# === 모션 보간 (30fps → 60fps)
print("🌀 모션 보간 중... (30fps → 60fps)")
cmd_interp = [
    "ffmpeg", "-y",
    "-i", temp_video,
    "-vf", "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
    interpolated_video
]
subprocess.run(cmd_interp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(f"✅ 보간된 영상 저장 완료: {interpolated_video}")

# === 배경 음악 병합
if os.path.exists(mp3_file):
    print("🎵 오디오 병합 중...")
    cmd_audio = [
        "ffmpeg", "-y",
        "-i", interpolated_video,
        "-i", mp3_file,
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", final_video
    ]
    subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"✅ 최종 영상 저장 완료: {final_video}")
else:
    print("⚠️ MP3 없음. 오디오 없이 저장됨.")
