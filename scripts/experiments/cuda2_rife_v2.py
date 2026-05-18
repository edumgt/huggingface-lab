
from diffusers import StableDiffusionImg2ImgPipeline
import torch
import os
import cv2
from PIL import Image
import subprocess
import time
import shutil

# === 설정 ===
prompt = (
    "A young woman wearing shorts and a light casual outfit, walking slowly along a riverside at 8pm, "
    "under soft moonlight and warm streetlights, with gentle reflections on the water, "
    "quiet and peaceful night atmosphere, cinematic lighting, photorealistic style"
)
output_dir = "jogging_frames"
rife_ready_dir = "rife_input"
interpolated_frames = "interpolated_frames"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(rife_ready_dir, exist_ok=True)
os.makedirs(interpolated_frames, exist_ok=True)

num_frames = 180   # 5초 × 30fps
fps = 30
target_fps = 60
width, height = 512, 768
guidance_scale = 7.5
strength = 0.35
seed = 12345
generator = torch.Generator("cuda").manual_seed(seed)

mp3_file = "background.mp3"
temp_video = "temp_no_audio.mp4"
interpolated_video = "interpolated_60fps.mp4"
final_video = "jogging_with_audio.mp4"

# === 모델 로드 ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "Lykon/dreamshaper-8",
    torch_dtype=torch.float16
).to("cuda")
pipe.enable_attention_slicing()

# === 첫 프레임 생성
print("🖼️ 첫 프레임 생성 중...")
first_image = pipe(
    prompt=prompt,
    image=Image.new("RGB", (width, height), "white"),
    strength=1.0,
    guidance_scale=guidance_scale,
    generator=generator
).images[0]
first_image.save(f"{output_dir}/frame_000.png")
prev_image = first_image

# === 프레임 생성
print(f"🚀 {num_frames - 1}개 프레임 생성 중...")
start_time = time.time()
for i in range(1, num_frames):
    image = pipe(
        prompt=prompt,
        image=prev_image,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    image.save(f"{output_dir}/frame_{i:03}.png")
    print(f"✅ frame_{i:03}.png 생성 완료")
    prev_image = image
print(f"🖼️ 전체 프레임 생성 완료! ⏱️ {time.time() - start_time:.2f}초")

# === RIFE 입력 프레임 이름 변경
print("🔄 RIFE용 프레임 이름 변환 중...")
frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
for i, f in enumerate(frame_files):
    shutil.copy(os.path.join(output_dir, f), os.path.join(rife_ready_dir, f"{i:05}.png"))
print(f"✅ 변환 완료: {len(frame_files)}개")

# === RIFE 보간 실행
print("🌀 RIFE 보간 시작...")
rife_script = "arXiv2020-RIFE/inference_img.py"
result = subprocess.run([
    "python", rife_script,
    "--img", rife_ready_dir,
    "--output", interpolated_frames,
    "--exp", "2"
], capture_output=True, text=True)

print("📤 STDOUT:", result.stdout)
print("❗ STDERR:", result.stderr)

# === ffmpeg로 영상 생성
print("🎞️ 보간 프레임 영상 생성 중...")
subprocess.run([
    "ffmpeg", "-y",
    "-framerate", "60",
    "-i", os.path.join(interpolated_frames, "%05d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    interpolated_video
])
print(f"🎬 보간 영상 저장 완료: {interpolated_video}")

# === 배경 음악 병합
if os.path.exists(mp3_file):
    print("🎵 오디오 병합 중...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", interpolated_video,
        "-i", mp3_file,
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", final_video
    ])
    print(f"✅ 최종 영상 저장 완료: {final_video}")
else:
    print("⚠️ MP3 없음. 오디오 없이 저장됨.")
