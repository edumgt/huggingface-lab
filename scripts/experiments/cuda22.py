import os
import cv2
import subprocess
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import time

# === 설정 ===
input_video = "input.mp4"
bgm_file = "background.mp3"
fps = 30
prompt = "anime girl jogging on riverside, soft morning light, dynamic movement, highly detailed"
seed_base = 1234

# === 폴더 준비 ===
frame_dir = "pose_inputs"
pose_dir = "pose_maps"
output_dir = "pose2img_frames"
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# === Step 1. 영상 → 프레임 추출 ===
print("🎥 영상 → 프레임 추출 중...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", input_video,
    "-vf", f"fps={fps},scale=512:512",
    os.path.join(frame_dir, "frame_%03d.png")
])

# === Step 2. 포즈 감지 ===
print("🦴 OpenPose 포즈 감지 시작...")
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
input_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

for i, fname in enumerate(input_frames):
    img = Image.open(os.path.join(frame_dir, fname)).resize((512, 512))
    pose = pose_detector(img)
    pose.save(os.path.join(pose_dir, f"pose_{i:03}.png"))
    print(f"🦴 pose_{i:03}.png")

# === Step 3. ControlNet pose2img 생성 ===
print("🎨 pose2img 생성 시작...")
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

pose_imgs = sorted(os.listdir(pose_dir))
start = time.time()
for i, fname in enumerate(pose_imgs):
    pose_img = Image.open(os.path.join(pose_dir, fname)).resize((512, 512))
    image = pipe(
        prompt=prompt,
        image=pose_img,
        num_inference_steps=30,
        guidance_scale=8.0,
        generator=torch.Generator("cuda").manual_seed(seed_base + i)
    ).images[0]
    image.save(os.path.join(output_dir, f"frame_{i:03}.png"))
    print(f"✅ frame_{i:03}.png")
print(f"🖼️ 전체 완료 ⏱️ {time.time() - start:.2f}초")

# === Step 4. 프레임 → 영상 생성 ===
print("🎬 영상으로 합치는 중...")
subprocess.run([
    "ffmpeg", "-y",
    "-r", str(fps),
    "-i", os.path.join(output_dir, "frame_%03d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "pose_output.mp4"
])

# === Step 5. 배경 음악 병합 (선택) ===
if os.path.exists(bgm_file):
    print("🎵 오디오 병합 중...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", "pose_output.mp4",
        "-i", bgm_file,
        "-c:v", "copy", "-c:a", "aac",
        "-shortest", "pose_output_with_audio.mp4"
    ])
    print("✅ 최종 파일: pose_output_with_audio.mp4")
else:
    print("⚠️ 배경음 없음, 영상만 저장됨 → pose_output.mp4")
