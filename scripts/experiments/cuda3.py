import os
import cv2
import subprocess
import torch
import time
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector

# === 사용자 설정 ===
input_video = "input.mp4"
bgm_file = "background.mp3"
prompt = "anime girl jogging on riverside, soft morning light, dynamic movement, highly detailed"
fps = 30
seed_base = 1234

# === 폴더 설정 ===
frame_dir = "pose_inputs"
pose_dir = "pose_maps"
img_out_dir = "pose2img_frames"
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)
os.makedirs(img_out_dir, exist_ok=True)

# === Step 1: 영상 → 프레임 추출 ===
print("🎥 Step 1: 영상 → 프레임 추출 중...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", input_video,
    "-vf", f"fps={fps},scale=512:512",
    os.path.join(frame_dir, "frame_%03d.png")
])

# === Step 2: 포즈 감지 ===
print("🦴 Step 2: OpenPose 포즈 감지 시작...")
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
input_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

for i, fname in enumerate(input_frames):
    img = Image.open(os.path.join(frame_dir, fname)).resize((512, 512))
    pose = pose_detector(img)
    pose.save(os.path.join(pose_dir, f"pose_{i:03}.png"))
    print(f"🦴 포즈 추출 완료: pose_{i:03}.png")

# === Step 3: pose2img 프레임 생성 ===
print("🎨 Step 3: pose2img 기반 프레임 생성 시작...")
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")
# pipe.enable_xformers_memory_efficient_attention()

pose_imgs = sorted(os.listdir(pose_dir))
start_time = time.time()

for i, fname in enumerate(pose_imgs):
    pose_img = Image.open(os.path.join(pose_dir, fname)).resize((512, 512))
    image = pipe(
        prompt=prompt,
        image=pose_img,
        num_inference_steps=30,
        guidance_scale=8.0,
        generator=torch.Generator("cuda").manual_seed(seed_base + i)
    ).images[0]
    out_path = os.path.join(img_out_dir, f"frame_{i:03}.png")
    image.save(out_path)
    print(f"✅ 생성 완료: {out_path}")

print(f"🖼️ 전체 프레임 생성 완료 ⏱️ {time.time() - start_time:.2f}초")

# === Step 4: 영상으로 합치기 ===
temp_video = "temp_pose2img.mp4"
print("🎬 Step 4: 프레임 → 영상 합치는 중...")
subprocess.run([
    "ffmpeg", "-y",
    "-r", str(fps),
    "-i", os.path.join(img_out_dir, "frame_%03d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    temp_video
])

# === Step 5: 모션 보간 (옵션) ===
interpolated_video = "pose2img_60fps.mp4"
print("🌀 Step 5: 모션 보간 중 (30fps → 60fps)...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", temp_video,
    "-vf", "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
    interpolated_video
])

# === Step 6: 배경음 삽입 ===
final_video = "pose2img_final.mp4"
if os.path.exists(bgm_file):
    print("🎵 Step 6: 오디오 병합 중...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", interpolated_video,
        "-i", bgm_file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest", final_video
    ])
    print(f"✅ 최종 영상 저장 완료: {final_video}")
else:
    print(f"⚠️ MP3 없음. {interpolated_video} 만 저장됨.")

