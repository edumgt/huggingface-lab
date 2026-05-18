# Chapter 11. VirtualBox + Kubernetes (k3s) + 폐쇄망 AI 이미지 생성 환경 구축

이 챕터에서는 **VirtualBox Ubuntu 24 VM** 위에서 **k3s(경량 Kubernetes)** 를 실행하고,
HuggingFace 모델을 사전 다운로드하여 **폐쇄망(인터넷 없음)** 환경에서도 AI 이미지 생성이
가능하도록 전체 환경을 구성하는 방법을 설명합니다.

---

## 아키텍처 개요

```
┌─ 호스트 PC ────────────────────────────────────────────┐
│  VirtualBox VM (Ubuntu 24.04, 6 GB RAM, 4 vCPU)         │
│  ┌─ k3s (single-node Kubernetes) ──────────────────┐    │
│  │  Namespace: imggen                               │    │
│  │  ┌─ Pod: imggen ───────────────────────────────┐ │    │
│  │  │  Container: python-generate-image:latest    │ │    │
│  │  │  - FastAPI (port 8000)                      │ │    │
│  │  │  - diffusers + torch (CPU)                  │ │    │
│  │  │  - HF 모델 캐시 → PVC(hf-cache-pvc)         │ │    │
│  │  └────────────────────────────────────────────┘ │    │
│  │  NodePort 30800 → Pod:8000                       │    │
│  └──────────────────────────────────────────────────┘    │
│  포트 포워딩: localhost:30800 → VM:30800                  │
└──────────────────────────────────────────────────────────┘
```

---

## 사전 조건 (호스트 PC)

| 소프트웨어 | 버전 | 다운로드 |
|-----------|------|---------|
| VirtualBox | 7.x | https://www.virtualbox.org/wiki/Downloads |
| Vagrant | 2.4+ | https://developer.hashicorp.com/vagrant/downloads |

---

## 빠른 시작 (인터넷 연결된 환경)

```bash
# 1. 저장소 클론
git clone https://github.com/edumgt/Python-Generate-image.git
cd Python-Generate-image

# 2. VM 생성 및 프로비저닝 (약 20~30 분 소요)
vagrant up

# 3. VM SSH 접속
vagrant ssh

# 4. 달 이미지 생성 테스트
cd /home/vagrant/app
python3 moon_test.py
# → moon_output.png 생성 확인

# 5. 웹 UI 확인
# 호스트 브라우저에서: http://localhost:30800
```

`vagrant up` 은 내부적으로 `scripts/setup-vm.sh` 를 실행하여 다음을 자동화합니다:

1. **시스템 패키지 설치** (Python 3.12, Docker, curl 등)
2. **Docker 설치** — 컨테이너 이미지 빌드용
3. **k3s 설치** — 경량 단일 노드 Kubernetes
4. **Python 가상 환경 + 의존성 설치** (diffusers, transformers, torch-cpu)
5. **HuggingFace 모델 사전 다운로드** (`scripts/preload-models.py`)
6. **Docker 이미지 빌드 + k3s에 배포** (`scripts/deploy.sh`)

---

## 폐쇄망(인터넷 차단) 환경 구성

### 방법: USB/HDD로 모델 캐시 전달

인터넷이 연결된 PC에서 모델을 미리 다운로드한 후 VM으로 복사합니다.

```bash
# [인터넷 연결 PC] 모델 캐시 다운로드
HF_HOME=./hf_cache python3 scripts/preload-models.py

# 캐시 디렉터리를 압축하여 이동 미디어에 복사
tar -czf hf_cache.tar.gz hf_cache/

# [VM 내부] 캐시 복원
tar -xzf hf_cache.tar.gz -C /home/vagrant/
```

VM 내부에서 완전 오프라인으로 실행:

```bash
export HF_HOME=/home/vagrant/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python3 moon_test.py         # 달 이미지 생성
# or
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 디렉터리 구조

```
Python-Generate-image/
├── Vagrantfile                  # VirtualBox VM 정의
├── Dockerfile                   # 컨테이너 이미지 빌드
├── docker-compose.yml           # Docker Compose (VM 없이 직접 실행)
├── moon_test.py                 # 달 이미지 생성 테스트 스크립트
├── k8s/
│   ├── namespace.yaml           # Kubernetes 네임스페이스
│   ├── configmap.yaml           # 환경 변수 (HF_HOME, OFFLINE 플래그)
│   ├── pvc.yaml                 # 모델 캐시 + 결과 파일 볼륨
│   ├── deployment.yaml          # Pod 정의 (이미지, 볼륨 마운트, 프로브)
│   └── service.yaml             # ClusterIP + NodePort:30800
├── scripts/
│   ├── setup-vm.sh              # VM 프로비저닝 스크립트 (전체 자동화)
│   ├── preload-models.py        # HF 모델 사전 다운로드
│   └── deploy.sh                # Docker 빌드 → k3s 이미지 임포트 → kubectl apply
└── backend/
    ├── requirements.txt         # Python 의존성 (diffusers, transformers, torch)
    ├── main.py                  # FastAPI 엔트리포인트
    └── app/
        └── generator.py         # AI 생성 (diffusers) + SVG 폴백
```

---

## k3s 운영 명령어

```bash
# 파드 상태 확인
k3s kubectl get pods -n imggen

# 파드 로그 확인
k3s kubectl logs -n imggen deployment/imggen -f

# 배포 재시작 (이미지 업데이트 후)
bash /home/vagrant/app/scripts/deploy.sh

# 서비스 URL 확인
k3s kubectl get svc -n imggen

# 헬스 체크
curl http://localhost:30800/api/health
```

---

## moon_test.py 실행 결과 예시

```
Device : cpu
Model  : runwayml/stable-diffusion-v1-5
Prompt : a single bright moon floating in outer space, deep cosmos, stars, photorealistic, 4K
Loading pipeline … (this may take a few minutes on first run)
Generating image …
100%|████████████████████████| 20/20 [02:14<00:00,  6.73s/it]

✓ Image saved: /home/vagrant/app/moon_output.png
```

생성된 `moon_output.png` 를 VM 에서 SCP 로 호스트로 복사하거나,
웹 UI(`http://localhost:30800`) 에서 같은 프롬프트를 입력하여 확인할 수 있습니다.

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `vagrant up` 중 "Box not found" | Ubuntu 24 Vagrant box 미존재 | `vagrant box add ubuntu/noble64` |
| k3s 파드가 `Pending` | PVC StorageClass 없음 | k3s 재설치 or `local-path` SC 확인 |
| `TRANSFORMERS_OFFLINE` 에러 | 모델 캐시 없음 | `preload-models.py` 먼저 실행 |
| 메모리 부족 | VM RAM < 4 GB | Vagrantfile `vb.memory = 6144` 확인 |
| `moon_test.py` 가 placeholder만 생성 | diffusers 미설치 | `pip install diffusers transformers torch` |
