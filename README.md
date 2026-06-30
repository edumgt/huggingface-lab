# python-generate-image

Python 기반 생성형 이미지/영상 실험 저장소입니다.
단일 FastAPI 백엔드 모듈과 프런트엔드가 결합된 **생성 스튜디오** 형태로 동작하며,
**AI 이미지 생성**과 **실시간 주식/금융 차트 분석**을 탭으로 분리해 제공합니다.

![alt text](image.png)

## 웹 스튜디오 기능

### AI 이미지 생성 탭

- Hugging Face 모델 선택 (`runwayml/stable-diffusion-v1-5`, `stabilityai/sdxl-turbo`)
- 스타일 프리셋 선택 — 기본 / 금융 포스터(`finance_poster`) / 인포그래픽(`infographic`)
  - 프리셋 선택 시 사용자 프롬프트에 스타일 키워드가 자동으로 덧붙여져 렌더링됩니다.
- 이미지/동영상(미리보기 GIF) 생성 모드 선택
- 이미지 Width/Height 입력, 동영상 사이즈 선택(square / landscape / portrait)

### 주식 차트 분석 탭

- 종목 코드 입력 (예: `AAPL`, `005930.KS`) → `yfinance` 로 실시간 시세 조회
- 차트 종류: 캔들스틱 / 라인 / 인포그래픽(포스터 스타일 요약 화면)
- 기간(1mo~5y) · 간격(일/주/월봉) 선택
- 보조지표: SMA20 · SMA60 · EMA20 · 볼린저밴드 · RSI · MACD · 거래량
- 종가, 변동률, 최고/최저가, 평균 거래량 요약 카드 제공

## API

- `GET /api/health` — 상태 확인
- `GET /api/files` — 루트 Python 실험 스크립트 목록
- `GET /api/options` — AI 이미지 생성 옵션(모델/출력 타입/스타일 프리셋 등)
- `POST /api/generate` — AI 이미지/동영상 생성
- `GET /api/stock-options` — 주식 차트 옵션(기간/간격/차트 종류/지표)
- `POST /api/stock-chart` — 실시간 시세 기반 차트 생성
- `GET /outputs/{filename}` — 생성된 결과 파일 정적 서빙

## 로컬 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

브라우저에서 `http://localhost:8000` 접속.

## Docker 실행

```bash
docker compose up --build
```

브라우저에서 `http://localhost:8000` 접속.

## 테스트

```bash
pytest -q
```

## 기술 스택

### 핵심 라이브러리

- `fastapi`, `uvicorn`, `pydantic` — REST API 서버 및 응답 검증
- `pytest`, `httpx` — 자동 테스트
- `torch`, `diffusers`, `transformers`, `accelerate` — AI 이미지 생성 추론
  (미설치 시 SVG 미리보기로 자동 폴백되어 GPU/모델 없이도 앱이 동작합니다)
- `yfinance`, `pandas`, `matplotlib` — 실시간 시세 조회 및 차트/인포그래픽 렌더링

### 기존 실험 스크립트 (`scripts/experiments/`)

저장소는 단일 Python 스크립트 중심의 생성형 AI 실험에서 출발했습니다.

- Stable Diffusion 계열 텍스트→이미지 생성, Img2Img 변형 생성
- ControlNet(OpenPose) 기반 조건부 생성
- 프레임 시퀀스 생성 후 `ffmpeg` 기반 영상 합성 (`mp4make.py`)
- GPU(CUDA) 환경에서는 `float16`으로 속도/메모리 효율 향상, CPU fallback 경로도 제공

이 스크립트들을 다루는 동영상 합성 파이프라인은 디스크/VRAM 사용량이 크고,
프레임 단위 장시간 작업에 대한 실패 복구 전략이 필요하다는 점에 주의합니다.

## 아키텍처

- `backend/main.py`: FastAPI 엔트리포인트. 정적 프런트엔드(`frontend/`)와 생성 결과(`generated/`)를 함께 서빙합니다.
- `backend/app/generator.py`: AI 이미지/동영상 생성 서비스. diffusers 사용 가능 여부에 따라 실제 추론 또는 SVG 폴백을 선택합니다.
- `backend/app/stock.py`: 실시간 시세 조회 및 캔들스틱/라인/인포그래픽 차트 렌더링 서비스.
- `frontend/`: 빌드 단계 없는 바닐라 JS UI. 좌측 사이드바의 탭 전환으로 AI 이미지 생성과 주식 분석을 오간다.
- `tests/`: FastAPI `TestClient` 기반 API 테스트.

설계 시 CORS는 로컬 개발 편의를 위해 전체 허용되어 있으며, 운영 환경에서는 허용 도메인을 최소화하는 것을 권장합니다.

## Docker / Kubernetes 운영

- `Dockerfile`: CPU-only PyTorch를 먼저 설치한 뒤 나머지 의존성을 설치하는 단일 이미지 빌드.
- `docker-compose.yml`: 로컬에서 포트/재시작 정책을 관리하며 VM 없이 직접 실행할 수 있습니다.
- `k8s/`: `namespace.yaml`, `configmap.yaml`(HF_HOME·오프라인 플래그), `pvc.yaml`(모델 캐시/결과 볼륨), `deployment.yaml`, `service.yaml`(NodePort `30800`)로 구성된 k3s 배포 매니페스트.
- `Vagrantfile` + `scripts/setup-vm.sh`: VirtualBox Ubuntu 24 VM에 Docker·k3s·Python 의존성을 자동 프로비저닝합니다.

### VirtualBox + k3s 빠른 시작 (인터넷 연결된 환경)

```bash
git clone https://github.com/edumgt/Python-Generate-image.git
cd Python-Generate-image

# VM 생성 및 프로비저닝 (약 20~30분 소요)
vagrant up
vagrant ssh

# 달 이미지 생성 테스트
cd /home/vagrant/app
python3 moon_test.py   # moon_output.png 생성 확인

# 호스트 브라우저에서 확인: http://localhost:30800
```

`vagrant up`은 내부적으로 `scripts/setup-vm.sh`를 실행해 시스템 패키지·Docker·k3s 설치, Python 가상환경 구성,
`scripts/preload-models.py`로 모델 사전 다운로드, `scripts/deploy.sh`로 이미지 빌드 및 k3s 배포까지 자동화합니다.

### 폐쇄망(인터넷 차단) 환경 구성

인터넷이 연결된 PC에서 모델 캐시를 미리 받아 USB/HDD로 옮기는 방식입니다.

```bash
# [인터넷 연결 PC] 모델 캐시 다운로드 후 압축
HF_HOME=./hf_cache python3 scripts/preload-models.py
tar -czf hf_cache.tar.gz hf_cache/

# [VM 내부] 캐시 복원 후 완전 오프라인 실행
tar -xzf hf_cache.tar.gz -C /home/vagrant/
export HF_HOME=/home/vagrant/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### k3s 운영 명령어

```bash
k3s kubectl get pods -n imggen                       # 파드 상태 확인
k3s kubectl logs -n imggen deployment/imggen -f      # 파드 로그 확인
bash /home/vagrant/app/scripts/deploy.sh             # 배포 재시작(이미지 업데이트 후)
k3s kubectl get svc -n imggen                        # 서비스 URL 확인
curl http://localhost:30800/api/health               # 헬스 체크
```

### 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `vagrant up` 중 "Box not found" | Ubuntu 24 Vagrant box 미존재 | `vagrant box add ubuntu/noble64` |
| k3s 파드가 `Pending` | PVC StorageClass 없음 | k3s 재설치 or `local-path` SC 확인 |
| `TRANSFORMERS_OFFLINE` 에러 | 모델 캐시 없음 | `preload-models.py` 먼저 실행 |
| 메모리 부족 | VM RAM < 4 GB | `Vagrantfile`의 `vb.memory = 6144` 확인 |
| `moon_test.py`가 placeholder만 생성 | diffusers 미설치 | `pip install diffusers transformers torch` |
| 주식 차트 생성 실패 | `yfinance` 네트워크 접근 불가 또는 잘못된 종목 코드 | 네트워크 연결 확인, 종목 코드 재확인 (한국 종목은 `.KS`/`.KQ` 접미사) |

## 운영/보안 권장 사항

- HF 토큰 등 민감정보는 환경 변수로 주입하고 코드에 하드코딩하지 않습니다.
- 운영 환경에서는 CORS 허용 도메인을 최소화하고, 리버스 프록시(Nginx) 연계를 고려합니다.
- 로그 레벨을 dev/prod로 분리하고, 헬스체크(`/api/health`) 기반 배포 자동화를 권장합니다.
- 의존성 취약점을 주기적으로 점검합니다.

## 로드맵

- **단기**: 기존 실험 스크립트 실행을 API 잡(Job)으로 래핑, 실행 상태/결과 파일 다운로드 API 제공
- **중기**: 작업 큐(Celery/RQ) 도입, WebSocket 기반 실시간 진행률 표시
- **장기**: 멀티 모델 라우팅, 사용자별 프로젝트/이력 관리, 클라우드 GPU 오토스케일링 연계
