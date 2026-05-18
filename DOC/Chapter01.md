# Chapter 01. 저장소 개요와 목표

이 저장소는 **Python 기반 생성형 이미지/영상 실험**을 중심으로 시작되었으며, 현재는 운영/테스트를 쉽게 하기 위해 FastAPI 백엔드 + 바닐라 JavaScript 프런트엔드 구조를 함께 갖추도록 확장되었습니다.

## 핵심 목표

1. 기존 실험 스크립트의 기술 스택을 문서화
2. 웹에서 확인 가능한 API/대시보드 제공
3. Docker 기반의 재현 가능한 테스트 환경 제공

## 기존 구성 요약

- 단일 Python 스크립트 중심 실험 (`scripts/experiments/cuda*.py`, `scripts/experiments/cpu.py`, `scripts/experiments/mp4make.py`)
- Hugging Face `diffusers` + `torch` 기반 이미지 생성
- `opencv`, `Pillow`, `ffmpeg` 기반 영상 후처리

## 확장 구성 요약

- `backend/main.py`: FastAPI 엔트리포인트
- `frontend/`: 정적 웹 UI (Vanilla JS)
- `tests/`: API 테스트
- `Dockerfile`, `docker-compose.yml`: 컨테이너 실행/테스트 환경
