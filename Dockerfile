FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch before other deps (smaller, ~700 MB)
RUN pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY . /app

# HuggingFace model cache directory.
# Mount a PVC here (see k8s/pvc.yaml) to persist pre-downloaded models.
# TRANSFORMERS_OFFLINE and HF_DATASETS_OFFLINE are intentionally NOT set here
# so the image works both online and offline.  Set them to "1" at runtime
# (e.g. via the k8s ConfigMap in k8s/configmap.yaml) for air-gapped operation.
ENV HF_HOME=/app/hf_cache

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
