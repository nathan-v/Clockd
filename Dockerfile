# ---- CPU target ----
FROM python:3.12-slim AS cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r clockd && useradd -r -g clockd -d /app -s /sbin/nologin clockd

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY configs/server.yaml configs/server.yaml

# Install CPU-only PyTorch first (much smaller than the default CUDA bundle),
# then install the project which will reuse the existing torch.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && mkdir -p /tmp/clockd_uploads \
    && python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')" \
    && chown -R clockd:clockd /app /tmp/clockd_uploads

USER clockd
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "clockd.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- GPU target ----
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r clockd && useradd -r -g clockd -d /app -s /sbin/nologin clockd

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY configs/server.yaml configs/server.yaml

RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && mkdir -p /tmp/clockd_uploads \
    && chown -R clockd:clockd /app /tmp/clockd_uploads

USER clockd
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "clockd.main:app", "--host", "0.0.0.0", "--port", "8000"]
