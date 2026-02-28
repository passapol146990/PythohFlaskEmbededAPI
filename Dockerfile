# ── Stage 1: download model weights ──────────────────────────────────────────
FROM python:3.11-slim AS model-downloader

WORKDIR /model-cache

# Install only what's needed to pull the model
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu \
    sentence-transformers==3.4.1

# Pre-download model into a known directory
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-base', cache_folder='/model-cache/hub')"


# ── Stage 2: production image ─────────────────────────────────────────────────
FROM python:3.11-slim

# Resource-optimization env vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Limit CPU parallelism — keeps RAM & CPU usage low
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TORCH_THREADS=2 \
    # Point HF to the pre-downloaded model
    TRANSFORMERS_CACHE=/app/hub \
    HF_HOME=/app/hub \
    PORT=5000

WORKDIR /app

# Install runtime dependencies
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy pre-downloaded model weights from stage 1
COPY --from=model-downloader /model-cache/hub /app/hub

# Copy application code
COPY embedding_service.py index.py ./

EXPOSE 5000

# 1 worker → model lives in a single process (no memory duplication)
# --preload  → model loaded before forking (saves RAM with multiple workers)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--timeout", "120", \
     "--preload", \
     "index:app"]
