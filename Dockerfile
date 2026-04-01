FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System build deps for some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download and bundle MiniLM model (no network at runtime)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('all-MiniLM-L6-v2', device='cpu'); \
m.save('/app/models/minilm-l6-v2')"

# Copy application code
COPY . .

# Precompute category and finding embeddings at build time
RUN python -c "\
from src.ml_models import precompute_all_embeddings; \
precompute_all_embeddings('/app/models/minilm-l6-v2', '/app/data/embeddings')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
