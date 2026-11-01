FROM python:3.11-slim

WORKDIR /app

# System deps:
#   curl          — used by compose healthcheck
#   libgl1        — required by docling's image-processing stack
#   libglib2.0-0  — same (transitive via OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so source edits don't bust the layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source — .dockerignore excludes data/, benchmark/, .env, .git, etc.
COPY . .

# Cache + uploads live on a volume mounted at /app/data
# Run as an unprivileged user; it owns only the data volume mount point.
RUN useradd --create-home --uid 1000 arse && mkdir -p /app/data && chown arse:arse /app/data
USER arse

ENV PYTHONUNBUFFERED=1 \
    ARES_CACHE=/app/data/cache.sqlite \
    GROBID_URL=http://grobid:8070

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Remaining Streamlit settings live in .streamlit/config.toml.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
