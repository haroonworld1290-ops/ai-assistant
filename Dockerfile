# Render-optimized Dockerfile for FastAPI app
# Uses Gunicorn + Uvicorn worker for production-grade serving

FROM python:3.11-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy app code
COPY . /app

# Ensure files are readable
RUN chown -R app:app /app

# Use non-root user
USER app

# Default port and healthcheck
ENV PORT=8000
EXPOSE 8000

# Healthcheck for Render
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

# Start using Gunicorn with Uvicorn workers
# Workers: use (2 x $num_cores) + 1 formula — keep it modest for small plans
CMD ["/bin/sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT} --workers 3 --timeout 120"]
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
