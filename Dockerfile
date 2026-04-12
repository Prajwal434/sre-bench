FROM python:3.11-slim

# Create non-root user (HF Spaces runs as user 1000)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models.py .
COPY client.py .
COPY openenv.yaml .
COPY inference.py .
COPY baseline.py .
COPY data/ ./data/
COPY server/ ./server/

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# ---------------------------------------------------------------------------
# Required environment variables (set at runtime or in HF Space secrets)
#   API_BASE_URL  – OpenAI-compatible API endpoint
#   MODEL_NAME    – Model identifier for inference
#   HF_TOKEN      – HuggingFace / API key
# ---------------------------------------------------------------------------
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""
ENV SREBENCH_URL="http://localhost:7860"

# Expose port
EXPOSE 7860

# Health check — /health must return 200 for HF Space ping requirement
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

# Start the FastAPI + Gradio server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
