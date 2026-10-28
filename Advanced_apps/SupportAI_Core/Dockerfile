FROM python:3.10-slim

WORKDIR /app

# No build tools needed — all pip packages have pre-built wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download spaCy model for Presidio PII detection
RUN python -m spacy download en_core_web_lg

# Pre-cache sentence-transformer model (avoids runtime download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')"

# Copy application code
COPY main.py streamlit_app.py logger.py kb.json ./
COPY .streamlit/ .streamlit/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Streamlit default port
EXPOSE 8501

# Health check — Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
