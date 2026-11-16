FROM python:3.11-slim-bookworm

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install PyTorch from the CUDA 12.6 index and drop it from the requirements
# pass: the plain PyPI wheel is built for CUDA 13, which no longer ships kernels
# for Pascal cards such as the GTX 1050 Ti. Leaving torch in requirements.txt
# would let pip "upgrade" it straight back to that CUDA 13 wheel.
RUN grep -v -E '^torch([^a-zA-Z0-9_.-]|$)' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126 \
    && pip install --no-cache-dir -r /tmp/requirements-no-torch.txt

COPY app.py ./

ENV ML_PORT=18082
ENV HF_HOME=/app/hf_cache
RUN mkdir -p /app/hf_cache

EXPOSE 18082
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "18082"]
