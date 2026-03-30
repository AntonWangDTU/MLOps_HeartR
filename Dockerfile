# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# ✅ Copy ONLY dependency files first
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

WORKDIR /

# ✅ Install dependencies (this will now be cached!)
RUN uv sync --locked --no-cache --no-install-project

# ❗ Copy the rest AFTER dependencies
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY README.md README.md
COPY LICENSE LICENSE

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "src.mlops_ha.api:app", "--host", "0.0.0.0", "--port", "8000"]
