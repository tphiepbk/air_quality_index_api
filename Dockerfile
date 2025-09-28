FROM python:3.13.7-slim

ARG PORT=8000
ARG POETRY_VERSION=2.2.0
ARG UVICORN_VERSION=0.35.0

ENV PORT=${PORT}

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}" "uvicorn==${UVICORN_VERSION}"

COPY pyproject.toml poetry.lock* /app/

COPY src /app/src/

COPY models /app/models/

RUN poetry install --only main --no-interaction --no-ansi --no-root || \
    poetry install --no-interaction --no-ansi --no-root

EXPOSE ${PORT}

CMD ["bash", "-c", "exec poetry run uvicorn src.air_quality_index_api.server:app --reload --host 0.0.0.0 --port ${PORT}"]

