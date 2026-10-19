# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10
WORKDIR /app

# Dependencies first: this layer is cached until requirements change.
# The pip cache lives in a BuildKit mount, so a failed build doesn't re-download
# everything and the cache never ends up in the image.
COPY requirements.txt requirements-ui.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt -r requirements-ui.txt

# NER model used by Presidio, in its own layer so a flaky GitHub download
# doesn't invalidate the dependency layer above.
ARG SPACY_MODEL_URL=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "$SPACY_MODEL_URL"

COPY matcher ./matcher
COPY api ./api
COPY ui ./ui
COPY scripts ./scripts
COPY data/sample ./data/sample

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
