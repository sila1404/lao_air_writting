# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm AS builder

RUN pip install --no-cache-dir uv && \
    apt-get update && apt-get install --no-install-recommends -y binutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy

# Install dependencies first so this layer is cached across source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --extra api --no-install-project --frozen

# Only the source actually imported by the API at runtime.
COPY src/lao_air_writting/__init__.py src/lao_air_writting/api.py src/lao_air_writting/
COPY src/utils/ src/utils/
COPY src/assets/fonts/ src/assets/fonts/
RUN uv sync --extra api --frozen

# clang/grpc are unused hard deps of tensorflow (doc/distributed tooling, never
# imported for plain keras inference - verified via a live predict() call);
# tensorflow/include is C++ headers, irrelevant to the Python runtime. Strip
# debug symbols from the rest. Cuts the venv roughly in half. Must run after
# the last `uv sync` - syncing again would reinstall everything from the lock.
RUN rm -rf .venv/lib/python3.11/site-packages/clang \
    .venv/lib/python3.11/site-packages/libclang*.dist-info \
    .venv/lib/python3.11/site-packages/grpc \
    .venv/lib/python3.11/site-packages/grpcio*.dist-info \
    .venv/lib/python3.11/site-packages/tensorflow/include && \
    find .venv -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null; \
    find .venv -iname "__pycache__" -exec rm -rf {} + 2>/dev/null; true

COPY model/ model/


FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
RUN chown appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /app/.venv .venv
COPY --from=builder --chown=appuser:appuser /app/src src
COPY --from=builder --chown=appuser:appuser /app/model model

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

CMD ["uvicorn", "lao_air_writting.api:app", "--host", "0.0.0.0", "--port", "8000"]
