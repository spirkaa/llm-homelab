# hadolint global ignore=DL3006,DL3008,DL3013

ARG BUILD_IMAGE=ghcr.io/astral-sh/uv:python3.14-trixie
ARG RUNTIME_IMAGE=ghcr.io/astral-sh/uv:python3.14-trixie-slim

FROM ${BUILD_IMAGE} AS build

ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

SHELL [ "/bin/bash", "-euxo", "pipefail", "-c" ]

ARG UV_ARGS=--no-dev

COPY pyproject.toml uv.lock /

RUN uv sync --locked ${UV_ARGS}


FROM ${RUNTIME_IMAGE} AS runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

SHELL [ "/bin/bash", "-euxo", "pipefail", "-c" ]

RUN addgroup --system --gid 1000 app \
    && adduser --system --uid 1000 --ingroup app app

RUN apt-get update \
    && apt-get install -y --no-install-suggests --no-install-recommends \
        wget \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build ${UV_PROJECT_ENVIRONMENT} ${UV_PROJECT_ENVIRONMENT}
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

WORKDIR /app

COPY --chown=app:app app.py ./

# hadolint ignore=DL3066
USER app

EXPOSE 8081

ENTRYPOINT [ "python", "app.py" ]
