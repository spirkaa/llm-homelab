# hadolint global ignore=DL3008

FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/*

RUN uv tool install nvitop-exporter

EXPOSE 5050

ENTRYPOINT ["uvx", "--quiet", "--offline", "nvitop-exporter"]
