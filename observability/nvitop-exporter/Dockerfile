FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-suggests --no-install-recommends \
        wget \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/*

RUN uv tool install nvitop-exporter

EXPOSE 5050

ENTRYPOINT ["uvx", "--quiet", "--offline", "nvitop-exporter"]
