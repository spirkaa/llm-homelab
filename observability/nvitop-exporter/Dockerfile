# hadolint global ignore=DL3008

FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN uv tool install nvitop-exporter

EXPOSE 5050

ENTRYPOINT ["uvx", "--quiet", "--offline", "nvitop-exporter"]
