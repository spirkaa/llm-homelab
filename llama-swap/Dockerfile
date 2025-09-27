FROM ghcr.io/mostlygeek/llama-swap:cpu

COPY --from=docker.io/library/docker:cli /usr/local/bin/docker /usr/local/bin/docker

HEALTHCHECK CMD curl -f http://localhost:8080/ || exit 1

ENTRYPOINT ["/app/llama-swap", "-config", "/app/config/config.yaml", "--watch-config"]
