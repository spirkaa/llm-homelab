FROM docker:29-cli AS docker-cli


FROM ghcr.io/mostlygeek/llama-swap:cpu

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

HEALTHCHECK --interval=10s --timeout=1s CMD ["curl", "-f", "http://localhost:8080/health"]

ENTRYPOINT ["/app/llama-swap", "-config", "/app/config/config.yaml", "--watch-config"]
