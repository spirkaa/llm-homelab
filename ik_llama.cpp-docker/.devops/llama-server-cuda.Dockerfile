# hadolint global ignore=DL3006,DL3008,DL3013

ARG UBUNTU_VERSION=24.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=13.2.0
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# Unless otherwise specified, we make a fat build.
ARG CUDA_DOCKER_ARCH=default

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential cmake git libcurl4-openssl-dev ccache

WORKDIR /app

COPY . .

ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
ENV CCACHE_DIR=/root/.cache/ccache
ENV PATH=/usr/lib/ccache:$PATH

# hadolint ignore=SC2046
RUN --mount=type=cache,target=/root/.cache/ccache \
    if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 ${CMAKE_ARGS} && \
    cmake --build build --config Release --target llama-server -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so" -exec cp -v {} /app/lib \;


FROM ${BASE_CUDA_RUN_CONTAINER} AS server

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/lib/ /app
COPY --from=build /app/build/bin/llama-server /app/llama-server

WORKDIR /app

HEALTHCHECK --interval=10s --timeout=1s CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
