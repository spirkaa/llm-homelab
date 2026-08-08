#!/usr/bin/env bash
#
# Build and run

set -o errexit
set -o nounset
set -o pipefail
# set -o xtrace

# TXT_RED="\e[31m"
TXT_GREEN="\e[32m"
TXT_CLEAR="\e[0m"

UBUNTU_VERSION=24.04
CUDA_VERSION=13.3.0
CUDA_DOCKER_ARCH=86

git pull --rebase

echo -e "${TXT_GREEN}###### Building llama.cpp ######${TXT_CLEAR}"
mkdir -p llama.cpp
cd llama.cpp || exit
git clone https://github.com/ggml-org/llama.cpp.git . || git pull --rebase
cp ../llama.cpp-docker/.devops/cuda.Dockerfile .devops/cuda.Dockerfile
docker build \
  -t git.devmem.ru/projects/llm-homelab/llama.cpp:server-cuda \
  -f .devops/cuda.Dockerfile \
  --target server \
  --build-arg UBUNTU_VERSION=$UBUNTU_VERSION \
  --build-arg CUDA_VERSION=$CUDA_VERSION \
  --build-arg CUDA_DOCKER_ARCH=$CUDA_DOCKER_ARCH \
  --progress=plain \
  .
git reset --hard
cd ..

echo ""
echo -e "${TXT_GREEN}###### Building ik_llama.cpp ######${TXT_CLEAR}"
mkdir -p ik_llama.cpp
cd ik_llama.cpp || exit
git clone https://github.com/ikawrakow/ik_llama.cpp.git . || git pull --rebase
cp ../ik_llama.cpp-docker/.devops/llama-server-cuda.Dockerfile .devops/llama-server-cuda.Dockerfile
docker build \
  -t git.devmem.ru/projects/llm-homelab/ik_llama.cpp:server-cuda \
  -f .devops/llama-server-cuda.Dockerfile \
  --build-arg UBUNTU_VERSION=$UBUNTU_VERSION \
  --build-arg CUDA_VERSION=$CUDA_VERSION \
  --build-arg CUDA_DOCKER_ARCH=$CUDA_DOCKER_ARCH \
  --progress=plain \
  .
git reset --hard
cd ..

echo ""
echo -e "${TXT_GREEN}###### Running observability ######${TXT_CLEAR}"
cd observability || exit
docker compose up -d --build
cd ..

echo ""
echo -e "${TXT_GREEN}###### Running llm-homelab ######${TXT_CLEAR}"
docker compose up -d --build
