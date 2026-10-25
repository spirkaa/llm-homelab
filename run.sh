#!/usr/bin/env bash
#
# Build and run llm-homelab.

set -o errexit
set -o nounset
set -o pipefail
# set -o xtrace

# TXT_RED="\e[31m"
TXT_GREEN="\e[32m"
TXT_YELLOW="\e[33m"
TXT_CLEAR="\e[0m"

UBUNTU_VERSION=24.04
CUDA_VERSION=13.3.0
CUDA_DOCKER_ARCH=86

run_with_retry() {
  local description="$1"
  shift
  if "$@"; then
    return 0
  fi
  echo -e "${TXT_YELLOW}!!! ${description} failed, retrying in 5s...${TXT_CLEAR}"
  sleep 5
  "$@"
}

git pull --rebase --autostash

echo ""
echo -e "${TXT_GREEN}###### Building llama.cpp ######${TXT_CLEAR}"
mkdir -p llama.cpp
cd llama.cpp || exit
git clone https://github.com/ggml-org/llama.cpp.git . || (git reset --hard && git pull --rebase)
cp ../llama.cpp-docker/.devops/cuda.Dockerfile .devops/cuda.Dockerfile
run_with_retry "docker build (llama.cpp)" docker build \
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
echo -e "${TXT_GREEN}###### Building Strata ######${TXT_CLEAR}"
mkdir -p Strata
cd Strata || exit
# git clone https://github.com/Niko1221/Strata . || (git reset --hard && git pull --rebase)
sed -i "s|FROM nvidia/cuda:[0-9.]*-devel-ubuntu[0-9.]*|FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}|" Dockerfile
run_with_retry "docker build (Strata)" docker build \
  -t git.devmem.ru/projects/llm-homelab/strata:latest \
  --build-arg CUDA_ARCHITECTURES=$CUDA_DOCKER_ARCH \
  --progress=plain \
  .
git reset --hard
cd ..

echo ""
echo -e "${TXT_GREEN}###### Building HyperQwen ######${TXT_CLEAR}"
mkdir -p hyperqwen
cd hyperqwen || exit
git clone https://github.com/syv-ai/HyperQwen . || (git reset --hard && git pull --rebase)
run_with_retry "docker build (HyperQwen)" docker build \
  -t ghcr.io/syv-ai/hyperqwen:latest \
  --progress=plain \
  .
cd ..

echo ""
echo -e "${TXT_GREEN}###### Pulling club-3090 ######${TXT_CLEAR}"
mkdir -p club-3090
cd club-3090 || exit
git clone https://github.com/noonghunna/club-3090 . || git pull --rebase --autostash
cd ..

echo ""
echo -e "${TXT_GREEN}###### Running observability ######${TXT_CLEAR}"
cd observability || exit
run_with_retry "docker compose up (observability)" docker compose up -d --build
cd ..

echo ""
echo -e "${TXT_GREEN}###### Running llm-homelab ######${TXT_CLEAR}"
run_with_retry "docker compose up (llm-homelab)" docker compose up -d --build
