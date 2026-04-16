#!/usr/bin/env bash
#
# Download models from Hugging Face

set -o errexit
set -o nounset
set -o pipefail
# set -o xtrace

TXT_GREEN="\e[32m"
TXT_CLEAR="\e[0m"

BASE_DIR=/mnt/data0/llm/models

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  exit 1
fi
HEADER="Authorization: Bearer $HF_TOKEN"

MODEL_URLS=(
  "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf?download=true"
  "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/mmproj-F16.gguf?download=true"
)

download_model() {
  local URL="$1"

  # Strip the scheme+host
  local _url_path="${URL#*://huggingface.co/}"

  # Grab the "org/model" part (everything before the first /resolve)
  local _org_model="${_url_path%%/resolve*}"
  local DIR="$BASE_DIR/$_org_model"

  # Take the tail of the path
  local _filename="${_url_path##*/}"
  # Remove the query (everything from "?" on)
  local OUT="${_filename%%\?*}"

  echo -e "${TXT_GREEN}###### Starting download for: ${URL}${TXT_CLEAR}"

  aria2c \
    --min-split-size=1M \
    --max-connection-per-server=16 \
    --split=16 \
    --max-concurrent-downloads=1 \
    --header="$HEADER" \
    --dir="$DIR" \
    --out="$OUT" \
    "$URL"

  echo -e "${TXT_GREEN}###### Completed: $OUT"
}

for URL in "${MODEL_URLS[@]}"; do
  download_model "$URL"
done
