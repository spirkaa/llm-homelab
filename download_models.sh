#!/usr/bin/env bash
#
# Download models from Hugging Face

set -o errexit
set -o nounset
set -o pipefail
# set -o xtrace

BASE_DIR=/home/spirkaa/.lmstudio/models

HEADER="Authorization: Bearer $HF_TOKEN"

URL="https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF/resolve/main/Qwen3-Next-80B-A3B-Instruct-UD-Q4_K_XL.gguf?download=true"

# Strip the scheme+host
_url_path="${URL#*://huggingface.co/}"
# Grab the "org/model" part (everything before the first /resolve)
_org_model="${_url_path%%/resolve*}"
DIR=$BASE_DIR/$_org_model

# Take the tail of the path
_filename="${_url_path##*/}"
# Remove the query (everything from "?" on)
OUT="${_filename%%\?*}"

aria2c \
  --min-split-size=1M \
  --max-connection-per-server=16 \
  --split=16 \
  --max-concurrent-downloads=1 \
  --header="$HEADER" \
  --dir="$DIR" \
  --out="$OUT" \
  "$URL"
