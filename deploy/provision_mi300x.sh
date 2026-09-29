#!/usr/bin/env bash
# Provisions a single AMD Developer Cloud MI300X droplet to serve the OCR
# vision model behind an OpenAI-compatible endpoint, then runs a smoke test.
#
# Prerequisites (done once, outside this script):
#   1. Join the AMD AI Developer Program and activate the $100 Developer
#      Cloud credit: https://developer.amd.com/ai-developer-program/
#   2. Create a "1x MI300X" GPU Droplet from the ROCm base image and note
#      its public IP. An SSH key must already be attached to the droplet.
#
# Usage (from a machine with the droplet's IP and your SSH key):
#   ./deploy/provision_mi300x.sh <droplet-ip> [model-id]
#
# The script is idempotent: re-running it on an already-provisioned host
# pulls the latest image and restarts the container rather than failing.

set -euo pipefail

HOST="${1:?usage: provision_mi300x.sh <droplet-ip> [model-id]}"
MODEL="${2:-PaddlePaddle/PaddleOCR-VL}"
SSH_USER="${TABULA_SSH_USER:-root}"
CONTAINER_NAME="tabula-vllm"
PORT="${TABULA_VLM_PORT:-8000}"

echo "==> Provisioning ${HOST} to serve ${MODEL} on ROCm vLLM"

# shellcheck disable=SC2087
ssh -o StrictHostKeyChecking=accept-new "${SSH_USER}@${HOST}" bash -s <<REMOTE
set -euo pipefail

echo "--> ROCm / GPU visibility check"
rocm-smi --showproductname || { echo "rocm-smi not found — is this a ROCm image?" >&2; exit 1; }

echo "--> Stopping any previous container"
docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true

echo "--> Pulling rocm/vllm:latest"
docker pull rocm/vllm:latest

echo "--> Starting vLLM (guided decoding enabled, xgrammar backend)"
docker run -d --name ${CONTAINER_NAME} \
  --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined \
  --restart unless-stopped \
  rocm/vllm:latest \
  vllm serve "${MODEL}" \
    --host 0.0.0.0 --port ${PORT} \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --guided-decoding-backend xgrammar

echo "--> Waiting for the server to report healthy (model load can take a few minutes)"
for _ in \$(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "    healthy"
    break
  fi
  sleep 5
done
REMOTE

echo "==> Smoke test from this machine"
curl -sf "http://${HOST}:${PORT}/v1/models" | python3 -m json.tool || {
  echo "Smoke test failed — check 'docker logs ${CONTAINER_NAME}' on ${HOST}" >&2
  exit 1
}

cat <<EOF

==> Ready. Point the API service at this endpoint:

    export TABULA_VLM_BASE_URL="http://${HOST}:${PORT}/v1"
    export TABULA_VLM_MODEL="${MODEL}"

Record utilisation and cost while you work:

    ssh ${SSH_USER}@${HOST} 'rocm-smi --showuse --showmemuse'

Destroy the droplet when you are done for the session — credits are billed
hourly whether or not the GPU is in use.
EOF
