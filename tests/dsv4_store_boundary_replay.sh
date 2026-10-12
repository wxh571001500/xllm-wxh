#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

RUN_DIR="${1:?usage: dsv4_store_boundary_replay.sh RUN_DIR}"
: "${DSV4_FUSED_DIR:?set DSV4_FUSED_DIR to the fused test directory}"
: "${DSV4_MODEL_PATH:?set DSV4_MODEL_PATH to the model directory}"
: "${STORE_LOCAL_HOSTNAME:?set STORE_LOCAL_HOSTNAME to the Store host:port}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUSED="$DSV4_FUSED_DIR"
PROBE="${DSV4_BOUNDARY_PROBE:-$SCRIPT_DIR/dsv4_store_boundary_probe.py}"
MODEL_PATH="$DSV4_MODEL_PATH"
MODEL_ID="${DSV4_MODEL_ID:-DeepSeek-V4-Flash}"
PORT="${DSV4_SERVICE_PORT:-18110}"
START_DEVICE="${DSV4_START_DEVICE:-0}"
TP_SIZE="${DSV4_TP_SIZE:-8}"

source "$FUSED/common.sh"
cd "$FUSED"
echo "[$(date +%T)] stop xLLM only; preserve Mooncake"
kill_all_xllm >/dev/null || true
sleep 3

echo "[$(date +%T)] relaunch xLLM for boundary replay"
DEVICE_NAMES= \
MC_FORCE_TCP=1 \
HOST_BLOCKS_FACTOR=4 \
STORE_PROTOCOL=tcp \
STORE_MASTER_SERVER_ADDRESS=127.0.0.1:50051 \
STORE_METADATA_SERVER=P2PHANDSHAKE \
STORE_LOCAL_HOSTNAME="$STORE_LOCAL_HOSTNAME" \
PREFETCH_BATCH_SIZE=8 \
PREFETCH_TIMEOUT=30000 \
HCCL_IF_BASE_PORT=47000 \
MASTER_PORT=14110 \
VLOG_LEVEL=1 \
ENABLE_PROFILE_STEP_TIME=false \
MAX_TOKENS_PER_CHUNK_FOR_PREFILL=2048 \
"$FUSED/launch.sh" dsv4-flash \
  --tp "$TP_SIZE" --mooncake-store --log-dir "$RUN_DIR" \
  --start-device "$START_DEVICE" --port "$PORT"
wait_ready "$MODEL_ID" "$PORT" 180 5

echo "[$(date +%T)] send exact boundary replay prompts"
python3 "$PROBE" \
  --model-path "$MODEL_PATH" \
  --output-dir "$RUN_DIR/boundary" \
  --phase replay \
  --endpoint "http://127.0.0.1:$PORT" \
  --model "$MODEL_ID" \
  --marker-prefix "${DSV4_MARKER_PREFIX:-dsv4-store-boundary}"
sleep 5

for file in "$RUN_DIR"/tp"$TP_SIZE"_rank*.log; do
  [[ -f "$file" ]] || continue
  rank="$(basename "$file" | sed -E "s/tp${TP_SIZE}_rank([0-9]+)\\.log/\\1/")"
  cp "$file" "$RUN_DIR/boundary_replay_rank${rank}.log"
done
echo "[$(date +%T)] boundary replay complete"
