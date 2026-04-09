#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CLIENT_COUNT="$(python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path("client.json").read_text())
print(config.get("active_client_count", len(config["Ability"]["ability"])))
PY
)"

export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

for ((node_id=0; node_id<CLIENT_COUNT; node_id++))
do
    echo "start $node_id"
    if [[ -n "${CUDA_VISIBLE_DEVICES+x}" && -n "${CUDA_VISIBLE_DEVICES}" ]]; then
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python client.py --node_id "$node_id" &
    else
        python client.py --node_id "$node_id" &
    fi
done

wait
echo "finish"
