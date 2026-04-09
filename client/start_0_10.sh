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

START_ID=0
END_ID=10

if (( CLIENT_COUNT == 0 )); then
    exit 0
fi

LAST_ID=$(( CLIENT_COUNT - 1 ))
if (( END_ID > LAST_ID )); then
    END_ID=$LAST_ID
fi

for ((node_id=START_ID; node_id<=END_ID; node_id++))
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
