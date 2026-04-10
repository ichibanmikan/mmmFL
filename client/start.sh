#!/bin/bash

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

# =========================
# 可调参数
# 用法: bash run.sh 4
# 表示使用 4 张 GPU 来跑所有 client
# =========================
USE_GPU_COUNT="${1:-1}"

# 这两个参数供 3 个 non-iid 划分脚本共用
NUM_CLIENTS=20
ALPHA=0.5

CLIENT_COUNT="$(python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path("client.json").read_text())
print(config.get("active_client_count", len(config["Ability"]["ability"])))
PY
)"

# 先准备数据划分
python ./CIFAR/datasets/split_cifar_train_dirichlet_noniid.py --num-clients "$NUM_CLIENTS" --alpha "$ALPHA"
python ./MNIST/datasets/split_mnist_train_dirichlet_noniid.py --num-clients "$NUM_CLIENTS" --alpha "$ALPHA"
python ./FMNIST/datasets/split_fmnist_train_dirichlet_noniid.py --num-clients "$NUM_CLIENTS" --alpha "$ALPHA"

mkdir -p ../server/test_datasets/CIFAR
cp -f ./CIFAR/datasets/test.pickle ../server/test_datasets/CIFAR/

module load cuda/12.8
module load miniforge/25.3.0-3
source activate py312

# =========================
# 获取当前作业可见 GPU 列表
# 优先使用 CUDA_VISIBLE_DEVICES
# 若未设置，则从 nvidia-smi 获取
# =========================
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a ALL_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t ALL_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | awk '{print $1}')
fi

TOTAL_VISIBLE_GPUS="${#ALL_GPUS[@]}"

if [[ "$TOTAL_VISIBLE_GPUS" -eq 0 ]]; then
    echo "[ERROR] No visible GPU found."
    exit 1
fi

if [[ "$USE_GPU_COUNT" -gt "$TOTAL_VISIBLE_GPUS" ]]; then
    echo "[ERROR] Requested $USE_GPU_COUNT GPUs, but only $TOTAL_VISIBLE_GPUS visible: ${ALL_GPUS[*]}"
    exit 1
fi

# 只取前 USE_GPU_COUNT 张卡
GPU_LIST=("${ALL_GPUS[@]:0:USE_GPU_COUNT}")

echo "[INFO] Total visible GPUs: $TOTAL_VISIBLE_GPUS"
echo "[INFO] Using GPUs: ${GPU_LIST[*]}"
echo "[INFO] Client count: $CLIENT_COUNT"

# =========================
# 启动 server
# 注意：如果 server 也会占 GPU，
# 它可能仍然和 client 抢显存
# 如 server 不需要 GPU，建议强制放 CPU
# =========================
SERVER_PID=$!

# =========================
# 启动 client，并均匀分配到不同 GPU
# 每个 client 只看到一张卡
# 这样 client 内部即使默认用 cuda:0，
# 实际也是各自绑定后的“本地唯一 GPU”
# =========================
for ((node_id=0; node_id<CLIENT_COUNT; node_id++))
do
    gpu_idx=$((node_id % USE_GPU_COUNT))
    assigned_gpu="${GPU_LIST[$gpu_idx]}"
    # echo $CLIENT_COUNT
    echo "start client $node_id on physical GPU $assigned_gpu"

    CUDA_VISIBLE_DEVICES="$assigned_gpu" \
        python client.py --node_id "$node_id" &
done

wait
echo "finish"
