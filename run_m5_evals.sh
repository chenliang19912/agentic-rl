#!/bin/bash
# M5-A1 训练后评测：4 个 merged checkpoint 各跑 dev bridge n=500 贪心
# 对照锚点（run3，同冷启动同数据切片同超参，仅协议不同）：
#   run3-iter0 EM 26.60 / iter1 26.20 / iter2 23.80；预注册判据见 m5_env.py 文件头
set -e
GPU=${1:-1}
IMG=1cat-vllm:v100-1.3.0
# 默认占位路径；实际运行用环境变量覆盖：WORK=/mnt/storage/<你的路径> bash run_m5_evals.sh
WORK="${WORK:-/mnt/storage/<user>/claudecode_projects/agentic_rl}"
COMMON=(-v /mnt/nas3/shared/model:/models:ro -v "$WORK:/work" -w /work -e HF_HUB_OFFLINE=1)

# 对照组：run3-iter0 冻结策略 + M5 SYSTEM（隔离"prompt 提及 k"的效应）
for it in 0 1 2 3; do
  echo "=== [$(date +%H:%M:%S)] m5_iter${it} eval ==="
  docker run --rm --gpus "\"device=$GPU\"" --security-opt seccomp=unconfined \
    "${COMMON[@]}" "$IMG" \
    python m5_rollout.py --model checkpoints/m5/iter${it}_merged \
      --data eval --n 500 --temperature 0 \
      --out logs/m5_eval_iter${it}.json > "logs/m5_eval_iter${it}.log" 2>&1
  /bin/grep -A 12 '"model"' "logs/m5_eval_iter${it}.log" | head -13
done
echo "=== 对照组：run3-iter0 + M5 SYSTEM ==="
docker run --rm --gpus "\"device=$GPU\"" --security-opt seccomp=unconfined \
  "${COMMON[@]}" "$IMG" \
  python m5_rollout.py --model checkpoints/m3/iter0_merged \
    --data eval --n 500 --temperature 0 \
    --out logs/m5_eval_run3iter0_m5sys.json > "logs/m5_eval_run3iter0_m5sys.log" 2>&1
/bin/grep -A 12 '"model"' "logs/m5_eval_run3iter0_m5sys.log" | head -13
echo "=== all M5 evals done $(date +%H:%M:%S) ==="
