#!/bin/bash
# C4 LCB held-out 评测串跑（预注册见 LCB_HELDOUT.md §三）
# 用法：在宿主机执行 bash run_lcb_evals.sh <GPU_ID>，逐 checkpoint 顺序评测
set -e
GPU=${1:-1}
IMG=1cat-vllm:v100-1.3.0
# WORK 换成本机仓库路径（与 c4 各脚本 docstring 的 <user> 占位同义）
WORK=/mnt/storage/<user>/claudecode_projects/agentic_rl
COMMON=(-v /mnt/nas3/shared/model:/models:ro -v "$WORK:/work" -w /work -e HF_HUB_OFFLINE=1)

run () {  # run <日志名> <镜像内命令...>
  local name=$1; shift
  echo "=== [$(date +%H:%M:%S)] $name ==="
  docker run --rm --gpus "\"device=$GPU\"" --security-opt seccomp=unconfined \
    "${COMMON[@]}" "$IMG" "$@" > "logs/$name.log" 2>&1
  tail -3 "logs/$name.log"
}

# ① 基座直答 oneshot（背题率测量；filter 输出到隔离目录，防覆盖训练池）
run c4_lcb_base_oneshot python c4_baseline.py --tasks data/lcb/tasks.jsonl \
  --hidden data/lcb/hidden.jsonl --out logs/c4_lcb_base_oneshot.json \
  --filter_out data/lcb/filter_isolated
# ② 基座多轮执行反馈
run c4_lcb_base_multiturn python c4_rollout.py --model /models/Qwen3-4B-Instruct-2507 \
  --tasks data/lcb/tasks.jsonl --hidden data/lcb/hidden.jsonl \
  --n 21 --temperature 0 --out logs/c4_lcb_base_multiturn.json --show -1
# ③ RFT merged
run c4_lcb_rft_multiturn python c4_rollout.py --model checkpoints/c4_rft_merged \
  --tasks data/lcb/tasks.jsonl --hidden data/lcb/hidden.jsonl \
  --n 21 --temperature 0 --out logs/c4_lcb_rft_multiturn.json --show -1
# ④ GRPO iter1（epi 线部署形态）
run c4_lcb_grpo_iter1_multiturn python c4_rollout.py --model checkpoints/c4_grpo/iter1_merged \
  --tasks data/lcb/tasks.jsonl --hidden data/lcb/hidden.jsonl \
  --n 21 --temperature 0 --out logs/c4_lcb_grpo_iter1_multiturn.json --show -1
# ⑤ GiGPO iter0 / iter1
run c4_lcb_gigpo_iter0_multiturn python c4_rollout.py --model checkpoints/c4_gigpo/iter0_merged \
  --tasks data/lcb/tasks.jsonl --hidden data/lcb/hidden.jsonl \
  --n 21 --temperature 0 --out logs/c4_lcb_gigpo_iter0_multiturn.json --show -1
run c4_lcb_gigpo_iter1_multiturn python c4_rollout.py --model checkpoints/c4_gigpo/iter1_merged \
  --tasks data/lcb/tasks.jsonl --hidden data/lcb/hidden.jsonl \
  --n 21 --temperature 0 --out logs/c4_lcb_gigpo_iter1_multiturn.json --show -1

echo "=== all evals done $(date +%H:%M:%S) ==="
