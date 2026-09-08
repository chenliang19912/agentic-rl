# 自研轻量 Agentic RL 训练栈（多轮 GRPO）+ 两个可插拔环境

> 单卡 V100（sm_70）上从零自研的**多轮工具调用 Agentic RL 训练栈**，配套两个可插拔环境：
> **检索环境**（多跳问答 `<search>`）与**代码执行环境**（多轮修码 `<run_code>`）。
> 核心论点：**环境是可插拔组件、训练层零改动复用**——同一套多轮可验证奖励 GRPO 方法横跨两类环境成立。

全部数字均为真实跑出（8×V100，Qwen3-4B）。本仓库只开源**代码 + 实验日志**；模型权重、检索索引、
数据集 parquet、训练日志等大体积产物不入仓（见 `.gitignore`，按下方命令重建）。

---

## 为什么自研（不用 verl / OpenRLHF / LLaMA-Factory）

- **V100 sm_70 无可用 kernel**：verl 官方镜像的预编译 torch 不含 sm_70，实测预研记录在案；
- **多轮 rollout 需与训练交替**且要自定义环境回复格式，现成框架的多轮接口约束多。

自研栈的三件核心工程（两环境共用）：
1. **vLLM 锁步多轮 rollout**：批量生成 → 解析工具标签 → 环境调用 → messages 追加 → next_active，
   单卡上 vLLM 推理与 HF/LoRA 训练交替；
2. **disable_adapter 零显存参考 KL**：参考模型 = 关闭 adapter 的同一份权重，不额外占显存；
3. **稳定性纪律**：组内归一化优势（GRPO）、零优势过滤（DAPO 动态采样同源）、作答/提交率熔断、
   逐轮轨迹落盘（崩溃可复盘）。

---

## 代码修码 Agent RL（多轮修码，可验证奖励）

把检索 Agent 项目的检索环境换成**代码执行沙盒**：模型看失败测试 → `<run_code>` 执行 → 看 stdout → 再修，
上限 5 轮，以 `<answer>` 提交终版；奖励 = 隐藏测试逐 case 通过率（稠密）。

### 环境设计（~100 行核心，无容器）

- **禁网** `unshare -rn` + **限时限内存禁写** `RLIMIT_CPU/AS(4GB)/FSIZE(0)` + 输出截断 + 最小 env；
- 与 2026 去 Docker 化趋势同向（SWE-MiniSandbox, ICML 2026）；
- **关键坑**：变异体验证必须在 **fork 子进程 + RLIMIT** 里跑——collatz 类变异（`3n+1→3n-1`）
  产生 bigint 指数爆炸，单次 C 级乘法进程内 SIGALRM 拦不住（实测膨胀到 46.8GiB 触发宿主机 OOM）。
  详见 [`C4_README.md`](C4_README.md) 坑记录 9。

### 奖励与防 reward hacking

`R = 1.0*r_pass + 0.2*r_format − 0.2*r_over`
- `r_pass` = 隐藏测试逐 case 通过率（稠密，缓解组内优势全零）；**终止门控**：无合法 `<answer>` → 0；
- **三道防作弊门**：① 禁网断远程泄题；② 隐藏测试只读、永不进 prompt、篡改判 0；
  ③ **查重门**：提交代码与参考解归一化相同 → 判 0（封堵逐字背诵污染）；
- 过程奖励默认不加（μCode"一步可恢复 MDP"理论，ICML 2025, arXiv 2502.20380）。

### 关键结果（89 净题训练子集，贪心解码）

**benchmark 污染一手实测**：Qwen3-4B-Instruct-2507 对 MBPP+ ∪ HumanEval+ 污染率 **76.5%**
（直答满分或逐字背参考解；MBPP+ 74.3% / HumanEval+ 81.0%），判 0 门 + 背题筛选后训练仅留 89 净题。

**三组对照 + GRPO 训练进程曲线**：

| 模型 | avg_pass（贪心） | submit | format_ok | avg_runs | dup |
|---|---|---|---|---|---|
| 基座直答（无执行反馈） | 0.2370 | — | — | — | — |
| 基座多轮（执行反馈，**+22.8pt**） | 0.4648 | 77.5% | 77.5% | 1.27 | 12 |
| RFT 多轮（93 正例冷启动，目标线） | 0.7295 | 88.8% | 88.8% | 1.76 | 8 |
| GRPO iter0 | 0.7305 | 88.8% | 88.8% | 1.82 | 6 |
| **GRPO iter1（峰值，+2.2pt over RFT）** | **0.7516** | **100%** | **100%** | 1.57 | 6 |
| GRPO iter2 | 0.7436 | 100% | 100% | 1.60 | 6 |
| GRPO iter3 | 0.7435 | 100% | 100% | 1.44 | 6 |

- **多轮 GRPO 超过 RFT 目标线**：贪心峰值 iter1 **0.7516（+2.2pt）**，对基座 +27.9pt（相对 60%）；
  格式/提交率一步到 100%，修复轮数收敛（1.82→1.44）。
- **修复行为深挖（`c4_deepdive.py`，配对判分修正自身叙事）**：轮数收敛 **不是**"学会修"——
  一次修对率 29.2%→**65.2%**（修复需求下降），而失败后修复成功率 11.1%→**0%**（小样本）。
  配对 Δ（首跑沙盒重判 vs 终版，对称 dup 门控）：基座在"首跑已过可见测试"上**过度编辑丢 9.3pt**，
  GRPO 把这个损失通道收敛到 **0.0**（学会"跑一次验证、过了原样提交"）；训练收益大头在**首版质量**
  （首跑 gated 0.6344→**0.7512**，+11.7pt）。F 分支（首跑失败 15~18 题、终版仅 0.07~0.17）是全部 headroom。
- **训练分 ≠ 部署分**：训练分布（temp 1.0）avg_pass 单调升 0.7065→0.7354，但**确定性贪心 iter1 见顶后平台**
  ——与检索 Agent 项目"贪心/采样口径背离"同现象**跨环境复现**，部署按贪心口径选 **iter1**。
  iter1→iter3 题级 diff：掉分**仅 1 题**、单题噪声 ±1.1pt > 回落 0.8pt，故为"iter1 起进入平台"而非系统性过拟合。
- **RL 收益形态由任务结构决定**（两环境镜像，一手日志对照）：检索 Agent 项目多跳 QA"一搜即对率 0%、天然多步"→
  RL 提升**多步内终答质量**（多搜 EM 1.40%→26.71%）；代码 Agent 项目函数级修码"一步可恢复"→ RL 最大化**一次成功率**。
- **reward hacking 无演化**：dup/grade_timeout 全程钉死 6/2，逐字 dup 全判 0，
  difflib 相似度抽查未见"改写绕查重"。

完整实验日志（M0 环境原型 → M1 基线 → M2 GRPO → M3 评测消融 + 10 条坑记录）：[`C4_README.md`](C4_README.md)。

---

## 检索 Agent RL（多跳问答，多轮 `<search>`）

同一套训练栈的检索环境：Qwen3-4B 训成"检索—推理—再检索"的多步 Agent（HotpotQA / Bamboogle）。

- **结果**：多跳 QA EM 由无检索 **4.20 → 26.60**（F1 13.15→37.04，n=500 贪心），
  超同检索器单次真实 RAG（13.80）**+12.8pp**，距 oracle 上界 3.4pp；OOD Bamboogle EM 20.0；
  RL 消融 +21.2pp；作答率 4.4%→99.6%，平均搜索 3.96→2.65（学会"少搜早答"）。
- **过程奖励门控教训**：不门控时"搜而不答"白拿过程分，一轮更新作答率 20%→0.2%（坍缩）；
  门控后一轮抬到 86%。
- 完整日志：[`M0_README.md`](M0_README.md)。

---

## 规模化与引擎优化（2026-09 迭代轮）

对标后训练工程的三个硬课题（规模放大 / 多机多卡 / vLLM·SGLang 深度定制），
在 8×V100 边界内做的迭代轮——**每个数字实测、估算显式标注**：

- **vLLM 深度定制实证**（`poc_vllm_bench.py`，`logs/poc_vllm/`）：CUDA graph 在
  sm_70 上 89 题交付负载 generate wall **−35.5%**（290→443 out tok/s）且贪心
  **0/89 翻转**（预注册不变性判据）；prefix caching V1 默认态审计（"没配置"≠"没开"）
  与 KV 容量约束（101k KV vs ~222k 并发会话前缀）；六配置矩阵+跨 run 噪声 ~10% 口径。
- **共置量化**：训练 iter 全环 rollout 相 **78.8%** vs 训练相 21.2%——
  "rollout 是第一瓶颈"的单卡实测；换载税本体仅 ~2–3%。
- **vLLM EngineCore fork 死锁排查**（[`ENGINE_HANG_RCA.md`](ENGINE_HANG_RCA.md)）：
  训练循环内建引擎 4/4 挂死 → 全新容器二分 E1-E12 → 模块组合触发的
  fork-while-threaded 死锁 → `VLLM_WORKER_MULTIPROC_METHOD=spawn` 修复。
- **GiGPO 式 turn 级优势**（`c4_gigpo.py` / `c4_train_gigpo.py`，单测 31/31）：
  Stage 0 离线反事实（"新增信号=0 是数学必然，真实机理是 mixed 组内信用重分配"）、
  Stage 1.5 配对 replay 完美单变量对照（R1−R0=0.0000，GATE-PASS-NO-SIGNAL）、
  Stage 2 全量 4 iters（`logs/poc_gigpo/`，详见 `C4_README.md` §三-D）。
- **规模放大方案**（[`SCALE_UP.md`](SCALE_UP.md)）：实测锚点表 + verl 组件级
  迁移映射（≈1.5–2 周）+ 8B/30B-MoE/70B 三档外推模型（假设显式）+
  千卡 disaggregated 预案（设计级，如实标注）+ 工程要求映射表。

---

## 代码结构

```
共享训练栈（两环境复用）
  m3_env.py            GRPO 组内归一化优势（环境无关，代码 Agent 项目直接 import 复用）
  mask_utils.py        encode_chat_assistant_only（仅 assistant token 计损失）
  m2_merge.py          LoRA merge 到 base

代码 Agent 项目：代码执行环境（c4_*）
  c4_env.py            无容器沙盒（unshare 禁网 + rlimit）+ 稠密奖励 + 判分
  c4_tasks.py          AST 单点变异注入 bug + MBPP+/HumanEval+ 双源任务构建
  c4_test_env.py       25 项 CPU 单测（沙盒防御 / 判分 / 防泄漏审计）
  c4_rollout.py        vLLM 锁步多轮 rollout（<run_code> 反馈进 messages）
  c4_baseline.py       基座直答基线 + 背题筛选
  c4_rft.py            RFT：采样 → 正例过滤 → LoRA SFT 冷启动
  c4_train.py          多轮 GRPO 训练循环（训练层逐行复用 m3_train.py）
  c4_fix_dup.py        task_id 撞车修复（见坑记录 10）
  c4_deepdive.py       修复行为深挖（Phase1 轨迹统计 + Phase2 首跑沙盒重判分配对，纯日志无 GPU）
  poc_vllm_bench.py    WS-1 vLLM 定制实证（A0 冒烟/A1 六配置矩阵+不变性/计时代理）
  c4_gigpo.py          GiGPO 式 turn 级优势（锚点=(task, 可见判定 P/F/E 全前缀)）
  c4_train_gigpo.py    turn 级优势训练臂（c4_train 姊妹拷贝，超参全不动，含 --replay）
  poc_gigpo_offline.py Stage 0 离线反事实（γ/ω 网格、锚点密度、信用重分配统计）
  poc_gigpo_test.py    31 项 CPU 单测（spans 对齐/恒等性/锚点分组）
  poc_gigpo_compare.py Stage 1.5/2 配对对比器（修复指标族+隐藏判分+预注册判定）

检索 Agent 项目：检索环境（m*_* / zh_*）
  m0_rollout.py m1_baseline*.py m2_sft_*.py m3_prep_rl_data.py
  m3_rollout.py m3_train.py m3_train_reinforce.py build_bm25_index.py textproc.py
  zh_*.py              中文迁移子实验（字 bigram 分词 + 四级门控）
```

---

## 复现（代码 Agent 项目）

> 长任务铁律：在 docker 内跑必须 `--security-opt seccomp=unconfined`（否则默认 seccomp 拦截
> `unshare -rn`，禁网静默降级）；长训练用 `docker run -d --name xxx`（detached，生命周期与会话解耦，
> 切勿 `--rm` + 后台/管道——docker client 被杀会连坐容器静默死亡，见坑记录 8）。

```bash
# 0. 数据：下载 EvalPlus 的 MBPP+ / HumanEval+ parquet 放到 data/mbpp/
#    mbppplus.parquet, humanevalplus.parquet

# 1. 沙盒单测（25 项，纯 CPU）
python c4_test_env.py

# 2. 构建任务池（AST 单点变异注入 bug + 判分表）
python c4_tasks.py --n 400 --seed 42 \
  --parquets data/mbpp/mbppplus.parquet,data/mbpp/humanevalplus.parquet
#    → data/c4/tasks.jsonl + hidden.jsonl

# 3. 基座直答基线 + 背题筛选（剔除污染题 → 89 净题训练池）
python c4_baseline.py --model Qwen/Qwen3-4B-Instruct-2507
#    → data/c4/tasks_train.jsonl + hidden_train.jsonl

# 4. 基座多轮贪心基线（执行反馈价值）
python c4_rollout.py --model Qwen/Qwen3-4B-Instruct-2507 \
  --tasks data/c4/tasks_train.jsonl --hidden data/c4/hidden_train.jsonl \
  --temperature 0 --G 1 --out logs/c4_baseline_multiturn.json

# 5. RFT 冷启动（采样 → 正例过滤 → LoRA SFT）
python c4_rft.py --model Qwen/Qwen3-4B-Instruct-2507
#    → checkpoints/c4_rft_merged

# 6. 多轮 GRPO 训练（训练层复用 m3_train.py）
python c4_train.py --base checkpoints/c4_rft_merged --n-iters 4 --group-size 4
#    → checkpoints/c4_grpo/iter{0..3}_merged + stats.jsonl

# 7. 贪心评测（按交付口径选点 = iter1）
python c4_rollout.py --model checkpoints/c4_grpo/iter1_merged \
  --tasks data/c4/tasks_train.jsonl --hidden data/c4/hidden_train.jsonl \
  --temperature 0 --G 1 --out logs/c4_grpo_iter1_multiturn.json
```

基座模型路径按实际环境替换（本仓库实测用本地挂载的 Qwen3-4B-Instruct-2507，HF_HUB_OFFLINE=1）。

---

## 坑记录精华（后人勿踩，完整 10 条见 C4_README.md §五）

1. **docker 必须 `seccomp=unconfined`**：默认 seccomp 拦截 `unshare -rn`，禁网静默降级。
2. **`--rm` + 后台/管道 = 容器静默死亡**：docker client 被杀触发容器 stop+rm，无 traceback、
   表象酷似崩溃。长任务用 `docker run -d --name xxx`，产物走 `-v` 挂载。
3. **变异体验证必须在 rlimit 子进程里跑**：collatz bigint 爆炸，进程内 SIGALRM 拦不住 C 级乘法，
   RLIMIT_CPU（内核级 SIGXCPU）+ RLIMIT_AS 才是超时保证。
4. **结构化结果通道必须恒短**：判分输出改聚合计数（非 0/1 数组），否则千级 case 输出超截断长度
   → JSON 解析失败 → 参考解误判 0 分。
5. **按 id join 的文件落盘后必须断言 id 唯一**：HumanEval 不同题共用函数名致 task_id 撞车、
   判分静默用错参考解；对账数字（总数 = 各分支之和）是最便宜的完整性检查。

---

## 与文献的关系

本仓库是这条线在**低算力（单卡 V100）下的独立实践**：RLTF（arXiv 2307.04349，测试反馈在线 RL 开山）
→ CodeRL → μCode（arXiv 2502.20380，一步可恢复 MDP，ICML 2025）→ MURPHY（arXiv 2511.07833，多轮 GRPO）
→ 小模型 RLVR（arXiv 2605.30478）；去 Docker 化沙盒对标 SWE-MiniSandbox（arXiv 2602.11210，ICML 2026）。
不刷单点 SOTA（算力不允许），而是把"轻量可验证环境 + 防作弊 + 跨环境迁移"做通并拿到诚实对照数字。

## License

[MIT](LICENSE) © 2026 Chen Liang (陈亮)
