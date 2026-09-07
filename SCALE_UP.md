# 规模放大方案：从单卡 V100 自研栈到生产级后训练（检索 Agent 项目/代码 Agent 项目共用）

> **文档定位**：设计级放大方案 + 实测锚点。2026-09 后训练工程的三个硬课题
> （百亿参数级模型经验 / Megatron·DeepSpeed 多机多卡 / vLLM·SGLang 深度定制）
> 逐条映射到本文证据（§6.3 映射表）。
>
> **诚实分级纪律（本文档的命脉）**：每个数字标注证据类型——
> `[实测]` = 有日志/产物路径可现场核验；`[估算]` = 假设 + 推导式，绝不冒充实测；
> `[文献]` = 外部参照。设计级内容明确标注"设计，非实操经历"。
>
> 写作时点：2026-09-07；WS-1（vLLM 定制实证 A0/A1/A2）与 WS-2（GiGPO Stage 0→2）
> 的实测锚点已于同日全部回填完毕（Stage 2 主判据 PASS，§6.4），文档可作为
> 可引用的工程文档；引用时保留各数字的 [实测]/[估算]/[文献] 分级标注。

---

## 0. 一页摘要

- **现状**：8×V100-32G，单卡"交替独占"架构（vLLM rollout → 释放 → HF+LoRA 训练 → merge），
  4B 模型多轮锁步 GRPO，全环 iter ≈ 16–17 分钟 `[实测]`。
- **放大三档路径**：8B（单机，梯度检查点已用）→ 30B-A3B MoE（内部训练平台 MI308X，
  MoE LoRA 训练实操已走通全 expert 命中）→ 70B+（H100/Megatron + disaggregated，设计级）。
- **系统层已实测的定制点**：CUDA graph 在 sm_70 上 89 题交付口径负载 generate 端
  **−35.5% wall（290→443 out tok/s），贪心输出 0 翻转** `[实测 A1]`；
  prefix caching 的真实约束是 **KV 容量 vs 并发会话前缀总量**（101k tokens KV vs
  89 会话×~2.5k ≈ 222k）`[实测+推导]`；A2 共置实测：**训练 iter 全环中
  rollout 相（引擎+采样+沙盒+换载+merge）占 ~79%、训练相仅 ~21%**——
  "rollout 是第一瓶颈"有了单卡实测支撑 `[实测 A2]`。
- **算法层**：turn 级信用分配（GiGPO 式二级分组）已实现并单测 31/31 通过；
  Stage 1.5 配对 replay（同 seed 同 192 子集，完美单变量）实测 **R1(gigpo)−R0(epi)
  = 0.0000**（单 iter 弱更新零行为分化，无害确认）；Stage 2 全量 4 iters
  **预注册主判据 PASS**：贪心峰值 **0.7561**（iter1）/0.7555（iter2）双点超过
  成功线 0.7526（epi 历史峰值 0.7516 单点），贪心修复成功率从 epi 四 iter
  全 0 → **iter0 2/17、iter1 1/15**，iter0 F 分支隐藏修复 Δ +0.0688 = epi 的
  34×；代价=iter3 背题漂移回落 0.7200（3 题全 dup 判 0）。幅度在单 run 噪声
  量级，只宣称"方向一致的初步证据"，不宣称大效应。`[实测]`
  `logs/poc_gigpo/{stage15_compare,stage2_analyze}.json`
- **迁移层**：verl 组件级映射表已备好（§2），"为什么不用 verl"从环境约束的防守答案
  升级为"随时可迁、且自研换来了逐行掌控"的进攻答案。

---

## 1. 现状架构与实测锚点

### 1.1 单卡交替独占架构（当前形态）

```
┌─ iter 循环（c4_train.py，全环 16–19 min [实测，§1.2 两口径]）─────────┐
│ 1. vLLM 引擎构建            17–19s（eager）/ 76s（CUDA graph 编译）[实测] │
│ 2. 锁步多轮 rollout          GPU generate ⇄ CPU 沙盒执行 交替             │
│    （89 题 × G4 = 356 轨迹，每轨迹 1–6 turn，每 turn 全对话重 prefill）    │
│ 3. del llm + empty_cache     换载税小头：引擎重建 18–37s ≈ iter 2–3%     │
│ 4. HF fp16 + LoRA(r32) 训练  192 条有效轨迹 × ACC8，梯度检查点            │
│    （disable_adapter 零显存参考 KL；组内归一优势；零优势过滤）             │
│ 5. merge + save              → 下轮 rollout 的权重                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 实测锚点表

| 锚点 | 数值 | 证据类型 / 路径 |
|---|---|---|
| 训练全环 wall（89题×G4，含 rollout+训练+merge） | ≈16–17 min/iter | [实测] `checkpoints/c4_grpo/iter{0..3}_rollouts.jsonl` 落盘时间戳间隔 |
| KV cache 容量 @util0.7 | 101,040 tokens（8192-len 并发 12.33×） | [实测] A0 引擎日志 `logs/poc_vllm/a0_all.log` |
| KV cache 容量 @util0.9 | 124,144 tokens；显存峰值 29,518 MiB 无 OOM | [实测] 同上（util 阶梯 0.75→0.9 全通过） |
| 引擎加载 | eager 17–19s；CUDA graph（inductor 编译）76s | [实测] A0 SUMMARY `engine_load_wall` |
| generate 吞吐（n=8 冒烟，贪心） | eager 55–56 out tok/s → CUDA graph **120.5 out tok/s（2.15×）** | [实测] A0 SUMMARY |
| 89 题头条（G1 贪心，交付口径） | eager 90–103s → CUDA graph **58.3s（−35.5%）**；290→443 out tok/s；0/89 翻转 | [实测] A1 `compare.json`，§1.4 全矩阵 |
| 训练 iter 分相占比（89题×G4×temp1.0 全环） | eager：rollout+引擎+换载+merge **926.6s（78.8%）** / 训练 249.8s（21.2%），全环 19.6min；graph 臂 18.7min（−4.7%） | [实测] A2 `checkpoints/c4_a2_probe/*/stats.jsonl` |
| save/merge+优势计算基线（replay 无 rollout） | 164.6s（413.7−249.1） | [实测] `checkpoints/c4_replay/stats_R0.jsonl` |
| 多轮负载前缀重叠率（analytic LCP） | 72–73% prompt token 理论可缓存 | [实测] A0 SUMMARY `analytic_cached_frac` |
| prefix caching 默认态 | vLLM V1 引擎默认 **ON**（历史 0.7516 评测即在 ON 下跑出） | [实测] A0 有效配置打印件 |
| 旧单流锚点 | prefill ~23 / decode ~26 tok/s（单流 eager，**不可外推批量**，仅历史参照） | [实测] `M0_README.md` 环境实测表 |
| RFT 采样（89×G8=712 条 temp1.0） | 397s 单卡 | [实测] `M0_README.md` |
| SFT（3 卡 LoRA） | 45 min | [实测] `M0_README.md` |

> 注：A2 探针 iter（19.6/18.7min）与历史交付 run（16–17min，rollouts 落盘时间戳
> 口径）的差值含 spawn 引擎启动、探针插桩与 ~10% 量级跨 run 噪声（§1.4）；
> A2 的用途是**分相占比**而非全环 wall 主张。

### 1.3 多轮工作负载的 token 经济学（为什么多轮 agentic RL 的瓶颈和单轮不同）

- **重 prefill 结构**：锁步第 k 轮对每条活跃序列重发全对话（~2–3k token），
  其中仅 ~200–500 token 是新增反馈——名义 prefill 量 ≈ 会话数 × 轮数 × 对话长，
  前缀重叠率 72% `[实测]` 意味着 prefix caching 的靶心正在这里。
- **但收益受两层约束**：① **KV 容量 vs 并发会话前缀总量**——89 会话 × ~2.5k ≈ 222k
  token > 101k KV（util0.7），轮间前缀被逐出，命中率打折 `[实测容量+推导]`；
  ② **小批量 decode-bound**——n=8 时 decode 占 wall 主导，prefix ON/OFF 仅差 ~1%
  `[实测 A0]`；89 题（并发 ≤89）矩阵定论：ON/OFF 差 1.2%，仍在噪声内
  `[实测 A1 §1.4]`——本负载规模下 prefix 收益被 decode 主导稀释，
  放大到数百并发会话时 prefill 占比上升，收益预期回升 [估算]。
- **推论（放大设计的第一性依据）**：规模化的多轮 rollout 必须把
  "并发会话数 × 会话长 ≤ KV 预算"当作容量规划约束（§4），或用 radix/prefix 感知的
  调度把同前缀会话聚簇。

### 1.4 89 题头条矩阵（交付口径负载：89 题 × G1 × temp0 贪心）`[实测]`

产物：`logs/poc_vllm/a1_config{0..5}.json` + `compare.json`（不变性核对）。
头条口径 = generate_wall_total（纯 `.generate()` 累计，扣除沙盒与代理开销）。

| config | generate_wall | vs 基线 | engine_load | mem_peak MiB | avg_pass | 不变性(预注册) |
|---|---|---|---|---|---|---|
| 0 现状基线（默认prefix+eager+util0.7） | 90.28s | — | 49.46s | 23,676 | **0.7516**（与历史评测逐位一致） | 参照 |
| 1 prefix 显式 ON | 101.58s | +12.5% | 18.14s | 23,676 | 0.7516 | PASS（0 翻转，norm 失配 1） |
| 2 prefix 显式 OFF | 102.85s | +13.9% | 17.46s | 23,978 | 0.7516 | PASS（0 翻转，norm 失配 3） |
| 3 CUDA graph（eager=OFF） | **58.26s** | **−35.5%** | 63.06s（首编译） | 23,596 | 0.7516 | PASS（0 翻转，norm 失配 2） |
| 4 prefixON + CUDA graph | **58.34s** | **−35.4%** | 37.08s（缓存命中） | 23,812 | 0.7516 | PASS（0 翻转，norm 失配 2） |
| 5 prefixON + util0.9 + graph | 58.48s | −35.2% | 36.07s | 30,290 | 0.7516 | PASS（0 翻转，norm 失配 1） |

**判定（预注册成功线："任一配置 generate_wall 降 ≥20% 且不变性"）：达成。**

- **胜出配置 = config4**（prefix 显式 ON + CUDA graph + util0.7）：速度与 3/5 同组，
  engine_load 最短（inductor 编译缓存命中 37s vs 首编 63s），显存峰值正常
  （util0.9 的 +6.5GiB 对本负载无净收益——KV 富余用不上）。
- **聚合吞吐**：eager 250–290 out tok/s → CUDA graph **443–445 out tok/s**；
  prompt 侧 1,429–1,703 → **2,551–2,573 tok/s**。
- **噪声口径（如实）**：config0 vs config1 名义同配置差 +12.5%（config0 的
  engine_load 49.46s 亦异常偏高，疑似该进程首跑的编译/文件系统冷启动）——
  跨 run 噪声 ~10% 量级；CUDA graph 三臂 −35% 显著超噪声，prefix ON/OFF
  差 1.2% 在噪声内（结论=小并发贪心负载 decode-bound，prefix 净收益微小，
  与 A0 n=8 冒烟一致；prefix 的收益靶心在大并发/长前缀场景，见 §1.3）。
- **不变性细节**：0/89 题 pass 翻转；answer_code（norm_ws 后）失配 1–3 题，
  属贪心跨 batch 形状 fp 归约顺序的平局漂移（预注册预期内），不影响判分。
- config0 avg_pass=0.7516 与历史交付评测逐位一致 = **复现保真锚点**。

---

## 2. verl 迁移映射表（"为什么不用"→"随时可迁"）

**当下不用的原因是环境约束而非能力取舍**：verl 官方镜像在 V100（sm_70）无可用
kernel（torch 2.11+cu130 预编译不含 sm_70；镜像内无 verl 包）——预研实测记录在
`M0_README.md`《verl 镜像预研结论》。自研的收益是训练层逐行掌控：三次行为坍缩的
定位与修复（过程奖励门控 / 优势估计对照 / 熔断）都依赖这层掌控。

**组件级映射（概念级；API 名以 verl 当期官方文档为准——本文档离线写作，
未经在线核验，此为诚实声明）**：

| 自研组件（本仓库） | verl 对应物 | 迁移动作 | 工作量[估算] |
|---|---|---|---|
| 锁步多轮 rollout（`c4_rollout.py`/`m3_rollout.py`） | Agentic RL 的 agent loop / 异步 rollout + tool 调度 | 环境包成 tool/interaction 插件（接口已是"messages 进、反馈文本出"，天然对齐） | 2–3 人日 |
| 组内归一优势（`m3_env.grpo_advantages`） | core_algos 的 GRPO outcome advantage（adv_estimator=grpo） | 语义相同，零迁移 | ~0 |
| disable_adapter 零显存参考 KL | LoRA 模式下 ref = 关 adapter 的 base（同思路）；全参模式独立 ref worker | 配置项 | 0.5 人日 |
| 零优势过滤 | DAPO recipe 的 filter_groups（dynamic sampling） | 配置项 | 0.5 人日 |
| fmt 熔断 + 逐轮落盘 | 自定义 metric + trainer 回调/钩子 | 小代码 | 1–2 人日 |
| 无容器沙盒（unshare+rlimit，`c4_env.py`） | tool 函数（本地子进程执行） | 直接注册 | 1 人日 |
| 查重门/隐藏判分（grade_solution） | RewardManager 自定义 reward fn | 直接注册 | 1 人日 |
| turn 级优势（`c4_gigpo.py`，GiGPO 式） | verl-agent（AMAP-ML）的 GiGPO 实现，或自定义 adv estimator | 对照/引用其实现，锚点定义按环境适配 | 2–5 人日 |
| **合计** | | 熟悉两栈的工程师 | **≈1.5–2 周迁移+回归[估算]** |

**直接带走、与框架无关的资产**：环境插件（检索/代码两个）、奖励门控纪律
（过程分必须被合法终止门控）、贪心/采样双口径选型纪律、污染筛选与防 hacking
三道门、坍缩诊断方法论——这些是方法资产，不是代码依赖。

---

## 3. 规模外推模型（全部 [估算]，假设显式）

### 3.1 假设清单（每个外推数字的口径）

| 假设 | 取值 | 依据 |
|---|---|---|
| rollout 数据并行线性度 | 0.8–0.9 | 多轮锁步含 CPU 沙盒串行段；[估算]保守取 0.85 |
| 训练侧扩展 | 4B/8B 走 FSDP/ZeRO（≤64 卡近线性）；30B+ MoE 走专家并行；70B dense 走 TP×PP | 业界通行配置 [文献] |
| decode 吞吐换算 | V100→A100 ≈2–2.5×、→H100 ≈4–6×（同精度同批量量级） | 公开 benchmark 量级参照 [文献]，非本机实测 |
| MFU（训练侧） | fp16/bf16 30–45%（LoRA 更低但显存省） | 通行范围 [文献] |
| 多轮 token 经济 | 每 iter token 量 ∝ 题数×G×平均轮数×(对话长+生成)；本仓库实测平均 1.44–1.82 runs、对话 ~2.5k token | [实测] logs/ 评测与训练落盘 |
| rollout:train 时间占比 | **实测 79:21**（A2：iter 全环 19.6min = rollout 相 926.6s + 训练 249.8s） | [实测] A2 |

### 3.2 三档放大路径

**档 1：8B（单机 8×V100/8×A100 可达）**
- 训练：LoRA r32 + 梯度检查点（本栈已用）；V100 32G 单卡 8B fp16 权重 16G +
  LoRA/优化器/激活 → 紧张但可行（ZeRO-2 或 2 卡模型分片）[估算]。
- rollout：vLLM tp=1×8 实例数据并行或 tp=2×4；KV 容量约束（§1.3）随对话长同比收紧。
- 预期 iter wall：以 4B 实测 16–17 min 为基，8B 生成端 ~×1.8、训练端 ~×2 →
  **≈30–35 min/iter（8×V100 数据并行 rollout 后可能回到 ~20 min）[估算]**。

**档 2：30B-A3B MoE（内部训练平台 MI308X 训练侧已走通，RL 侧设计级）**
- 训练侧一手经验（[实测]）：LLaMA-Factory + MI308X（ROCm/bf16），
  LoRA 命中全部 128 expert、adapter ≈1.7GB、OOM 回退预案备妥。
- RL 侧 [估算+设计]：rollout 用 vLLM MoE 支持（MI308X/H100；**V100 上 vLLM MoE
  全线失败是本项目实测过的硬边界**，issue #17392 佐证）；A3B 激活 3B →
  生成端成本接近 3–4B dense、显存按 30B 权重算（bf16 ~60GB → 2–4 卡分片/实例）。
- 意义：踩到"百亿参数级"门槛；口径纪律：**LoRA 复现非全参，主动说清**。

**档 3：70B+ dense / 更大 MoE（H100 集群，设计级）**
- 训练：Megatron TP4×PP2 起（70B bf16 权重 140GB → ≥8 卡承载 + 优化器分片）；
  64 卡量级一个完整 GRPO iter（千条轨迹级）【估算：以 §3.1 假设推，给出区间而非点值，
  回填时附推导式】。
- rollout：disaggregated（§4），vLLM/SGLang tp4×N 实例。
- 本仓库不假装做过：这一档的全部内容是**设计+外推**，证据类型如实标注。

### 3.3 为什么放大后 rollout 仍是第一瓶颈（多轮特有）

单轮 RLHF 的常识是"生成占大头"；多轮 agentic RL 再加三条本仓库实测/推导过的机制：
① 锁步空转——长尾轨迹（MAX_TURNS 6 vs 平均 1.44 runs）拖住整批 `[实测轮数分布]`；
② 前缀重 prefill 与 KV 逐出（§1.3）；③ **交替独占的串行结构**——A2 分相拆分
`[实测，跨 run 差分口径]`：iter 全环 19.6min ≈ 引擎构建+释放 ~20–40s（**换载税
本体仅 ~2–3%，不是大头**；重建=A0/A1 engine_load 实测，释放=秒级估算）+
G4 rollout（generate+沙盒+判分）~730–760s（**62–65%，真正的第一瓶颈**）+
优势计算/save/merge ~165s（~14%，R0 replay 基线 413.7−249.1）+ 训练 ~250s（21%）。
真正的架构成本不是换载本身，而是 rollout 相与训练相**串行无法重叠**；
规模化的答案不是更大的卡，而是**架构换形**：§4（disaggregated 让两相分卡并行，
顺带消灭换载税）。

---

## 4. 千卡预案（设计级——如实标注：非实操经历）

**参照系**：SkyRL（Mercor，训 397B 知识工作 agent）[文献]；verl HybridEngine /
异步 rollout [文献]；Tongyi DeepResearch-30B 的 agentic 训练管线 [文献]。

```
                    ┌────────────── controller ──────────────┐
                    │ 任务池流式供给 / staleness 控制 / 熔断监控 │
                    └──────┬───────────────────┬─────────────┘
             ┌─────────────┴─────┐   ┌─────────┴──────────┐
             │ rollout farm       │   │ trainer            │
             │ N×(vLLM/SGLang     │   │ Megatron/FSDP      │
             │  实例, tp 按模型档) │   │ 权重版本广播/resharding│
             │ + 环境 worker 池    │   │ GRPO/turn级优势     │
             │  (CPU 沙盒隔离,     │   │ KL ref worker      │
             │   横向扩容)         │   └─────────┬──────────┘
             └─────────┬──────────┘             │
                       │  轨迹流（partial rollout 可中断续跑）
                       └────────► 经验缓冲/判分服务（无状态横扩）
```

设计要点（每条对应本仓库实测过的问题）：
1. **rollout/训练分卡（disaggregated）**：消灭换载税（§3.3③）；权重同步用
   resharding/广播，同步频率=每 iter 一次（on-policy 纪律）。
2. **异步 + partial rollout**：长尾轨迹截断续跑（§3.3①），staleness 上界控制
   （off-policy 偏移用重要性截断或直接丢弃超界轨迹——与 DAPO clip-higher 同族思路）。
3. **KV 容量规划**：并发会话数 × 会话长 ≤ 单实例 KV 预算（§1.3 的教训放大版）；
   prefix/radix 感知调度把同任务组的会话聚簇到同实例。
4. **容错**：trainer checkpoint 每 iter；rollout 节点失败=任务重入队（环境无状态）；
   判分服务（沙盒/隐藏测试）无状态横扩，配额隔离防单题烧机（本仓库 bigint 爆炸
   三连死教训的规模化版本，`C4_README.md` §坑记录）。
5. **数据与防作弊在规模上的延伸**：污染筛选（76.5% 污染率方法论 `[实测]`）、
   查重门、隐藏测试永不进 prompt——题池扩到万级时这三道门必须先于训练自动化。
6. **监控**：贪心/采样双口径背离（两个环境独立复现的固有性质 `[实测]`）、
   fmt 熔断、reward hacking 定期抽查（difflib 相似度分布）。
7. **成本量级**[估算]：千卡 H100 × 多轮 agentic RL 的卡时成本以
   "轨迹数×token 经济"推：回填时给推导式与区间。

---

## 5. 推理引擎深度定制路线（对应"深度定制优化 vLLM/SGLang"）

### 5.1 本仓库已实测的定制点（证据 = `logs/poc_vllm/`）

| 定制点 | 结果 | 证据 |
|---|---|---|
| CUDA graph（enforce_eager=False）在 sm_70 | **可用且是本轮最大赢家**：n=8 冒烟 2.15×（55.4→120.5 out tok/s）；89 题头条 **−35.5% generate_wall（90.3→58.3s）**；代价=首编译加载 63s（inductor 缓存命中后 37s） | [实测] A0+A1 §1.4 |
| prefix caching 默认态审计 | V1 引擎默认 ON——"没配置"≠"没开"；显式 OFF 可用（对照臂成立） | [实测] A0 有效配置打印件 + A1 config2 |
| prefix caching 真实约束 | 命中受 KV 容量 vs 并发前缀总量约束（101k vs ~222k token）；小并发负载净收益 ~1%（n=8 与 89 题两处一致） | [实测+推导] A0+A1 |
| gpu_memory_utilization 上限 | 0.9 无 OOM（A0 峰值 29.5 GiB / A1 30.3 GiB），KV 101k→124k tokens；但小并发负载下无速度净收益 | [实测] A0 util 阶梯 + A1 config5 |
| 输出不变性纪律 | 贪心跨配置不保证 bit 级一致（batch 形状→归约顺序→平局 token）；A1 实测 5 配置全部 **0/89 翻转**、answer_norm 失配 1–3 题（预期内漂移） | [预注册+实测] `poc_vllm_bench.py` + `compare.json` |
| EngineCore fork 死锁排查 | 训练循环内建引擎 4/4 挂死于 V2 Model Runner 后握手；E1-E12 全新容器二分定位为模块组合触发（单成分均不触发）；根因=fork 起子进程 + 父进程线程态（fork-while-threaded 死锁族）；**修复 `VLLM_WORKER_MULTIPROC_METHOD=spawn`** 并在真实路径验证（引擎 init 24.19s） | [实测] `ENGINE_HANG_RCA.md` |
| rollout-训练共置量化 | 交替独占实测分相：rollout 相（引擎+采样+沙盒+换载+merge）**78.8%** vs 训练 21.2%（eager 全环 19.6min；graph 臂 18.7min，−4.7%）；换载税本体仅 ~2–3%，大头是 rollout 串行（§3.3 拆分式） | [实测] A2 `checkpoints/c4_a2_probe/*/stats.jsonl` + R0 replay 差分 |

**"深度定制"的能力证据链**（工程口径）：不是"调过参数"，而是
① 审计引擎有效配置发现默认态与假设不符；② 用 analytic LCP 与引擎日志双口径
互证命中率；③ 找到 KV 容量 vs 并发会话的第一性约束；④ 预注册不变性判据防
"提速但输出变了"的假优化。

### 5.2 量化与更大硬件的边界（诚实分层）

- sm_70（V100）：fp16 为本仓库实测基线；weight-only INT4（AWQ/GPTQ）在该架构
  的 kernel 支持随 vLLM 版本收窄，1.3.0 定制镜像维持 sm_70 属自建维护 [实测(镜像)+文献]。
- FP8 需 sm_89+；A100/H100 上 INT4/FP8 量化 rollout 是标准降本手段 [文献]，
  外推进 §3 模型（rollout 成本 ×0.5–0.7 量级[估算]）。
- SGLang 对照：RadixAttention 与本负载（同前缀多轮会话）天然匹配；本栈 rollout
  接口抽象（messages 进/反馈出）换引擎层不动训练层 [设计]。

---

## 6. 设计级 Q&A 与工程要求映射

### 6.1 "把这套栈放大到千卡，你怎么办？"（预判必杀问）

**30 秒版**：先说瓶颈在哪（我有实测）：多轮 agentic RL 的规模瓶颈不在训练侧，
在 rollout——锁步长尾空转、前缀重 prefill、KV 容量顶不住并发会话、交替独占的换载税。
所以放大 = 架构换形：rollout/训练分卡的 disaggregated 布局、异步 partial rollout
治长尾、prefix 感知调度 + KV 容量规划、判分沙盒无状态横扩；训练侧按模型档走
FSDP→Megatron TP/PP；监控沿用我两个环境验证过的双口径背离与熔断纪律。

**3 分钟版**：按 §4 七个要点展开，每个要点都指回本仓库的实测出处
（换载税 A2、KV 约束 A0、轮数长尾分布 logs、污染筛选方法论 M1）。
如实边界：千卡本身是设计级，我实操过的上限是 8×V100 单机 + 内部训练平台
MI308X 的 30B-A3B MoE LoRA。

### 6.2 "为什么不直接 verl？"

三层：① 环境约束——sm_70 无预编译 kernel（预研实测记录）；② 迁移映射已备好
（§2，≈1.5–2 周[估算]），随时可迁，"不用"不是"不会"；③ 自研的真实收益——
三次行为坍缩能定位到"过程奖励未门控/优势估计器选择/格式熔断"这一层，靠的是
训练层逐行掌控；用黑盒框架时这些会以"训练不 work"的形式被埋掉。

### 6.3 工程要求映射表

| 工程要求（2026-09 业界调研） | 本文证据 | 证据类型 |
|---|---|---|
| 百亿参数级模型后训练经验 | §3.2 档 2/档 3 + 30B-A3B MoE LoRA（内部训练平台） | 30B 训练侧=实操；RL 侧=设计+估算 |
| Megatron/DeepSpeed 多机多卡 | §3.1/§4（FSDP→TP×PP 配置、disaggregated 预案） | 设计级（如实标注，不冒充实操） |
| RLVR / Long-CoT RL | 本仓库全部（两环境可验证奖励多轮 GRPO + 消融 + 坍缩诊断） | 实测 |
| vLLM/SGLang 深度定制 | §5（CUDA graph −35.5% 头条+0 翻转、prefix/KV 约束、不变性纪律、共置量化） | 实测（本仓库） |
| turn/tree 级信用分配前沿 | `c4_gigpo.py` + Stage 0 反事实 + Stage 1.5 配对对照（Δ=0.0000）+ Stage 2 主判据 PASS（峰值 0.7561 vs 0.7516，贪心修复率 0→11.8%/6.7%） | 实测（单 run，幅度≈噪声量级，如实标注） |

### 6.4 "你的 turn 级优势和 GiGPO/Tree-GRPO 什么关系？"

GiGPO（NeurIPS25, arXiv 2505.10978）的二级分组思想落地到修码环境：锚点 =
(task, 可见判定 P/F/E 全前缀)；与原文差异及原因——γ=1.0（Stage 0 实测 γ<1 会在
同分组引入长度效率信号，污染单机制归因，修订记录在 `c4_train_gigpo.py` docstring）；
Tree-GRPO 需要 rollout 引擎支持状态分叉，锁步引擎改造上一个量级，故本轮选
GiGPO 式（改动面=优势函数+mask 轮边界+loss 段）。

**Stage 1.5 配对对照实测（`logs/poc_gigpo/stage15_compare.json`）**：iter0 rollouts
上 R0(epi)/R1(gigpo) 同 seed=42 同 192 子集重放训练（γ=1.0 下零优势过滤集合恒等，
配对性由构造保证，单测 T6 实证 140=140）→ 贪心 89 题评测：
- **avg_pass 0.7401 = 0.7401，Δ=0.0000**；题级 0 翻转；Phase1 修复指标族
  （saw_fail 17 / repair_success 0.0588 / visible_fixed 0.2353）与 Phase2 首跑
  隐藏判分（first_gated 0.7477 / F 分支 Δ+0.019 / P 分支 Δ−0.0139）**逐位相同**。
- replay 保真度：R0 0.7401 vs 历史 iter1 0.7516 = −0.0115（差异来源=shuffle 子集
  不可复原 + fp16 非确定性；如实报告，不作复现主张）。
- R1 训练健康度：turn 加权路径开销 +0.9%（251.3s vs 249.1s）；mean|A_tok| 1.372，
  量纲比值 ~1.2 < 1.5 预警线；926 turns 仅 1 个无 span。
- **预注册判定**：主判据（Δ≥+0.005）FAIL；备选（Δ≥−0.005 且修复指标任一严格改善）
  FAIL（全平）；方向门（Δ≥0 进 Stage 2）PASS → **GATE-PASS-NO-SIGNAL**：
  单 iter 弱更新（lr 5e-6 × 192 样本）不足以产生任何贪心行为分化，无害性确认、
  机制起效零宣称。Stage 2（全量 4 iters）为真正检验，结果见下。

**Stage 2 全量对照实测（2026-09-07，`logs/poc_gigpo/stage2_analyze.json` + 
`stage2_eval_iter{0..3}.json`；gigpo 臂 rollout 吃 WS-1 胜出配置，评测两侧同 CLI
同引擎贪心）**：

| 指标 | epi 臂（历史 c4_train） | gigpo 臂（本轮） |
|---|---|---|
| 贪心 avg_pass iter0→3 | 0.7305 / **0.7516** / 0.7436 / 0.7435 | 0.7182 / **0.7561** / **0.7555** / 0.7200 |
| 贪心修复成功率（saw_fail 全修复） | 四 iter 全 **0** | **iter0 2/17=11.8%、iter1 1/15=6.7%** |
| 题级（vs 同 iter epi） | — | iter1：0 掉分+1 新满分；iter2：0 掉分+1 新满分 |
| Phase2 F 分支隐藏修复 Δ | iter0 +0.002 / iter1 +0.0022 | **iter0 +0.0688（34×）** / iter1 −0.0067 |
| 首跑隐藏质量（first_gated，iter1） | 0.7512 | 0.7572 |
| dup 背题（贪心评测） | 6 稳态 | 6 稳态；iter0=8、**iter3=9（3 题新 dup 全判 0）** |

- **预注册判定（机械）**：主判据"任一 iter ≥0.7526"**PASS**（iter1 0.7561、
  iter2 0.7555 双点过线；epi 峰值 0.7516 为单点）；备选"≥0.7416 且修复率>0%
  且 F 分支 Δ>0.17"FAIL（iter1 修复率 6.7%>0 但 F Δ=−0.0067）。
  整体 **PASS-main**——"或"结构下主判据单独充分。
- **机理画像（方向一致的协同证据）**：① 修复通道从无到有（epi 四 iter 贪心修复
  全 0 → gigpo iter0/iter1 >0；iter0 隐藏口径 F Δ=epi 的 34×）——与 Stage 0
  "mixed 组内信用重分配"预测一致；② 峰值更高且平台更宽（双 iter 过线 vs 单点）；
  ③ 边界题更稳（c4_he_order_by_points：epi iter2/3=0.073，gigpo iter2/3 满分，
  5-run 深度修复正是 turn 级信用的靶心场景）；④ iter1 首跑隐藏质量本身更高
  （0.7572 vs 0.7512）——收益不完全来自修复轮，首版质量也在升。
- **诚实边界（如实）**：① 单 run vs 单 run，+0.0045 峰值差在重跑噪声量级
  （Stage 1.5 replay 保真度差 −0.0115 可参照）——协同证据（双 iter 过线 +
  修复率 0→>0 + F Δ 34×）方向一致，但**不宣称大效应，只宣称过线与方向一致
  的初步证据**；② gigpo 训练 rollout 用 graph+prefixON 引擎（temp1.0 轨迹=
  不同 RNG 实现，属重跑级差异，贪心评测已证跨引擎 0/89 翻转）；③ iter3 回落
  0.7200 的 3 道掉分题**全部是 dup_ref=True（背题被查重门判 0）**，非修复能力
  退化（visible_fixed 保持 0.1333）——turn 级信用锐化与背题倾向的关系单 run
  不可归因，记为 v2 观察项；④ 部署选点仍按贪心口径 iter1（纪律与 epi 线相同）。

---

## 附：产物与复现

- 实测产物：`logs/poc_vllm/`（A0/A1 全矩阵 + compare + 不变性核对）、
  `logs/poc_gigpo/stage0_report.json`（反事实）、`checkpoints/c4_replay/`（配对 replay）、
  `checkpoints/c4_a2_probe/`（共置分相）、`checkpoints/c4_gigpo/`（Stage 2 训练
  stats）、`logs/poc_gigpo/stage2_eval_iter{0..3}.json` + `stage2_analyze.json`
  （Stage 2 贪心评测与配对分析）、`poc_gigpo_test.py`（31 项单测）、
  `poc_gigpo_stage2_analyze.py`（Stage 2 分析驱动，预注册判据机械判定）。
- 复现命令：`poc_vllm_bench.py` / `c4_train_gigpo.py` docstring 内含 docker 全命令。
- 文献坐标：GiGPO 2505.10978 · Tree-GRPO 2509.21240 · DAPO 2503.14476 ·
  μCode 2502.20380 · MURPHY 2511.07833 · SkyRL 2511.16108 · verl Agentic RL 文档。
