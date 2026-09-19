# C4：可验证奖励 RL 训练多轮代码修复 Agent —— 实验日志

> 奖励设计与任务构造的完整设计口径见本文 §四。
> 定位：检索 Agent 项目（m3 检索环境）的姊妹篇——同一套自研锁步多轮训练栈，
> 工具调用 `<search>` → `<run_code>`，证明"环境可插拔、方法可迁移"。
> **红线（同检索 Agent 项目）：所有数字真实跑出才记录；未跑出的不写。**

## 一、文件清单（c4_* 与 m3_* 一一对应）

| 文件 | 对应 m3 | 职责 |
|---|---|---|
| `c4_env.py` | `m3_env.py` | 沙盒执行（unshare -rn 禁网/RLIMIT 限时限内存禁写/输出截断/最小 env）、`CodeExecEnv.run_visible`（轮内工具调用）、`grade_solution`（隐藏测试逐 case 稠密判分）、`parse_trajectory_code`、`compute_reward_code`（终止门控） |
| `c4_tasks.py` | —（新） | MBPP+ → AST 单点变异注入 bug → 验证（变异体可见测试必须失败、plus 率<0.95）→ `tasks.jsonl`（模型可见）/ `hidden.jsonl`（判分私有）分离落盘 |
| `c4_test_env.py` | —（新） | CPU 单测 24 项：沙盒防御层 6 + run_visible 3 + grading 5 + parse/reward 6 + 任务抽查/防泄漏审计 5 |
| `c4_rollout.py` | `m3_rollout.py` | vLLM 锁步多轮 rollout（stop=`</run_code>`/`</answer>`，MAX_TURNS=6，MAX_RUNS=5，max_new_tokens=768）+ grading + 查重门 + 轨迹落盘 |
| `c4_baseline.py` | `m1_baseline.py` | M1 基线①：基座直答（one-shot 贪心，无执行反馈）+ 背题筛选（直答满分/逐字 dup → 剔除出训练集，产出 tasks_train/hidden_train/excluded） |
| `c4_rft.py` | —（新） | M1 基线②：RFT 采样（train 集 G 条/题 temp=1.0）→ 正例过滤（pass≥阈值 & 格式合法 & 非 dup）→ SFT 数据落盘（训练走 m2_sft_train.py） |
| `data/c4/tasks.jsonl` / `hidden.jsonl` | — | 任务集（M0：60 题 MBPP；M1 起：双源扩池 MBPP+ ∪ HumanEval+，见 §三-B） |

## 二、运行命令（docker，坑见 §五）

```bash
# CPU 单测 + 任务生成（无 GPU；禁网用例必须 seccomp=unconfined）
docker run --rm --security-opt seccomp=unconfined \
  -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
  1cat-vllm:v100-1.3.0 bash -c "python c4_test_env.py && python c4_tasks.py --n 60 --seed 42 --out data/c4"

# GPU rollout（M0 冒烟：单卡 8 题）
docker run --rm --gpus '"device=0"' --security-opt seccomp=unconfined \
  -v /mnt/nas3/shared/model:/models:ro \
  -v /mnt/storage/<user>/claudecode_projects/agentic_rl:/work -w /work \
  -e HF_HUB_OFFLINE=1 \
  1cat-vllm:v100-1.3.0 python c4_rollout.py --model /models/Qwen3-4B-Instruct-2507 --n 8 --G 1 --show 0
```

## 三、M0 环境原型（2026-09-05）——✅ 通过止损门

### 3.1 CPU 单测：24/24 PASS

沙盒防御层全验证：禁网（unshare -rn 下外连 1.1.1.1:53 失败）、禁写
（RLIMIT_FSIZE=0 → flush 抛 EFBIG）、死循环超时被杀、宿主 env 不泄漏、
输出截断 ≤1200 字符；grading 稠密分正确（部分正确解 = 1/3 而非 0/1）、
harness 缺失退化为原始 assert、死循环/语法错误解安全判 0；奖励终止门控
（无 `<answer>` → r_pass=0，检索 Agent 项目 run1 教训移植）验证通过。

### 3.2 任务集生成（seed=42）

`{"seen": 81, "built": 60, "no_valid_mutant": 21, "mut_reject_visible_pass": 15,
"mut_reject_plus_rate": 1, "plus_ok": 59}` —— 81 题筛出 60 题合格任务，
59/60 带 EvalPlus plus harness（逐 case 稠密奖励可用）。
防泄漏审计：tasks.jsonl 不含参考解/harness 文本（参考解只存 hidden.jsonl）。

### 3.3 首次真实多轮 rollout（Qwen3-4B-Instruct-2507，8 题 G=1，temperature=1.0）

```
submit_rate=0.875  avg_pass_rate=0.2644  avg_reward=0.4394
format_ok=87.5%    avg_n_runs=1.12       grade_timeout=0  dup_ref=4
```

- **第一条完整轨迹（c4_751，min-heap 检查）**：读坏代码 → 1 次 `<run_code>`
  重写修复（可见测试 PASSED）→ `<answer>` 提交 → 隐藏测试 `pass_rate=1.0`
  → `reward=1.2`（0.2 格式分满格）。多轮修码闭环真实成立。
- c4_102：4 次 run 迭代利用执行反馈，修到 plus 率 0.115（多轮行为出现）。
- c4_630：未提交（终止门控判 r_pass=0，符合设计）。

### 3.4 ⚠️ 当场抓到 benchmark contamination（4/8 = dup_ref）

c4_743/284/139/77 四题，模型提交的代码与 MBPP 参考解**归一化后逐字符相同**
（如 `def rotate_right(l, m): return l[-m:] + l[:-m]`）——Qwen3 预训练背过
MBPP 公开题。这正是 Georgia Tech 2026 实证的"本地泄题"攻击面在本环境的
真实复现；判 0 门全部正确拦截（4 条 pass_rate 均置 0）。

**含义与对策**：
1. 拿到第一手素材：防御先于训练生效，抓到即记录（本文件即证据）。
2. 判 0 是设计决策（背答案不能证明修复能力），但存在军备竞赛风险：模型可能
   学会"改写空格/变量名绕过逐字查重"。M1 对策：
   - 基座直答筛选：base 模型单次生成即通过隐藏测试的题 → 从训练集剔除
     （无训练信号且有污染风险），只保留"必须修"的题；
   - 查重升级为 AST 规范化对比（重命名不变式，防改名绕过）——M1 视 dup 率决定；
   - 每次评测统计 dup_ref 率入日志，监控绕过行为演化。
3. M1 基线数字须在"筛选后任务集"上跑，并同时报告剔除比例（污染率本身就是
   可报告的实证数字）。

### 3.5 M0 结论

止损门（1 周内跑通单条多轮轨迹）**提前通过**：环境层（沙盒+判分+奖励）、
任务层（注入 bug+防泄漏）、rollout 层（锁步多轮）全部真实验证。
下一步 M1：基座单次生成基线 + RFT 基线（含背题筛选）。

## 三-B、M1 基线（2026-09-05 进行中）

### B.1 基座直答基线（one-shot 贪心，无执行反馈）+ 背题筛选 —— 60 题 MBPP 池

```
submit_rate=0.9333  avg_pass_rate=0.5211
solved_full=26      dup_ref_verbatim=16   → exclusion_rate=0.70
partial=8  zero=10  train_keep=18/60
```

**重大发现：Qwen3-4B-Instruct-2507 对 MBPP 的污染率 = 70%**（直答满分 26 +
逐字背参考解 16；贪心解码下 16 题输出与 ref_code 归一化后逐字符相同——污染实锤）。
判 0 门 + 筛选流程全部按设计工作。这个数字本身是可报告的一手实证
（benchmark contamination 定量测量）。

**决策：扩池**——18 题撑不起 GRPO。任务池扩为 MBPP+ 全量 378 ∪ HumanEval+ 164
（EvalPlus 双源，`c4_tasks.py` 已重写为统一 schema 适配层）：
- HumanEval+ 差异：`canonical_solution` 是函数体（签名在 prompt 里）→ 参考解=
  prompt+canonical_solution 拼接；`entry_point` 直接给函数名；`check(candidate)`
  体内含 inputs/results → ast 提取重建为 MBPP 同构顶层 harness_prefix；
  无 test_list → 可见测试从 check 案例拼装（优先非浮点 case）。
- load_pool 实测：MBPP+ 378 + HumanEval+ 157（7 题 harness 重建失败被跳过）。

### B.1b 全池直答基线终版（378 题 = MBPP+ 257 ∪ HumanEval+ 121，贪心）

```
submit_rate=0.9418  avg_pass_rate=0.6193
solved_full=213     dup_ref_verbatim=76  → exclusion_rate=0.7646
partial=41  zero=48 → 训练集 89 题（mbpp+ 66 / humaneval+ 23）
按源污染率：mbpp+ 74.3%（逐字背 64/257=24.9%）｜ humaneval+ 81.0%（直答满分 71%）
训练子集直答对照值：avg_pass=0.2370（89 题均非满分题，48 题直答 0 分）
```

**结论**：Qwen3-4B-Instruct-2507 对两大经典 benchmark 的污染率均 ~3/4，
HumanEval+ 反而更高（教科书式解法背得更熟，但逐字 dup 率低——长代码风格方差大；
MBPP 短函数逐字背出率 24.9%）。89 题训练集可支撑 M2 小规模 GRPO 起步
（每 iter 全量采样），后续扩池方向：LiveCodeBench（训练截止后新题）、
多点组合变异（让背过的题重新"不可背"）。

途中修复两事故（详见坑记录 9/10）：collatz 变异体 bigint 爆炸卡死构建
（guarded 子进程 rlimit 修复）；HE 不同题共用函数名致 tid 撞车、判分用错
参考解（tid 带题号 + 8 题补跑重算，修复后 89+289=378 数字自洽）。

### B.2 grading 截断 bug（扩池时抓到，已修）

HumanEval+ 单题 plus case 上千（task 0 = 1006 case），初版 GRADE_JSON 输出
0/1 数组 → stdout 远超 MAX_OUT_CHARS=1200 截断 → JSON 解析失败 → 参考解被
误判 0 分。修复：grading 输出改**聚合计数**（orig_pass/orig_n/plus_pass/plus_n，
恒定长度）。修复后 HE 参考解 grading = 1006/1006 满分，单测回归 25/25 PASS。
教训：沙盒输出截断与结构化结果通道必须分离（结果通道恒短，内容通道才截断）。

### B.1c 基座多轮贪心基线（同 89 题训练子集，RLTF 口径）——执行反馈的价值

```
submit_rate=0.7753  avg_pass_rate=0.4648  avg_reward=0.6176
format_ok=77.53%    avg_n_runs=1.27       dup_ref=12  grade_timeout=3
```

**核心对照（同题集、同贪心解码）**：

| 口径 | avg_pass_rate | 增量 |
|---|---|---|
| 直答（无执行反馈） | 0.2370 | — |
| 多轮（带 run_code 沙盒反馈） | **0.4648** | **+22.8pt（相对提升 96%）** |

执行反馈让基座在同题集上通过率翻倍——多轮修码训练（M2 GRPO）的起点与
存在性依据。观察点：① dup_ref=12/89，多轮下模型仍会收敛到背诵参考解，
判 0 门拦截中（M2 军备竞赛监控点：警惕"改写空格绕过查重"策略演化）；
② grade_timeout=3，失控提交被沙盒 RLIMIT/超时正确判 0；
③ 贪心下 22.5% 轨迹未产生合法 <answer>（终止门控判 0），格式行为有训练空间。

### B.2b RFT 基线（M1 基线②）

**采样**（89 题 × G=8 = 712 条，temp=1.0 多轮）：
```
submit_rate=0.7247  n_pos_raw=220 (30.9%)  n_pos_kept=93（每题≤2，覆盖 53/89 题）
dup_ref_in_pool=72 (10.1%)  avg_n_runs_pos=1.03
```
- **温度采样下每 10 条轨迹就有 1 条逐字背参考解**，全部被过滤排除在 SFT
  数据外——污染防线在训练数据侧生效（背题轨迹不进 SFT，防污染扩散）。
- **正例几乎全是一轮修对（1.03）**：RFT 只能教"一次写对"，教不会"利用
  反馈迭代修复"——M2 多轮 GRPO 的差异化空间，也是"为什么 RFT 不够"的
  一手证据。

**训练**：93 条 messages（len p50=622/p95=1617/max=2673），LoRA r=32
（trainable 1.6%），MAX_LEN 4096，BS2×ACC4，5 epochs，lr 1e-4 cosine，
loss 0.102→0.017，runtime 397s（单卡 V100）→ merge 到 `checkpoints/c4_rft_merged`。

**评测①直答（同 89 题贪心）**：
```
submit_rate=1.0000（基座 0.9418→满分提交）  avg_pass_rate=0.6952
solved_full=51/89   dup_ref_verbatim=4      partial=25  zero=9
```
**对照基座直答 0.2370 → RFT 直答 0.6952（+45.8pt，2.9 倍）**——这 89 题全是
基座直答不满分的题，RFT 后 51 题直答满分。RFT 冷启动在小数据（93 条）下
对"修复能力+提交格式"双重生效；新增 dup 仅 4 条（SFT 数据已滤背题轨迹，
模型自发逐字背诵是预训练记忆残留，量级可控）。

**评测②多轮（同 89 题贪心）**：
```
submit_rate=0.8876  avg_pass_rate=0.7295  avg_reward=0.8958
format_ok=88.76%    avg_n_runs=1.76       dup_ref=8  grade_timeout=2
```

### B.2c M1 三组数字总表（89 题训练子集，贪心，全部真实跑出）

| 口径 | 基座 Instruct-2507 | RFT（93 条正例 SFT） | RFT 增量 |
|---|---|---|---|
| 直答（无执行反馈） | 0.2370 | 0.6952 | **+45.8pt（2.9×）** |
| 多轮（run_code 沙盒反馈） | 0.4648 | **0.7295** | **+26.5pt** |
| 执行反馈增量（同模型内） | +22.8pt | +3.4pt | — |
| submit_rate（多轮） | 0.7753 | 0.8876 | +11.2pt |
| avg_n_runs（多轮） | 1.27 | 1.76 | 更愿用工具 |

解读：
1. RFT 多轮 0.7295 = **M2 GRPO 要超越的目标线**；
2. RFT 后执行反馈增量收窄（22.8→3.4pt）：RFT 教会"一次写对"，减少对反馈
   的依赖——但也说明**迭代修复行为没有被 RFT 学到**（正例 avg_n_runs=1.03
   的镜像），这是多轮 GRPO 的差异化空间；
3. dup_ref：基座多轮 12 → RFT 多轮 8，SFT 数据滤除背题轨迹后自发背诵略降；
4. 多轮 format_ok（88.76%）< 直答 submit（100%）：多轮轨迹轮次多、invalid
   轮机会多，格式仍有训练空间（GRPO 的 r_format 项会压这个）。

### B.3 M1 进度清单

1. ✅ 全池任务构建（378 题双源；两次事故修复见坑记录 9/10）
2. ✅ 基线①：直答基线（全池）+ 背题筛选 → tasks_train.jsonl（89 题）
3. ✅ 基线①b：基座多轮贪心（B.1c，执行反馈 +22.8pt）
4. ✅ 基线②：RFT（采样 712 → 正例 93 → LoRA SFT → 双口径评测，B.2b/B.2c）
5. ✅ M2 前检查点：三组数字齐（B.2c 总表）——**M1 收官，GRPO 目标线 0.7295**

## 三-C、M2 主实验：多轮 GRPO（2026-09-06）

起点 `c4_rft_merged`（RFT 冷启动），89 题全量 × G=4（356 轨迹/iter），
temp=1.0，lr 5e-6，KL β 0.05，零优势过滤，有效轨迹上限 192，
fmt_pct 腰斩熔断（未触发）。**训练层从 m3_train.py 零改动复用**
（c4_train.py 仅换环境接口与统计语义——"环境可插拔"论点的代码级证据）。

### C.1 训练分布曲线（4 iters，rollout 即时分，temp 1.0）

| iter | avg_reward | avg_pass | fmt/submit % | avg_runs | dup_cnt/356 | grade_timeout | 有效轨迹 |
|---|---|---|---|---|---|---|---|
| 0 | 0.8773 | 0.7065 | 90.2 | 1.70 | 26 | 4 | 140 |
| 1 | 0.8794 | 0.7064 | 91.0 | 1.68 | 24 | 5 | 104 |
| 2 | 0.9115 | 0.7225 | 96.3 | 1.45 | 22 | 7 | 112 |
| 3 | 0.9303 | **0.7354** | **98.3** | 1.48 | 22 | 8 | 100 |

解读：
1. 最大收益是**格式/提交率 +8.1pt**（90.2→98.3）：r_format 辅助项生效，
   多轮 invalid 轮几乎消失；
2. avg_pass +2.9pt（0.7065→0.7354）；轮数收敛 1.70→1.48（**§C.5 深挖修正：
   机制是一次修对率上升→修复需求下降，非"学会迭代修复"**）；
3. **dup 率未恶化**（7.3%→6.2%）：未观察到"改写绕过查重"的军备竞赛迹象；
4. 观察项：grade_timeout 4→8 上升（评测后抽查是边界慢代码还是策略性行为）；
5. 零优势组占比升高（有效轨迹 140→100）：成功率上升后组内信号变稀，
   与检索 Agent 项目经验一致（稠密奖励缓解但未消除）。

### C.2 贪心评测（iter3_merged，89 题多轮贪心，temperature=0）

**GRPO 超过 RFT 目标线：0.7295 → 0.7435（+1.4pt）**；对基座 +27.9pt
（0.4648→0.7435，相对提升 60%）。达到方案 §三"良好"标准
（多轮 RL 对 RFT 正增量 ✓ + 行为曲线"学会修" ✓）。

| 模型 | avg_pass_rate | submit_rate | format_ok % | avg_n_runs | dup_ref | grade_timeout |
|---|---|---|---|---|---|---|
| 基座多轮 | 0.4648 | 77.5 | 77.5 | 1.27 | — | — |
| RFT 多轮 | 0.7295 | 88.8 | 88.8 | 1.76 | 8 | 2 |
| **GRPO iter3** | **0.7435** | **100.0** | **100.0** | 1.44 | 6 | 2 |

解读：
1. **格式/提交率 100%**：多轮贪心下 r_format 辅助项把 RFT 遗留的 11.2%
   不提交/格式坏全部修复——GRPO 相对 RFT 的最大单项收益；
2. avg_n_runs 1.44 介于基座(1.27)与 RFT(1.76)之间（**§C.5 深挖修正：轮数收敛
   = 一次修对率上升的副产品；"跑一次验证、过了原样提交"是 GRPO 学到的策略，
   失败后修复未涌现**）；
3. dup_ref 8→6、grade_timeout 2：均低于/持平 RFT，判 0 门持续拦截。

### C.3 reward hacking 抽查（2026-09-06）

**方法**：iter3 全部 89 条提交代码 vs 参考解，difflib 归一化相似度分布 +
近重复逐条 diff + grade_timeout 轨迹人工审读。

- 逐字 dup（sim≈1.0）6 条——全部被查重门拦截判 0 ✓；
- near-dup（0.85≤sim<1.0）40 条。**这是任务构造的必然后果**：buggy 版
  本是参考解的 AST 单点变异，改对即高度接近原文——非"改写绕查重"。
  抽查 3 条定性：c4_mb_261 差异仅为 generator 外加一层冗余括号
  （风格差，pass 1.0）；c4_he_generate_integers 少写 `min(b+1,10)`
  上界（pass 0.0）；c4_mb_233 模型"顺手修正"了参考解函数名拼写
  `lateralsuface→lateral_surface_area`，harness 调原名 → 0 分
  （过度重构失败模式，同时反证非背题——背题会原样抄对）；
- **grade_timeout 定性**（训练期 4→8 上升的观察项）：贪心评测仅 2 条，
  c4_mb_300（O(n²) bigint 二项式，算法天然慢）、c4_he_generate_integers
  （上界写错 + 大区间 case）——均为普通错误代码，**非策略性慢代码**；
  训练期上升归因于 temp=1.0 采样噪声 + 提交率 90→98% 带来更多判分调用；
- 相似度 <0.6 的独立解法 24 条——模型并非只会还原参考解。

**结论：4 iters 内未观察到 reward hacking 演化迹象**（dup 率未升、
无绕查重改写、无策略性超时）。

### C.4 训练进程行为曲线（iter0→3，89 题多轮贪心，temperature=0）

四个 merged checkpoint 同口径贪心评测，连成训练进程曲线：

| 模型 | avg_pass_rate | submit_rate | format_ok % | avg_n_runs | dup_ref | grade_timeout |
|---|---|---|---|---|---|---|
| 基座多轮 | 0.4648 | 77.5 | 77.5 | 1.27 | — | — |
| RFT 多轮（=GRPO 起点） | 0.7295 | 88.8 | 88.8 | 1.76 | 8 | 2 |
| GRPO iter0 | 0.7305 | 88.8 | 88.8 | 1.82 | 6 | 2 |
| **GRPO iter1（峰值）** | **0.7516** | **100.0** | **100.0** | 1.57 | 6 | 2 |
| GRPO iter2 | 0.7436 | 100.0 | 100.0 | 1.60 | 6 | 2 |
| GRPO iter3 | 0.7435 | 100.0 | 100.0 | 1.44 | 6 | 2 |

**关键发现（如实记录，非单调）：**

1. **贪心峰值在 iter1（0.7516，+2.2pt over RFT），不是 iter3**。iter2/iter3
   回落到 0.7436/0.7435 平台；策略在训练分布(temp 1.0)上仍涨（C.1 avg_pass
   0.7065→0.7354 单调），但**确定性贪心性能 iter1 见顶**——"训练分↑≠部署分↑"
   的现场对照，也是 M2 我先前只报 iter3(0.7435,+1.4pt) 的修正：
   **GRPO 最佳 checkpoint = iter1**。
   **深挖修正（§C.5.3）**：题级 diff 显示 iter1→iter3 回落几乎全部来自**单题翻转**
   （c4_he_order_by_points 1.0→0.073；n=89 单题噪声 ±1.1pt > 0.8pt 差距）——
   正确解读是"iter1 起进入 0.743~0.752 平台"，而非系统性过拟合；
   iter1 仍为点估计最高的部署选型。
2. **格式/提交率一步到位**：iter0 还是 RFT 的 88.8%，**iter1 直接 100% 并保持**
   ——r_format 辅助项的第一个完整 GRPO 更新就把 RFT 遗留的 11.2% 坏格式清零。
3. **轮数单调收敛** 1.82→1.57→1.60→1.44——**深挖修正（§C.5.1）**：机制是
   "一次修对率上升（29.2%→65.2%）→ 修复需求下降"，**不是"学会迭代修复"**：
   失败反馈后的修复成功率未升（基座 11.1% → GRPO 0%，15~18 题小样本），
   方案 §三"二次提交通过率上升"的字面标准**未满足**（如实记录）。
4. **dup_ref / grade_timeout 全程钉死 6 / 2**：四个 checkpoint 无一恶化，
   C.3 的"无 hacking 演化"结论在曲线上再次坐实。

**结论**：GRPO 对 RFT 取得稳定正增量（峰值 +2.2pt、末轮 +1.4pt），
格式修复是最大单项收益；**部署应选 iter1 而非 iter3**。行为解读的修正与
修复增量的直接量化见 §C.5（轮数收敛机制 = 一次写对率上升 + 过度编辑灭绝）。
后续突破 0.75 平台的杠杆：早停 iter1、降 lr/升 KL β、扩大题池（89 题信号偏稀，
零优势组占比升高见 C.1），**或补强 F 分支——多 case 可见反馈 / 保持型奖励 /
回溯信用分配（§C.5.4，全部 headroom 所在）**。

### C.5 修复行为深挖（2026-09-06，`c4_deepdive.py`——M3 行为曲线的细化）

**动机**：方案 §三"良好"标准点名"二次提交通过率上升"；C.1/C.4 的"轮数收敛=学会修"
是推断性解读。本节用既有评测日志把"修复"直接量化：Phase 1（纯日志）修复成功率族 +
Phase 2（首跑代码沙盒离线重判分）隐藏口径修复增量。

#### C.5.1 Phase 1：修复成功率族

一次修对 = n_runs==1 且隐藏满分；修复成功 = 拿到过失败反馈（F/E）的轨迹终版满分。

| 模型 | n_saw_fail | 修复成功率 | 可见修复率(F后修到P) | 一次修对率 | 多跑(≥2)满分率 |
|---|---|---|---|---|---|
| 基座多轮 | 18/72 ran | 11.1% | 33.3% | 29.2% | 12.5% |
| RFT 多轮 | 17 | 11.8% | 29.4% | 61.8% | 11.8% |
| GRPO iter0 | 17 | 5.9% | 17.6% | 62.9% | 5.9% |
| GRPO iter1 | 15 | **0%** | 13.3% | **65.2%** | **0%** |
| GRPO iter2 | 17 | 0% | 17.6% | 64.0% | 0% |
| GRPO iter3 | 16 | 0% | 18.8% | 64.0% | 0% |

1. **GRPO 收益的主引擎是"一次修对率" 29.2%→65.2%**（RFT 贡献大头 29.2→61.8，
   GRPO 再加 +3.4pt）+ 格式修复（88.8→100%）；**"失败后救回"没有涌现**（修复成功率 11%→0%）。
2. 轮数收敛（1.82→1.44）是一次修对率上升的**副产品**（修复需求下降），不是修复能力上升；
   方案"二次提交通过率上升"标准未满足——如实记录，防过度宣称"学会修"。
3. 选择效应注记：F 分支首跑隐藏分逐代下降（0.1335→0.0995→0.0660→0.0709）——
   模型越强，落到失败分支的越是难题，修复率绝对值被压低；但"未上升"的方向跨模型一致。

#### C.5.2 Phase 2：隐藏口径修复增量（首跑代码 vs 终版，配对判分）

把首跑 `<run_code>` 的代码用隐藏测试离线判分（c4_env.grade_solution 沙盒守护），
与终版提交配对。**方法学修正两则**：
- v1 首跑分未过 dup 判 0 门而终版已过 → dup 轨迹虚增负 delta（iter1 的 6 条 dup
  ≈ 贡献 −0.054，几乎解释 v1 全部 −0.0558）——**v1 的"修复净负"是口径伪影**。
  v2 两侧对称门控。教训：**配对比较两侧口径必须对称（门控/归一化同施）**。
- 按首跑可见结果分支：P 分支 = 验证通道（应然 Δ≈0）；F 分支 = 真实修复价值。

| 模型 | 全量 Δ(终−首) | P 分支 Δ (n) | F 分支 Δ (n) | 首跑 gated → 终版 |
|---|---|---|---|---|
| 基座多轮 | **−0.0599** | **−0.0926** (54) | +0.0382 (18) | 0.6344 → 0.5745 |
| RFT 多轮 | +0.0017 | −0.0139 (72) | **+0.0676** (17) | 0.7278 → 0.7295 |
| GRPO iter1 | +0.0004 | **0.0** (74) | +0.0022 (15) | 0.7512 → 0.7516 |
| GRPO iter3 | 0.0 | **0.0** (73) | 0.0 (16) | 0.7435 → 0.7435 |

1. **基座"过度编辑"实锤**：首跑已 PASSED 的轨迹基座还继续改，隐藏口径平均**丢 9.3pt**
   （P 分支 −0.0926）——改坏了原本能过的隐藏 case，是基座配对丢分（−6pt）的真实机制。
2. **GRPO 的行为收益 = 过度编辑损失通道灭绝**：P 分支 Δ 从 −0.0926 收敛到 **0.0**
   （学会"跑一次验证、过了就原样提交"），全量 Δ≈0——提交代码≈首跑代码。
   GRPO 没有打开"修复增益"通道（F 分支 Δ≈0），而是关掉了"乱改损失"通道。
3. **训练收益的大头在首版质量**：首跑 gated 分（配对口径）基座 0.6344 → iter1 0.7512
   （+11.7pt）——"第一遍写出的代码"大幅变强，多轮行为的价值收敛为轻量验证。
4. F 分支修复增量在基座/RFT 为正（+3.8/+6.8pt，部分分）但绝对值极低（终版 0.17）；
   GRPO ≈0——修复行为存在但弱，且没有被 RL 放大。

#### C.5.3 题级 diff：iter1 峰值 = 单题翻转，非系统性过拟合

iter1→iter3：平均 Δ −0.0081，**掉分仅 1 题**（c4_he_order_by_points 1.0→0.073：
iter1 一跑修对，iter3 陷入 5 跑全败循环——该题负数位和规则怪异，5 轮没修对
docstring 可见断言），新满分 0 题。n=89 单题噪声 ±1.1pt > iter1-iter3 差 0.8pt——
C.4"峰值"的正确解读是 **iter1 起进入平台**，iter1 为点估计最高的部署选型。

#### C.5.4 跨项目对照（检索 Agent 项目镜像分析，logs/eval_*_greedy_n500.json）

| | 检索 Agent（多跳 QA，n=500 贪心） | 代码 Agent（函数级修码，89 题贪心） |
|---|---|---|
| 任务天然多步？ | 是（**一搜即对率 0%**，多搜占比 ~100%） | 否（一次修对率可达 65%） |
| RL 收益形态 | **多步行为内终答质量**：多搜轨迹 EM 1.40%→26.71%；效率收敛 avg_search 3.96→2.65（逼近 2-hop 下界） | **一次成功率**：一次修对率 29.2%→65.2% + 格式 100%；失败后修复未涌现 |
| 贪心峰值位置 | run3 iter0（26.60 > 26.20 > 23.80） | iter1（0.7516，iter2 起平台=单题翻转） |
| 训练分/部署分背离 | 有（选 iter0） | 有（选 iter1）——**跨环境复现** |

**统一结论：RL 收益形态由任务结构与奖励地形决定**——一步可恢复任务（函数级修码）上
"一次做对"就是最优路径、修复行为无梯度信号（难题组内全败→零优势被过滤），
是 μCode"一步可恢复 MDP"（arXiv 2502.20380）的实证注脚；天然多步任务（多跳 QA）上
RL 才在多步行为内部提质量。要让"修复"涌现，杠杆在：**多 case 可见反馈**（单 assert
信号太弱，编辑检测不到改坏了哪些 case）、**保持型奖励**（终版 vs 首跑对比，惩罚
改坏已通过行为）、**回溯信用分配**（MURPHY 的动机正在于此）——F 分支（15~18 题、
终版仅 0.07~0.17）就是全部 headroom 所在。

复现：`python3 c4_deepdive.py --phase2`（Phase 2 需沙盒，容器内跑；逐行结果
`logs/c4_deepdive_p2_rows.jsonl`，339 行）。

## 三-D、规模化与信用分配迭代轮：vLLM 定制实证（WS-1）+ GiGPO turn 级优势（WS-2）（2026-09-07）

> 动机与方法论细节见 `SCALE_UP.md`（规模放大方案）；本节只记实验事实。

### D.1 WS-1 vLLM 深度定制实证（A0→A2）

- **A0 冒烟**（`poc_vllm_bench.py`，`logs/poc_vllm/a0_*`）：sm_70+vLLM1.3.0 上
  `enforce_eager=False` 可用（CUDA graph n=8 冒烟 2.15×：55.4→120.5 out tok/s）；
  **prefix caching V1 默认 ON**（审计发现，"没配置"≠"没开"）；util 0.75→0.9 阶梯
  无 OOM（KV 101k→124k tokens）；多轮负载前缀重叠率 analytic LCP 72–73%。
- **A1 六配置矩阵**（89 题×G1×贪心交付负载，预注册不变性判据）：
  CUDA graph 三臂 generate_wall **−35.2%~−35.5%**（90.3→58.3s，290→443 tok/s），
  5 配置全部 **0/89 翻转**（answer_norm 失配 1–3 题=fp 归约平局预期）；
  prefix ON/OFF 仅差 1.2%（噪声内，小并发贪心 decode-bound）；util0.9 无净收益。
  **胜出配置 = prefix 显式 ON + CUDA graph + util0.7**。跨 run 噪声 ~10%
  （config0 vs config1 名义同配置差 12.5%，如实记录）。config0 avg_pass=0.7516
  与历史交付评测逐位一致=复现保真锚点。
- **A2 共置量化**（`checkpoints/c4_a2_probe/`）：训练 iter 全环 19.6min 中
  **rollout 相 78.8% vs 训练相 21.2%**（graph 臂 18.7min）；换载税本体（引擎重建）
  仅 ~2–3%——瓶颈不是换载而是 rollout 串行。拆分式见 `SCALE_UP.md` §3.3。

### D.2 WS-2 GiGPO 式 turn 级优势（Stage 0→2）

- **Stage 0 离线反事实**（`logs/poc_gigpo/stage0_report.json`，0 GPU）：
  ① 零优势过滤主要滤全对饱和组（54 组中 43 组 reward=1.2）；② all-same 组锚点
  子组继承零方差 →"GiGPO 新增信号=0"是数学必然，真实机理=mixed 组内信用重分配
  （53.8% 非零优势 step 被实质改变）；③ 修复行为在采样层存在（mixed 组首跑失败
  轨迹 18/59=31% 可见修复）→ 预注册假设改写为"信用锐化→贪心修复率>0%"。
  **γ 修订 0.95→1.0**（γ<1 在同分组引入长度效率信号，污染单机制归因）。
- **Stage 1 实现+单测**：`mask_utils.encode_chat_assistant_turns`（零回归新增）、
  `c4_gigpo.py`（锚点=(task_id, 可见判定 P/F/E 全前缀)）、`c4_train_gigpo.py`
  （c4_train 姊妹拷贝，超参全不动，恒等性断言）；CPU 单测 31/31。
- **Stage 1.5 配对 replay 方向门**（`logs/poc_gigpo/stage15_compare.json`）：
  R1(gigpo)−R0(epi)=**0.0000**（双双 0.7401、0 翻转、修复指标族与 Phase2 隐藏判分
  逐位相同）→ 主判据/备选 FAIL、方向门 PASS = **GATE-PASS-NO-SIGNAL**：
  单 iter 弱更新（lr 5e-6×192）零行为分化，无害确认、机制起效零宣称，
  按预注册进 Stage 2。replay 保真度 R0 vs 历史 iter1 = −0.0115（如实报告）。
- **Stage 2 全量 4 iters**（rollout 吃 WS-1 胜出配置 graph+prefixON；评测两侧
  同 CLI 同引擎贪心）：**预注册主判据 PASS**——贪心 avg_pass iter0→3 =
  0.7182 / **0.7561** / **0.7555** / 0.7200，iter1/iter2 双点超过成功线 0.7526
  （epi 历史峰值 0.7516 单点）；**贪心修复成功率从 epi 四 iter 全 0 →
  iter0 2/17、iter1 1/15**；iter0 F 分支隐藏修复 Δ +0.0688 = epi（+0.002）的
  34×；iter1/iter2 题级 0 掉分各 +1 新满分（c4_mb_615、c4_he_order_by_points
  ——epi 从未修好的顽固题，5-run 深度修复）。iter3 回落 0.7200 的 3 道掉分
  **全部是 dup_ref 背题判 0**（非能力退化，visible_fixed 保持 0.1333）。
  诚实边界：+0.0045 峰值差在单 run 噪声量级，协同证据方向一致但**不宣称大效应**；
  部署选点仍按贪心 iter1。全表与逐项判定：`SCALE_UP.md` §6.4、
  `logs/poc_gigpo/stage2_analyze.json`。

## 四、奖励设计

`R = 1.0*r_pass + 0.2*r_format − 0.2*r_over`
- `r_pass` = 隐藏测试通过率（有 plus 用 plus 逐 case 率，否则原始 assert 率）；
  **终止门控**：无合法 `<answer>` 或提取不到代码 → 0；
- **查重门**：提交代码与参考解归一化相同 → r_pass 置 0（dup_ref=True 落日志）；
- `r_format` = 全部轮合法 + 以 answer 结束 + 至多一个 answer + 代码可提取；
- `r_over` = 超 MAX_RUNS/MAX_TURNS 的惩罚（封顶 1.0）。
- 过程奖励默认不加（μCode"一步可恢复 MDP"依据，arXiv 2502.20380）。

## 五、坑记录（后人勿踩）

1. **docker 必须 `--security-opt seccomp=unconfined`**：默认 seccomp 拦截
   `unshare -rn`，禁网静默降级（c4_env 有缓存探测+优雅降级，但防御就没了）。
2. **AST 变异初版 bug**：位点收集携带旧树节点引用，变异改旧树、unparse 新树
   → 所有"变异体"==参考解原文 → built=0。修复：`NodeTransformer` 按前序遍历
   序号定位，变异一定落在被 unparse 的树上（`c4_tasks.py` 注释有详述）。
3. **RLIMIT_FSIZE=0 测试要显式 flush**：`open(...).write(...)` 临时对象在 GC
   flush 时的 EFBIG 会被 Python 忽略（rc 仍 0），初版测试因此误判禁写失效。
4. **防泄漏审计防"空串恒真"**：plus_ok=False 的任务 harness_prefix=""，
   `"" in blob` 恒 True，审计须先判非空。
5. **docker 内产物是 root 属主**：宿主机 rm 不掉，删除/覆盖走 docker 内执行。
6. **heredoc 进 docker 要 `docker run -i`**：少了 `-i` stdin 不进容器，脚本静默不执行。
7. MBPP prompt 字段自带 geeksforgeeks URL——模型可见无妨（禁网下访问不了，
   本身就是禁网必要性的现场论据）。
8. **`--rm` + 后台/管道运行长任务 = 容器静默死亡**（M1 构建连死两次的根因，
   取证过程：第三次去掉 `--rm` 后 client 被 SIGTERM 杀掉、容器照样存活运行）：
   docker client（nohup &、tee/grep 管道链、harness 超时清理都会杀它）死亡时，
   `--rm` 会触发容器 stop+rm——无 traceback、无 OOM 记录、日志停在半路，
   表象酷似进程崩溃。**长任务铁律：`docker run -d --name xxx`（detached，
   生命周期与会话彻底解耦），产物走 -v 挂载落盘，日志用 `docker logs` 取，
   结束后 `docker rm`**。构建类脚本必须有进度打印（`[build] i/N`），
   日志静默无法区分"在跑"与"死了"。
9. **变异体验证必须在带 rlimit 的子进程里跑**（M1 构建三连死的真正根因，
   完整取证链见下）。`c4_he_get_odd_collatz`：collatz 函数的单点变异
   （如 `3*n+1`→`3*n-1`）产生**整数指数爆炸**——n 每步翻倍增长，单次 bigint
   乘法是 C 级调用，SIGALRM 的 Python handler 只在字节码边界执行，根本拦不住；
   实测 10 分钟内存膨胀到 46.8GiB（`docker stats` 取证），前两次构建的
   "静默死亡"= 膨胀到宿主机 OOM killer SIGKILL（所以无 traceback、无退出码）。
   修复：`_try_exec_visible`/`_plus_rate` 的验证 exec 移入 fork 子进程，
   RLIMIT_CPU（SIGXCPU 内核级杀，不依赖字节码边界）+ RLIMIT_AS 4GB
   （bigint 分配失败 2.6 秒即 MemoryError）+ 父进程超时 kill 三层防护。
   两个配套语义修复：① MemoryError/RecursionError/OverflowError 是"跑飞"
   必须弃用变异体（返回 None），不能当"断言失败"（False）收进任务集；
   ② `_plus_rate` 逐 case 的 `except Exception` 会吞掉 MemoryError，
   资源型异常必须 re-raise 给 guard 判废。
   **教训通用化：任何"执行不可信/变异代码"的验证环节，进程内 SIGALRM
   不是超时保证，rlimit 子进程才是。**
10. **task_id 撞车 = 判分静默错乱**：HumanEval 存在不同题共用函数名
    （add/solve/sum_squares/correct_bracketing 各×2），tid=c4_he_{fn} 撞车后
    hidden dict 加载时后行覆盖前行 → 8 行任务全部用错误参考解判分，且无任何
    报错。暴露信号是**数字不自洽**（keep 89 + excluded 286 = 375 ≠ 378）。
    修复：tid 带 HE 题号（c4_he_{fn}_{num}）+ c4_fix_dup.py 重命名/8 题补跑/
    重算筛选。教训：任何按 id join 的两份文件，落盘后必须断言 id 唯一；
    对账数字（总数 = 各分支之和）是最便宜的完整性检查。
11. **vLLM 1.3.0 EngineCore fork 死锁：训练循环内建引擎必须
    `VLLM_WORKER_MULTIPROC_METHOD=spawn`**。非 replay 训练路径 4/4 挂死在
    "Using V2 Model Runner" 后（挂 17-18h 不恢复，GPU 330MiB 权重未加载）；
    同镜像下评测 CLI / replay / 裸 `LLM()` 均正常。E1-E12 全新容器二分定位：
    触发条件=**模块组合**（torch+transformers+peft+自研训练栈 import 链共同在场，
    单成分均不触发）——父进程带线程态 fork，EngineCore 子进程死在权重加载前的
    父子握手（经典 fork-while-threaded 死锁族）。修复=spawn 起子进程，
    真实路径验证 init 24.19s。完整取证链见 `ENGINE_HANG_RCA.md`。
    教训：容器挂死先看活体早期信号（引擎 init ~1min 内应出现
    "Loading model from scratch"），别等 OOM/traceback——死锁两者都没有。

## 六、里程碑进度

| 里程碑 | 状态 | 备注 |
|---|---|---|
| M0 环境原型 | ✅ 2026-09-05 通过 | 本文件 §三；止损门未触发 |
| M1 基线（直答/多轮/RFT） | ✅ 2026-09-06 收官 | §三-B.2c 三组数字总表；GRPO 目标线 0.7295；污染率 76.5% 实证 |
| M2 主实验（多轮 GRPO） | ✅ 2026-09-06 收官 | §三-C：贪心峰值 iter1 0.7516（+2.2pt over RFT），格式一步到 100%，无熔断；训练层零改动复用检索 Agent 项目 |
| M3 评测 + 消融 | ✅ 2026-09-06 收官 | 三组数字（§B.2c）+ hacking 抽查（§C.3）+ iter0→3 行为曲线（§C.4）齐；曲线非单调，部署选 iter1 |
| M4 开源 + 博客 | ✅ 2026-09-06 收官 | 开源（代码+README+LICENSE+requirements+实验日志，大体积产物 .gitignore 排除）；技术博客 `BLOG_代码AgentRL.md`。红线达成（数字全真实跑出后才记录）|
| 规模化与信用分配迭代轮（WS-1 vLLM 实证 / WS-2 GiGPO / WS-3 放大方案） | ✅ 2026-09-07 收官 | WS-1 ✅（§三-D.1，−35.5% 头条+0 翻转+共置 79:21）；WS-2 ✅ Stage 0→2 全线（§三-D.2，Stage 2 主判据 PASS：峰值 0.7561、贪心修复率 0→11.8%/6.7%）；WS-3 ✅ `SCALE_UP.md` 实测锚点全部回填 |
| LCB held-out 扩池评测 | ✅ 2026-09-19 收官 | §七 + `LCB_HELDOUT.md`；池外 21 题 GRPO 0.9091 保持领先（超 RFT +6.7pt），GiGPO 增益未迁移如实报告 |

## 七、LCB held-out 扩池评测（2026-09-19）——✅ 收官

train89 的训练集与评测集同池，缺少池外泛化证据（§B.1b 局限）。用 LiveCodeBench
test6（2025-01~04，175 题）构建池外修码任务 21 题（基座 32 采样生成参考解 + 同一套
AST 变异管线；easy 12 / medium 7 / hard 2），6 个 checkpoint 同口径贪心评测。
预注册、构建迭代、完整结果与诚实边界见 `LCB_HELDOUT.md`。

| checkpoint | avg_pass | submit | hard(n=2) |
|---|---|---|---|
| 基座直答 | 0.8378 | 1.000 | 0.000 |
| 基座多轮 | 0.7143 | 0.762 | 0.000 |
| RFT | 0.8422 | 0.905 | 0.000 |
| **GRPO iter1** | **0.9091** | 0.952 | **0.702** |
| GiGPO iter0/1 | 0.8422 | 0.905/0.952 | 0.000 |

要点：① 主线 GRPO 池外保持领先（超 RFT +6.7pt / 超基座多轮 +19.5pt），无过拟合
train89 迹象；② GiGPO 的 train89 增益未迁移（如实报告）；③ 直答满分率 76.2% 触发
预注册污染标注（dup=0、hard 层 0/2 为反证，结论以相对比较为主）；④ n=21 单 run、
窗口紧邻模型发布点，边界随数字一并报告。
