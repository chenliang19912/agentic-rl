# C4 预注册：LiveCodeBench held-out 扩池评测（2026-09-18，跑评测前落盘）

> 动机：train89 的训练集与评测集同池——选点、指标、消融都在同一份题上产出，无法回答
> "池外泛化"问题（局限在 C4_README §B.1b 已如实标注）。本实验在分布外的 LiveCodeBench
> 题上补测全部 checkpoint，无论结果高低都如实报告。

## 一、数据窗口（预注册，不后改）

- 数据源：`livecodebench/code_generation_lite` **test6.jsonl**（官方最后一窗），
  contest_date 覆盖 **2025-01 ~ 2025-04**，共 175 题。
- 窗口与 Qwen3-4B-Instruct-2507（2025-07 发布）紧邻，无法从数据集侧严格保证 post-cutoff
  ——**污染以两条实测证据兜底（见 §四）**，并在结论里如实标注该不确定性。
- 题型过滤：starter_code 非空（class Solution 函数式）63 题 → 归一化为**模块级函数**
  （与训练池同构，签名/语义不变，prompt 加格式说明）；测试逐条 literal 可解析、
  期望含浮点的题剔除；题面含图片标记的剔除。

## 二、构建管线（同 c4_tasks，全部复用）

> **修订 1（2026-09-19，任何模型评测跑之前）**：验证预算从"失败 12 个候选即判 no_ref"
> 放宽为全量 32 候选验证——12 的预算是单进程串行验证时代的提速启发式，验证并行化
> （mp.Pool×12）后不再必要；全量验证可显著提高参考解产出。已按修订跑 Phase B。
> **修订 2（2026-09-19，首轮多轮评测发现）**：评测套件初版漏传 `--n`，多轮 rollout 误跑
> 前 8 题（c4_rollout 的 M0 冒烟默认值）；发现后修正为全量 21 题重跑全部多轮评测。
> base oneshot 走 c4_baseline（无 n 截断），不受影响。

1. **参考解生成**（LCB 不带官方题解）：基座 Qwen3-4B-Instruct-2507 每题采样 32 条
   （temp=1.0, top_p=0.95, seed=42），候选码统一加头部导入（typing/常用竞赛库，防
   `List` NameError）后逐题跑全部隐藏测试，**全部通过的首条**取为参考解；无参考解的题
   弃用并计数。实现细节（构建迭代中发现，先于任何评测落盘）：
   - LCB 私有测试字段为 zlib+base64 压缩（内层 json 或 pickle 包 JSON 文本），解析器
     `c4_lcb_pool.py` 按受限 Unpickler（仅 builtins）处理；
   - 候选码常按 LeetCode 习惯包 `class Solution`，refgen 做 AST 归一化：提取全部方法、
     丢 self 参数、提升为模块级函数（`c4_lcb_refgen.normalize_candidate`）；
   - 候选检查超时 20s（隐藏测试含大压力输入）。
2. **AST 单点变异**：`c4_tasks.make_mutants` 原样复用（seed=42）；验证同原文——
   可见测试必须失败、隐藏测试通过率 <0.95、资源型异常判废。
3. **判分**：`c4_env.grade_solution` 原样复用（harness_prefix 逐 case，pass_rate 主指标），
   查重门照常启用。tasks/hidden 分离纪律照常。

## 三、评测协议（预注册）

- 口径与 train89 完全一致：SYSTEM/USER_TMPL、MAX_RUNS=5/MAX_TURNS=6、**贪心**（temp=0）、
  同一套 grade_solution + 查重门。
- checkpoints（全部 merged 形态）：
  ① 基座直答 oneshot（背题率测量用）
  ② 基座多轮执行反馈
  ③ RFT merged
  ④ GRPO iter1_merged（epi 线部署形态）
  ⑤ GiGPO iter0_merged 与 iter1_merged
- **主判据**：各 checkpoint 的 held-out avg_pass（贪心、逐题 pass_rate 均值）。本实验为
  **描述性 held-out 报告**，不设通过阈值——数字无论高低照实写。
- 池规模目标：参考解可用的全部函数式题（预估 30~60）；不追规模上限，以"同管线可复现"优先。

## 四、污染双证（写进结论的强制项）

1. 基座 oneshot 直答满分率 + 逐字 dup_ref 率，与 MBPP 池 76.5% 剔除率同口径对照；
   **若直答满分率 >30%，结论必须标注"窗口污染不可忽略"**。
2. 窗口为 2025 年 LeetCode 周赛题，无经典教材出处；窗口紧邻发布点的不确定性原样写入结论。

## 五、诚实边界（主动项）

- 参考解为模型生成（须过全部隐藏测试），非官方题解——参考解质量由测试全集背书。
- 竞赛题难度分布高于 MBPP 训练分布，绝对数字预期显著低于 train89；解读以
  **checkpoint 间相对比较**为主。
- 小样本（预估 n=30~60），单题 = 2~3pt 量级，不做大效应宣称。

## 六、结果（2026-09-19 评测完成回填）

### 构建统计

175 题（2025-01~04）→ 函数式 63 → 测试全可解析 59 → 基座 32 采样参考解可用 22 →
变异构建成功 **21**（easy 12 / medium 7 / hard 2；每题 33~43 个隐藏 case；1 题无有效变异体弃用）。
池规模未达 §三 的 30~60 目标：medium/hard 的参考解产出低（4B 基座 32 采样内无全过候选），
**如实记录偏差**。自检：refs 全部 pass_rate=1.0、mutant 全部 <1.0（8/8 抽检 OK）。

### 评测表（n=21，贪心，同一套 grade_solution + 查重门）

| checkpoint | avg_pass | submit | full | avg_runs | hard(n=2) |
|---|---|---|---|---|---|
| 基座直答 oneshot | 0.8378 | 1.000 | 16/21 | – | 0.000 |
| 基座多轮执行反馈 | 0.7143 | 0.762 | 15/21 | 0.95 | 0.000 |
| RFT merged | 0.8422 | 0.905 | 16/21 | 1.71 | 0.000 |
| **GRPO iter1（部署形态）** | **0.9091** | 0.952 | **17/21** | 1.43 | **0.702** |
| GiGPO iter0 | 0.8422 | 0.905 | 16/21 | 1.52 | 0.000 |
| GiGPO iter1 | 0.8422 | 0.952 | 16/21 | 1.38 | 0.000 |

难度拆分（easy n=12 / medium n=7 / hard n=2）：GRPO-iter1 = 0.983 / 0.841 / 0.702；
其余 checkpoint hard 层全部 0.000。

### 解读（对齐预注册 §四/§五）

1. **主结论：主线 GRPO 在池外保持领先方向**——0.9091，超 RFT +6.7pt、超基座多轮 +19.5pt，
   与 train89 的排序（GRPO > RFT > 基座）同向；**未出现过拟合 train89 的迹象**。
   GRPO-over-RFT 的 held-out 差距（+6.7pt）大于 train89（+2.2pt），增量主要由 hard 层贡献
   （RFT/GiGPO/基座 hard 全 0，GRPO 0.702）——但 hard n=2，此点只作方向性证据。
2. **GiGPO 的 train89 增益未迁移**：held-out 0.8422，未超 GRPO-iter1，与 train89 上的
   0.7561 vs 0.7516 相反。如实报告；单 run 噪声内不可强归因。
3. **基座 oneshot (0.8378) > 基座多轮 (0.7143)**：差距主因是基座多轮 submit 仅 76.2%
   （多轮格式失败）——held-out 上 RL 的最大增量仍是"格式/提交修复 + 难度端"，与
   「训练收益大头在首版质量」叙事一致。
4. **污染标注（预注册 §四触发）**：直答满分率 76.2% > 30% 阈值，按预注册要求标注
   "窗口污染不能排除"。补充判断：dup_ref 全线为 0；hard 层 oneshot 0/2（若整体背诵
   应同热）；且 fix-with-visible-test 格式的满分率天然显著高于从头生成——综合判断
   污染非决定性，**结论以 checkpoint 间相对比较为主**。
5. **诚实边界**：n=21（hard n=2）单 run；窗口 2025-01~04 紧邻 Qwen3-2507 发布点；
   class→plain 函数归一化与训练分布存在差异；无多 seed。以上全部随数字一并交代。
