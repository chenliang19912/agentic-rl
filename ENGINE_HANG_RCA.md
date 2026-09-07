# vLLM 1.3.0 V1 EngineCore 初始化死锁：排查记录与修复（2026-09-07）

## 现象

`c4_train_gigpo.py` **非 replay 路径**（即训练循环内构建 vLLM 引擎做 rollout）的容器
100% 挂死：EngineCore 子进程打印 `Using V2 Model Runner` 后无任何进展（下一步本应
是 `Loading model from scratch...`），GPU 330MiB（权重未加载）、util 0%，
挂 17-18 小时不恢复。**4/4 复现**（A2 eager 臂 ×2、S2 graph 臂 ×2）。

同镜像同负载下**不受影响**的路径：`c4_rollout.py` CLI（B16 评测、A1 六配置全矩阵）、
`poc_vllm_bench.py`、`c4_train_gigpo.py --replay`（不建引擎）、裸 `python -c` 直接
`LLM()`——历史上 `c4_train.py` 也成功跑完 4 iters。

## 二分定位（E1-E12，每条=全新容器、110s 超时、判据=是否出现 Loading model）

| # | 父进程 import/操作 | 结果 |
|---|---|---|
| E1 | torch + vllm + LLM() | **过**（权重 5.39s） |
| E4 | torch+transformers+peft+CPU自检+tokenizer加载+vllm | **挂** |
| E5 | torch+transformers+vllm | 过 |
| E6 | torch+peft+vllm | 过 |
| E7 | torch+transformers+peft+tokenizer加载+vllm | 过 |
| E8 | `import c4_train_gigpo`（全链）+ selftest + build_llm | **挂** |
| E11a | torch+numpy+vllm（fork 默认） | 过 |
| E12 | torch+`import m3_env`（numpy+BM25）+vllm | 过 |
| E8b | E8 同路径 + **VLLM_WORKER_MULTIPROC_METHOD=spawn** | **过**（init 24.19s，KV 101,040 tokens） |

## 结论

- 触发条件是**模块组合**（torch/transformers/peft + 自研训练栈 import 链共同在场；
  单独任何一个成分都不触发，E5/E6/E7/E11a/E12 全过）。精确到"哪两个模块的组合"
  未继续深挖——4 次复现 + 修复验证已满足工程需要。
- 机制定性：vLLM V1 的 EngineCore 子进程在 Python <3.14 默认用 **fork** 起进程；
  父进程带着上述组合的运行时状态（线程池/锁，典型如 BLAS/OpenMP 线程注册）fork，
  子进程在首次与父进程握手（权重加载前的消息泵）处死锁。这属于经典的
  fork-while-threaded 死锁族。
- **修复：`-e VLLM_WORKER_MULTIPROC_METHOD=spawn`**（docker run 环境变量）。
  E8b 在完整真实路径上验证通过；随后 A2/S2 双容器带此变量重启，全部恢复正常
  （引擎初始化 + rollout 正常推进）。

## 对本仓库的影响面

- 所有"训练循环内建 vLLM 引擎"的入口（`c4_train_gigpo.py` 非 replay、未来同类
  训练脚本）在 sm_70 定制镜像 1cat-vllm:v100-1.3.0 上**必须带 spawn 变量**。
- 评测类入口（`c4_rollout.py` CLI）不受影响但带上也无害（spawn 略增引擎启动
  开销 ~秒级）。
- 已计入 SCALE_UP.md §5.1（引擎深度定制路线：fork 死锁排查与修复本身是
  "深度定制"的证据链一环）。
