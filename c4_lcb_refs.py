"""C4 LCB held-out · Phase B：候选并行验证 → 参考解（docker 内跑，CPU 多进程）

预算纪律（预注册口径的实现细节）：每题按序验证候选，首个全量通过的取为参考解；
失败候选满 12 个即判 no_ref（预算耗尽），单候选沙盒超时 20s。
并行：mp.Pool（fork），进程数 = min(12, cpu)。

用法（docker 内，无需 GPU）：python c4_lcb_refs.py
输出 data/lcb/refs.jsonl：{tid, fn_name, difficulty, n_pass, n_tried,
                          budget_exhausted, ref_code}
"""
import json
import multiprocessing as mp
import os

from c4_lcb_pool import LCB_PRELUDE, run_check

FAILED_BUDGET = int(os.environ.get("LCB_FAILED_BUDGET", "32"))
TIMEOUT = 20


def _worker(row):
    """单题验证：按序验证候选，早停于首个全过；失败预算 FAILED_BUDGET。
    参考解统一带 LCB_PRELUDE 存储（与 c4_tasks 的 ref_code 含头约定一致）。"""
    fn_name = row["fn_name"]
    n_pass = 0
    n_tried = 0
    n_tests = 0
    ref_code = None
    for cand in row["cands"]:
        if n_tried - n_pass >= FAILED_BUDGET:
            break
        n_tried += 1
        all_pass, (cp, cn) = run_check(LCB_PRELUDE + "\n" + cand["code"],
                                       fn_name, row["tests"], timeout=TIMEOUT)
        n_tests = cn
        if cp == cn and cn:
            n_pass += 1
            if ref_code is None and all_pass:
                ref_code = LCB_PRELUDE + "\n" + cand["code"]
    return {"tid": row["tid"], "fn_name": fn_name,
            "difficulty": row["difficulty"], "n_cands": len(row["cands"]),
            "n_tried": n_tried, "n_pass": n_pass, "n_tests": n_tests,
            "budget_exhausted": (n_tried - n_pass) >= FAILED_BUDGET and ref_code is None,
            "ref_code": ref_code}


def main():
    cands = {json.loads(l)["tid"]: json.loads(l)
             for l in open("data/lcb/candidates.jsonl")}
    pool = {json.loads(l)["tid"]: json.loads(l) for l in open("data/lcb/pool.jsonl")}
    rows = [{**c, "tests": pool[tid]["tests"]} for tid, c in cands.items()]

    workers = min(12, os.cpu_count() or 4)
    with mp.Pool(workers) as pool_mp:
        out_rows = []
        for i, r in enumerate(pool_mp.imap_unordered(_worker, rows, chunksize=1)):
            out_rows.append(r)
            print(f"[refs] {i + 1}/{len(rows)} {r['tid']} pass={r['n_pass']}/{r['n_tried']} "
                  f"ref={'Y' if r['ref_code'] else 'N'}", flush=True)

    with open("data/lcb/refs.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    found = sum(1 for r in out_rows if r["ref_code"])
    by_diff = {}
    for r in out_rows:
        d = by_diff.setdefault(r["difficulty"], {"n": 0, "ref": 0})
        d["n"] += 1
        d["ref"] += int(bool(r["ref_code"]))
    print(json.dumps({"pool": len(out_rows), "ref_found": found,
                      "by_difficulty": by_diff}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
