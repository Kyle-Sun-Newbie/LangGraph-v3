# app/experiments/run_experiments.py
import argparse
import time
import json
import random
from pathlib import Path
from tqdm.auto import tqdm

from .config import OUT_DIR, MORTAR_DIR, DEFAULT_TTL
from .data_loader import load_dataset
from .sparql_eval import run_sparql, metrics_set_based, metrics_against_rows, rows_preview
from .report_utils import save_report, compute_summary, save_summary_json

# Schemes
from .schemes.fewshot_faiss import build_index, generate_sparql as fewshot_generate
from .schemes.langgraph_like import run_langgraph_like_scheme
from .llm import LLMClient
from .schemes.hnsw import run_scheme_2  # Scheme 2（HNSW）

# Scheme 6（时序 + TTL 实体对齐）
from .timeseries import (
    build_site_resolver, load_timeseries_dataset,
    compute_stats_for_sites, choose_by_comparison,
    build_answer_text, summarize_rows, DEFAULT_EPS
)

# ---------------- 工具函数 ----------------
def _ttl_path_from(brick_model: str) -> Path:
    if not brick_model:
        return DEFAULT_TTL
    p = Path(brick_model)
    if not p.suffix:
        p = Path(MORTAR_DIR) / p
    if not p.is_absolute():
        p = (Path(MORTAR_DIR) / p).resolve()
    return p


def _gold_rows(ttl: Path, ex: dict):
    """取 gold 行集（优先用样本里的 gold_rows，否则执行 gold SPARQL）。"""
    provided = ex.get("gold_rows") or ex.get("result") or []
    if provided:
        return provided, "provided", False
    goldq = ex.get("answer_sparql") or ""
    if goldq:
        try:
            res = run_sparql(ttl, goldq).rows
            return res, "executed", False
        except Exception:
            return [], "executed", True
    return [], "none", False


def _build_suffix_from_args(args) -> str:
    parts = []
    if getattr(args, "use_llm", False): parts.append("llm")
    if getattr(args, "train_max", None) and int(getattr(args, "scheme", 0)) == 3:
        parts.append(f"tmax{args.train_max}")
    if getattr(args, "limit_train", None): parts.append(f"train{args.limit_train}")
    if getattr(args, "limit_test", None):  parts.append(f"test{args.limit_test}")
    if getattr(args, "random_test", False): parts.append(f"rand{getattr(args, 'seed', 42)}")
    return ("_" + "_".join(parts)) if parts else ""


# ---------------- Scheme 1（fewshot_rag） ----------------
def run_scheme_1(train, test, use_llm: bool, out_dir: Path,
                 limit_train=None, limit_test=None, save_every=50,
                 filename_suffix: str = "", random_test: bool = False, seed: int = 42):
    if limit_train:
        train = train[:limit_train]
    if limit_test:
        if random_test:
            rng = random.Random(seed)
            k = min(limit_test, len(test))
            test = rng.sample(test, k=k)
            print(f"[fewshot] 测试集随机采样: k={k}, seed={seed}")
        else:
            test = test[:limit_test]

    print(f"[fewshot] 建索引：train={len(train)}")
    idx = build_index(train)
    llm = LLMClient() if use_llm else None

    rows = []
    pbar = tqdm(total=len(test), desc="Predicting (Scheme1)")
    for i, ex in enumerate(test, 1):
        q = ex.get("question") or ex.get("input") or ""
        ttl = _ttl_path_from(ex.get("brick_model", ""))

        pred = fewshot_generate(
            idx, q,
            mode=("llm_generate" if use_llm else "copy_nearest"),
            llm=llm,
            brick_model=str(ttl.name)
        )

        parse_error = False
        try:
            res = run_sparql(ttl, pred).rows if pred else []
        except Exception:
            parse_error = True
            res, pred = [], f"--PARSE-ERROR--\n{pred}"

        goldq_res, goldq_source, goldq_err = _gold_rows(ttl, ex)
        met_label = metrics_set_based(res, ex.get("expected"))
        met_goldq = metrics_against_rows(res, goldq_res) if goldq_res else {}

        row = {
            "id": i,
            "scheme": "fewshot_rag",
            "question": q,
            "brick_model": str(ttl.name),
            "pred_sparql": pred,
            "n_rows": len(res),
            "parse_error": parse_error,
            "pred_rows_preview": rows_preview(res, limit=10),
            "goldq_sparql": ex.get("answer_sparql") or "",
            "goldq_source": goldq_source,
            "goldq_n_rows": len(goldq_res),
            "goldq_parse_error": goldq_err,
            "goldq_rows_preview": rows_preview(goldq_res, limit=10),
        }
        row.update(met_label); row.update(met_goldq)
        rows.append(row)

        if i % save_every == 0:
            save_report(rows, out_dir / f"scheme1_fewshot{filename_suffix}_partial.csv")
        pbar.update(1)
    pbar.close()

    base = f"scheme1_fewshot{filename_suffix}"
    save_report(rows, out_dir / f"{base}.csv")
    summary = compute_summary(rows, "fewshot_rag")
    save_summary_json(summary, out_dir / f"{base}_summary.json")

    print("\n=== JSON Summary (Scheme 1) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return rows


# ---------------- Scheme 3/4/5（LangGraph-like） ----------------
def _select_and_sample(test, limit_test=None, random_test=False, seed=42):
    if limit_test:
        if random_test:
            rng = random.Random(seed)
            k = min(limit_test, len(test))
            return rng.sample(test, k=k)
        else:
            return test[:limit_test]
    return test


def _run_langgraph_like_common(test, out_dir: Path, mode: str,
                               filename_suffix: str = "", save_every: int = 50):
    """mode: "template" | "llm" | "rag_llm" """
    pretty_name = {"template": "langgraph_like_template",
                   "llm": "langgraph_like_llm",
                   "rag_llm": "langgraph_like_ragllm"}[mode]

    rows = []
    pbar = tqdm(total=len(test), desc=f"Predicting ({pretty_name})")

    samples = []
    for ex in test:
        samples.append({
            "qid": ex.get("id") or ex.get("qid"),
            "question": ex.get("question") or ex.get("input") or "",
            "expected": ex.get("expected"),
            "brick_model": ex.get("brick_model", ""),
            "answer_sparql": ex.get("answer_sparql", ""),
            "gold_rows": ex.get("gold_rows") or ex.get("result") or [],
        })

    for i, ex in enumerate(samples, 1):
        q = ex["question"]
        ttl = _ttl_path_from(ex.get("brick_model", ""))

        outs = run_langgraph_like_scheme([{"qid": ex.get("qid"), "question": q}], ttl_path=str(ttl), mode=mode)
        assert len(outs) == 1
        pred = outs[0].get("pred_sparql", "")
        res = outs[0].get("pred_rows", []) or []

        goldq_res, goldq_source, goldq_err = _gold_rows(ttl, ex)
        met_label = metrics_set_based(res, ex.get("expected"))
        met_goldq = metrics_against_rows(res, goldq_res) if goldq_res else {}

        row = {
            "id": i,
            "scheme": pretty_name,
            "question": q,
            "brick_model": str(ttl.name),
            "pred_sparql": pred,
            "n_rows": len(res),
            "parse_error": False,
            "pred_rows_preview": rows_preview(res, limit=10),
            "goldq_sparql": ex.get("answer_sparql") or "",
            "goldq_source": goldq_source,
            "goldq_n_rows": len(goldq_res),
            "goldq_parse_error": goldq_err,
            "goldq_rows_preview": rows_preview(goldq_res, limit=10),
        }
        row.update(met_label); row.update(met_goldq)
        rows.append(row)

        if i % save_every == 0:
            save_report(rows, out_dir / f"{pretty_name}{filename_suffix}_partial.csv")
        pbar.update(1)
    pbar.close()

    base = f"{pretty_name}{filename_suffix}"
    save_report(rows, out_dir / f"{base}.csv")
    summary = compute_summary(rows, pretty_name)
    save_summary_json(summary, out_dir / f"{base}_summary.json")

    print(f"\n=== JSON Summary ({pretty_name}) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return rows


def run_scheme_3(test, out_dir: Path, limit_test=None, save_every=50,
                 filename_suffix: str = "", random_test: bool = False, seed: int = 42):
    test = _select_and_sample(test, limit_test, random_test, seed)
    return _run_langgraph_like_common(test, out_dir, mode="template",
                                      filename_suffix=filename_suffix, save_every=save_every)


def run_scheme_4(test, out_dir: Path, limit_test=None, save_every=50,
                 filename_suffix: str = "", random_test: bool = False, seed: int = 42):
    test = _select_and_sample(test, limit_test, random_test, seed)
    return _run_langgraph_like_common(test, out_dir, mode="llm",
                                      filename_suffix=filename_suffix, save_every=save_every)


def run_scheme_5(test, out_dir: Path, limit_test=None, save_every=50,
                 filename_suffix: str = "", random_test: bool = False, seed: int = 42):
    test = _select_and_sample(test, limit_test, random_test, seed)
    return _run_langgraph_like_common(test, out_dir, mode="rag_llm",
                                      filename_suffix=filename_suffix, save_every=save_every)

# ---------------- Scheme 6（Timeseries + TTL 实体对齐） ----------------
def run_scheme_6(ts_json_paths: list[Path],
                 ttl_path: Path,
                 out_dir: Path,
                 filename_suffix: str = "",
                 eps: float = DEFAULT_EPS):
    """
    Scheme 6：时序统计评测（实体对齐依赖 .ttl 的 ext:siteID [ brick:value "..."]）
    仅输出一份 combined 的 CSV/JSON，不再为每个 JSON 单独落盘。
    """
    resolver = build_site_resolver(ttl_path)

    all_rows = []
    for ts_json_path in ts_json_paths:
        # 读取该数据集（会在 loader 内完成对象名->site_id 解析）
        cases = load_timeseries_dataset(ts_json_path, resolver=resolver)

        for ex in cases:
            # ① 计算所有可用站点的统计
            stats_all = compute_stats_for_sites(ex.series_by_site, ex.column, ex.stat_func)

            # ② 有解析结果时仅在该子集上比较；否则回退全部
            candidate_ids = ex.resolved_site_ids or list(stats_all.keys())
            stats = {k: v for k, v in stats_all.items() if k in candidate_ids}

            pred_site, pred_value = choose_by_comparison(stats, ex.compare_mode)

            # 展示名（优先 ttl 的 label/本地名）
            display_name = resolver.best_name(pred_site) if pred_site else None
            answer_text = build_answer_text(pred_site, pred_value, site_name=display_name)

            gt = ex.groundtruth or {}
            gt_site = gt.get("site")
            gt_value = gt.get("value")
            gt_site_name = gt.get("site_name")

            site_correct = (str(pred_site) == str(gt_site)) if (gt_site is not None) else None
            abs_err = None
            if gt_value is not None and isinstance(pred_value, (int, float)):
                abs_err = abs(float(pred_value) - float(gt_value))

            all_rows.append({
                "dataset": ts_json_path.name,  # 记录来自哪个 JSON（只在 combined 中展示）
                "id": ex.case_id,
                "scheme": "timeseries_eval",
                "question": ex.question,
                "pred_site": str(pred_site) if pred_site else None,
                "pred_site_name": display_name,
                "pred_value": pred_value,
                "gt_site": str(gt_site) if gt_site is not None else None,
                "gt_site_name": gt_site_name,
                "gt_value": gt_value,
                "abs_err": abs_err,
                "site_correct": site_correct,
                "answer_text": answer_text,
                "stat_func": ex.stat_func,
                "compare_mode": ex.compare_mode,
                "column": ex.column,
                "params": ex.params,
                # 解析记录（排障用）
                "requested_site_names": ex.requested_site_names,
                "resolved_site_ids": ex.resolved_site_ids,
                "unresolved_site_names": ex.unresolved_site_names,
                "used_candidate_ids": candidate_ids,
            })

    # 仅输出 combined
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"scheme6_timeseries__combined{filename_suffix}"
    save_report(all_rows, out_dir / f"{base}.csv")
    summary_all = summarize_rows(all_rows, eps=eps)
    save_summary_json(summary_all, out_dir / f"{base}_summary.json")
    print(f"[scheme 6] wrote combined rows: {out_dir / (base + '.csv')}")
    print(f"[scheme 6] wrote combined summary: {out_dir / (base + '_summary.json')}")
    return all_rows

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", type=int, default=1,
                    help=("1=fewshot_rag, 2=fewshot_hnsw, "
                          "3=langgraph_like_template, 4=langgraph_like_llm, "
                          "5=langgraph_like_ragllm, 6=timeseries_eval"))
    ap.add_argument("--use_llm", action="store_true",
                    help="Scheme1/2 是否调用 LLM（否则最近邻拷贝）")
    ap.add_argument("--limit_train", type=int, default=None,
                    help="仅使用前N条训练样本（调试/预览，Scheme1生效）")
    ap.add_argument("--limit_test", type=int, default=None,
                    help="仅评测前N条测试样本（调试/预览）")
    ap.add_argument("--save_every", type=int, default=50,
                    help="每N条增量落盘一次")
    ap.add_argument("--random_test", action="store_true",
                    help="随机抽样测试集（仅当设置了 --limit_test 时生效）")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子，用于 --random_test")

    # Scheme 6 专用
    ap.add_argument("--ts_dataset", type=str, default=None,
                    help=("Timeseries JSON 文件或目录（可为目录以读取全部 *.json）。"
                          "例如: F:\\Task\\RAG-LangGraph-Demo-bcp\\data\\dataset_ma"))
    ap.add_argument("--ttl", type=str, default=None,
                    help=("TTL 文件（包含 ext:siteID [ brick:value ... ] 映射）。"
                          "例如: F:\\Task\\RAG-LangGraph-Demo-bcp\\data\\dataset_ma\\PV_System_data.ttl"))
    ap.add_argument("--ts_eps", type=float, default=1e-6,
                    help="数值严格判定的误差容忍度 epsilon（within_eps / strict 指标阈值），默认 1e-6。")

    args = ap.parse_args()

    # SPARQL 相关方案的数据
    train = load_dataset("train_data_")
    test = load_dataset("test_data_")

    out_dir = OUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _build_suffix_from_args(args)
    t0 = time.time()

    if args.scheme == 1:
        run_scheme_1(train, test, use_llm=args.use_llm, out_dir=out_dir,
                     limit_train=args.limit_train, limit_test=args.limit_test,
                     save_every=args.save_every, filename_suffix=suffix,
                     random_test=args.random_test, seed=args.seed)

    elif args.scheme == 2:
        run_scheme_2(train, test, out_dir=out_dir, use_llm=args.use_llm,
                     limit_test=args.limit_test, save_every=args.save_every,
                     filename_suffix=suffix, random_test=args.random_test, seed=args.seed)

    elif args.scheme == 3:
        run_scheme_3(test, out_dir=out_dir, limit_test=args.limit_test,
                     save_every=args.save_every, filename_suffix=suffix,
                     random_test=args.random_test, seed=args.seed)

    elif args.scheme == 4:
        run_scheme_4(test, out_dir=out_dir, limit_test=args.limit_test,
                     save_every=args.save_every, filename_suffix=suffix,
                     random_test=args.random_test, seed=args.seed)

    elif args.scheme == 5:
        run_scheme_5(test, out_dir=out_dir, limit_test=args.limit_test,
                     save_every=args.save_every, filename_suffix=suffix,
                     random_test=args.random_test, seed=args.seed)

    elif args.scheme == 6:
        if not args.ts_dataset:
            raise SystemExit("--ts_dataset 需要指定为一个 .json 文件或一个包含多份 .json 的目录")
        if not args.ttl:
            raise SystemExit("--ttl 需要指定到 PV_System_data.ttl")

        ds_path = Path(args.ts_dataset)
        ttl_path = Path(args.ttl)
        if not ds_path.exists(): raise SystemExit(f"Timeseries 路径不存在: {ds_path}")
        if not ttl_path.exists(): raise SystemExit(f"TTL 文件不存在: {ttl_path}")

        # 支持：单文件 or 目录内全部 *.json
        if ds_path.is_dir():
            ts_json_paths = sorted([p for p in ds_path.glob("*.json") if p.is_file()])
            if not ts_json_paths:
                raise SystemExit(f"目录下未找到任何 .json：{ds_path}")
            print(f"[scheme 6] 发现 {len(ts_json_paths)} 个 JSON：")
            for p in ts_json_paths: print("  -", p.name)
        else:
            if ds_path.suffix.lower() != ".json":
                raise SystemExit("--ts_dataset 必须是 .json 文件或包含 .json 的目录")
            ts_json_paths = [ds_path]

        out_dir_ts = out_dir / "timeseries"
        out_dir_ts.mkdir(parents=True, exist_ok=True)

        run_scheme_6(
            ts_json_paths=ts_json_paths,
            ttl_path=ttl_path,
            out_dir=out_dir_ts,
            filename_suffix=suffix,
            eps=args.ts_eps
        )

    else:
        raise SystemExit("Unknown --scheme")

    print(f"Done in {time.time()-t0:.1f}s, results at: {out_dir}")


if __name__ == "__main__":
    main()
