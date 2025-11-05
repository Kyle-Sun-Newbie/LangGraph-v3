# app/experiments/schemes/hnsw.py
from pathlib import Path
from typing import List, Dict, Any, Optional
import os, random, json, re
from tqdm.auto import tqdm

from ..config import OUT_DIR, MORTAR_DIR, DEFAULT_TTL, ALL_DATASET_DIR
from ..sparql_eval import run_sparql, metrics_set_based, metrics_against_rows, rows_preview
from ..report_utils import save_report, compute_summary, save_summary_json
from ..llm import LLMClient
from .fewshot_faiss import build_index, generate_sparql as fewshot_generate
from ..data_loader import _read_json_or_jsonl, _normalize_record

# ---------------- 工具 ----------------
def _ttl_path_from(brick_model: str) -> Path:
    if not brick_model:
        return DEFAULT_TTL
    p = Path(brick_model)
    if not p.suffix:
        p = Path(MORTAR_DIR) / p
    if not p.is_absolute():
        p = (Path(MORTAR_DIR) / p).resolve()
    return p

def _gold_rows(ttl: Path, ex: Dict[str, Any]):
    """常规 gold 获取：优先使用 provided（原样），否则执行 answer_sparql。"""
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

# ---------- 仅用于 test_data_2：最简数值抽取 + 执行回退 ----------
# 例："(rdflib.term.Literal('2', datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#integer')),)"
_LITERAL_WITH_DT_VALUE_RE = re.compile(
    r"Literal\(\s*['\"]([^'\"]*)['\"]\s*,\s*datatype\s*=\s*rdflib\.term\.URIRef\(", re.I
)
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

def _extract_literal_value_from_str_tuple_minimal(s: str) -> Optional[str]:
    """只处理字符串化 Literal('N', datatype=...)，提取 'N'；否则 None。"""
    if not isinstance(s, str):
        return None
    inner = s.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()
        if inner.endswith(","):
            inner = inner[:-1].strip()
    m = _LITERAL_WITH_DT_VALUE_RE.search(inner)
    if m:
        return m.group(1)
    return None

def _normalize_for_test_data_2(provided) -> List[Dict[str, str]]:
    """
    把 test_data_2 的 provided gold 粗暴规整为 [{"count": "<数值文本>"}]。
    只识别 Literal('N', datatype=...)；其余留空由外层回退。
    """
    norm: List[Dict[str, str]] = []
    for item in (provided or []):
        val = None
        if isinstance(item, str):
            v = _extract_literal_value_from_str_tuple_minimal(item)
            if v is not None:
                val = v
        elif isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], str):
            v = _extract_literal_value_from_str_tuple_minimal(item[0])
            if v is not None:
                val = v
        if val is not None:
            norm.append({"count": str(val)})
    return norm

def _rows_have_numeric_count(rows: List[Dict[str, Any]]) -> bool:
    for r in rows or []:
        if isinstance(r, dict) and "count" in r and isinstance(r["count"], (str, int, float)):
            s = str(r["count"]).strip()
            if _NUM_RE.match(s):
                return True
    return False

def _build_count_rows_from_exec(ttl: Path, sparql: str) -> List[Dict[str, str]]:
    """执行 answer_sparql，从返回里抓第一处数值，封成 [{'count':'N'}]。"""
    if not sparql:
        return []
    try:
        rows = run_sparql(ttl, sparql).rows
    except Exception:
        return []
    for row in rows or []:
        if isinstance(row, dict):
            vals = list(row.values())
        elif isinstance(row, (list, tuple)):
            vals = list(row)
        else:
            vals = [row]
        for v in vals:
            try:
                s = str(getattr(v, "toPython", lambda: v)())
            except Exception:
                s = str(v)
            s = s.strip()
            if _NUM_RE.match(s):
                return [{"count": s}]
    return []

# ---------------- 主入口：每文件报告 + 合并报告（K 仅由 --limit_test 控制） ----------------
def run_scheme_2(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],          # 保留签名但不直接使用；从 ALL_DATASET_DIR 逐文件读取
    out_dir: Path,
    use_llm: bool,                        # 是否用 LLM：由入参控制
    limit_test: Optional[int] = None,     # 每文件抽样条数 K；未提供则默认 50
    save_every: int = 50,
    filename_suffix: str = "",
    random_test: bool = False,            # True=随机抽样K；False=取前K条
    seed: int = 42,                       # 随机种子（与 --random_test 配合）
):
    """
    - 遍历 ALL_DATASET_DIR 下每个 test_data_*.json；
    - 对每个文件：按随机/顺序抽 K 条样本评测（K=--limit_test；未传则 K=50）；
    - 是否使用 LLM 由 --use_llm 控制；
    - 为每个文件输出 CSV + summary；同时输出合并 CSV + summary。
    """
    # HNSW / few-shot 参数（可用环境变量微调）
    hnsw_backend = os.getenv("HNSW_BACKEND", "faiss_hnsw")
    hnsw_M = int(os.getenv("HNSW_M", "32"))
    hnsw_efC = int(os.getenv("HNSW_EFC", "300"))
    hnsw_efS = int(os.getenv("HNSW_EFS", "256"))
    k_shot = int(os.getenv("HNSW_K_SHOT", "6"))

    # 每文件抽样条数 K：只用 --limit_test 控制；未提供则默认 50
    per_file_k = int(limit_test) if limit_test is not None else 50

    # LLM 句柄（仅当 use_llm=True 时创建）
    llm = LLMClient() if use_llm else None

    # 基于 train 建一次索引（复用到所有测试文件）
    print(f"[Scheme2-HNSW] 建索引：train={len(train)}, backend={hnsw_backend}, M={hnsw_M}, efC={hnsw_efC}, efS={hnsw_efS}, k_shot={k_shot}")
    idx = build_index(
        train,
        backend=hnsw_backend,
        hnsw_params={"M": hnsw_M, "efC": hnsw_efC, "efS": hnsw_efS},
    )

    # 找到所有测试集文件
    base_dir = Path(ALL_DATASET_DIR)
    test_files = sorted(list(base_dir.glob("test_data_*.json")))
    if not test_files:
        print("[Scheme2-HNSW] 未发现任何 test_data_*.json，未输出任何报告。")
        return []

    rng = random.Random(seed)
    combined_rows: List[Dict[str, Any]] = []

    print(f"[Scheme2-HNSW] 每文件抽样 K={per_file_k}；use_llm={use_llm}；random_test={random_test}；seed={seed}")
    print(f"[Scheme2-HNSW] 将按 {len(test_files)} 个测试集分别输出报告…")
    for f in test_files:
        # 读单文件
        try:
            raw_rows = _read_json_or_jsonl(f)
        except Exception:
            print(f"  - {f.name}: 读取失败，跳过")
            continue
        full_set = [_normalize_record(r) for r in raw_rows]
        if not full_set:
            print(f"  - {f.name}: 空文件，跳过")
            continue

        # 选取子集：随机 or 顺序
        k = min(per_file_k, len(full_set))
        if random_test:
            test_subset = rng.sample(full_set, k=k)
        else:
            test_subset = full_set[:k]

        # 逐题评测（该测试文件的单独报告）
        rows: List[Dict[str, Any]] = []
        pbar = tqdm(total=len(test_subset), desc=f"Predicting (Scheme2-HNSW | {f.stem} | n={k})")
        for i, ex in enumerate(test_subset, 1):
            q = ex.get("question") or ex.get("input") or ""
            ttl = _ttl_path_from(ex.get("brick_model", ""))

            # few-shot 生成（是否用 LLM 由 use_llm 控制）
            pred = fewshot_generate(
                idx, q,
                mode=("llm_generate" if use_llm else "copy_nearest"),
                llm=llm,
                brick_model=str(ttl.name),
                k=k_shot,
            )

            parse_error = False
            try:
                res = run_sparql(ttl, pred).rows if pred else []
            except Exception:
                parse_error = True
                res, pred = [], f"--PARSE-ERROR--\n{pred}"

            # —— gold 获取（常规）
            goldq_res, goldq_source, goldq_err = _gold_rows(ttl, ex)

            # ✅ 仅对 test_data_2：把 provided gold 粗暴规整为 [{"count":"N"}]；若失败，回退执行 answer_sparql
            if f.stem == "test_data_2":
                provided = ex.get("gold_rows") or ex.get("result") or []
                if provided:
                    forced = _normalize_for_test_data_2(provided)
                    if _rows_have_numeric_count(forced):
                        goldq_res = forced
                        goldq_source, goldq_err = "provided_forced_count", False
                    else:
                        forced_exec = _build_count_rows_from_exec(ttl, ex.get("answer_sparql") or "")
                        if _rows_have_numeric_count(forced_exec):
                            goldq_res = forced_exec
                            goldq_source, goldq_err = "executed_fallback_count", False

            # —— 指标（保持原样；评分口径在 report_utils 里统一处理）
            met_label = metrics_set_based(res, ex.get("expected"))
            met_goldq = metrics_against_rows(res, goldq_res) if goldq_res else {}

            row = {
                "id": i,
                "scheme": "fewshot_hnsw",
                "dataset_tag": f.stem,            # 标注来自哪个测试文件
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
            row.update(met_label)
            row.update(met_goldq)
            rows.append(row)

            if i % save_every == 0:
                save_report(rows, out_dir / f"scheme2_hnsw{filename_suffix}__{f.stem}_partial.csv")
            pbar.update(1)
        pbar.close()

        # —— 单文件落盘
        base = f"scheme2_hnsw{filename_suffix}__{f.stem}"
        save_report(rows, out_dir / f"{base}.csv")
        summary = compute_summary(rows, "fewshot_hnsw")
        save_summary_json(summary, out_dir / f"{base}_summary.json")

        print(f"\n=== JSON Summary (Scheme 2: HNSW | {f.stem}) ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        # 累加到合并行
        combined_rows.extend(rows)

    # —— 合并报告
    base_combined = f"scheme2_hnsw{filename_suffix}"
    save_report(combined_rows, out_dir / f"{base_combined}.csv")
    summary_combined = compute_summary(combined_rows, "fewshot_hnsw")
    save_summary_json(summary_combined, out_dir / f"{base_combined}_summary.json")

    print(f"\n=== JSON Summary (Scheme 2: HNSW | COMBINED over per-file K={per_file_k}) ===")
    print(json.dumps(summary_combined, ensure_ascii=False, indent=2))

    return combined_rows
