# app/experiments/report_utils.py
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import json
from statistics import mean

def save_report(rows: List[Dict[str, Any]], to_path: Path):
    df = pd.DataFrame(rows)
    to_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(to_path, index=False, encoding="utf-8-sig")
    return to_path

def _avg(vals):
    vals = [v for v in vals if v is not None]
    return round(mean(vals), 4) if vals else None

def _jsonable(obj) -> str:
    """稳定序列化用于比较（避免 key 顺序/空格差异）。"""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)

def _apply_test_data_2_override(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    仅对 test_data_2 的样本：用 pred_rows_preview 与 goldq_rows_preview 是否一致来打分。
    - 将该行视作 count 题：r['mode'] = 'count'
    - 设置 r['acc'] = 1.0 (一致) / 0.0 (不一致)
    其它数据集不做改动。
    """
    patched = []
    for r in rows:
        if r.get("dataset_tag") == "test_data_2":
            pred_pv = r.get("pred_rows_preview")
            gold_pv = r.get("goldq_rows_preview")
            eq = _jsonable(pred_pv) == _jsonable(gold_pv)
            rr = dict(r)  # 复制一份避免副作用
            rr["mode"] = "count"
            rr["acc"] = 1.0 if eq else 0.0
            # 清空“集合题”用的字段，避免干扰
            rr["exact_set_match"] = None
            rr["f1"] = None
            rr["local_f1"] = None
            rr["simple_f1"] = None
            rr["best_f1"] = None
            rr["size_eq"] = None
            rr["any_overlap"] = None
            rr["uri_overlap"] = None
            rr["uri_exact_set_match"] = None
            patched.append(rr)
        else:
            patched.append(r)
    return patched

def compute_summary(rows: List[Dict[str, Any]], scheme_name: str) -> Dict[str, Any]:
    # 先做 test_data_2 的特判重写（其余数据集保持原状）
    rows = _apply_test_data_2_override(rows)

    total = len(rows)
    parse_err = sum(1 for r in rows if r.get("parse_error"))
    valid = total - parse_err
    nonempty = sum(1 for r in rows if (r.get("n_rows", 0) > 0))

    set_rows = [r for r in rows if r.get("mode") == "set" and not r.get("parse_error")]
    cnt_rows = [r for r in rows if r.get("mode") == "count" and not r.get("parse_error")]

    # —— 严格 & 派生（集合题）
    strict_em   = _avg([1.0 if r.get("exact_set_match", 0.0) == 1.0 else 0.0 for r in set_rows])
    strict_f1   = _avg([r.get("f1", 0.0) for r in set_rows])
    local_f1    = _avg([r.get("local_f1", 0.0) for r in set_rows])
    simple_f1   = _avg([r.get("simple_f1", 0.0) for r in set_rows])
    best_f1     = _avg([r.get("best_f1", 0.0) for r in set_rows])

    size_match_rate = _avg([r.get("size_eq", 0.0) for r in set_rows])
    overlap_rate    = _avg([r.get("any_overlap", 0.0) for r in set_rows])

    # —— URI 维度（只统计 r['uri_*'] 有值的那些题）
    uri_rows = [r for r in set_rows if r.get("uri_overlap") is not None]
    uri_overlap_rate = _avg([r.get("uri_overlap", 0.0) for r in uri_rows])
    uri_exact_rate   = _avg([r.get("uri_exact_set_match", 0.0) for r in uri_rows])

    # —— 计数题
    count_acc = _avg([1.0 if r.get("acc", 0.0) == 1.0 else 0.0 for r in cnt_rows])

    # —— 宽松成功判定（收紧！）
    # 原来：best_f1>=0.5 OR size_eq==1
    # 现在：best_f1>=0.5 OR (size_eq==1 AND any_overlap==1)
    relaxed_success_terms = []
    for r in set_rows:
        ok = (r.get("best_f1", 0.0) >= 0.5) or (r.get("size_eq", 0.0) == 1.0 and r.get("any_overlap", 0.0) == 1.0)
        relaxed_success_terms.append(1.0 if ok else 0.0)
    for r in cnt_rows:
        relaxed_success_terms.append(1.0 if r.get("acc", 0.0) == 1.0 else 0.0)
    relaxed_acc = _avg(relaxed_success_terms)

    # —— 严格 overall（保持原口径）
    acc_terms = []
    if set_rows:
        acc_terms.extend([1.0 if r.get("exact_set_match", 0.0) == 1.0 else 0.0 for r in set_rows])
    if cnt_rows:
        acc_terms.extend([1.0 if r.get("acc", 0.0) == 1.0 else 0.0 for r in cnt_rows])
    overall_strict = _avg(acc_terms)

    return {
        "scheme": scheme_name,
        "num_samples": total,
        "num_valid": valid,
        "parse_error_rate": round(parse_err / total, 4) if total else None,
        "nonempty_result_rate": round(nonempty / total, 4) if total else None,
        "metrics": {
            "set": {
                "n": len(set_rows),
                "strict_macro_f1": strict_f1,
                "exact_set_match_rate": strict_em,
                "local_macro_f1": local_f1,
                "simple_macro_f1": simple_f1,
                "best_macro_f1": best_f1,
                "size_match_rate": size_match_rate,
                "overlap_rate": overlap_rate,
                "uri_rows": len(uri_rows),
                "uri_overlap_rate": uri_overlap_rate,
                "uri_exact_match_rate": uri_exact_rate,
            },
            "count": {
                "n": len(cnt_rows),
                "accuracy": count_acc,
            },
        },
        "overall_accuracy_strict": overall_strict,
        "overall_accuracy_relaxed": relaxed_acc,
    }

def save_summary_json(summary: Dict[str, Any], to_path: Path) -> Path:
    to_path.parent.mkdir(parents=True, exist_ok=True)
    to_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return to_path
