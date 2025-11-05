from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from rdflib import Graph
from pathlib import Path
import re
import json

@dataclass
class QueryResult:
    rows: List[Dict[str, Any]]

def run_sparql(ttl_path: Path, query: str) -> QueryResult:
    g = Graph()
    g.parse(str(ttl_path))  # rdflib 自动识别 ttl
    res = g.query(query)
    rows = []
    for r in res:
        row = {}
        for k, v in r.asdict().items():
            row[str(k)] = str(v)
        rows.append(row)
    return QueryResult(rows=rows)

# —— 集合提取与归一化 —— #

# 优先 ?result 列（你的数据是按 ?result 变量导出的）
PREF_COLS = ["result", "room", "x", "s", "subject", "entity", "pt", "node", "name", "label"]

def to_setlike(rows: List[Dict[str, Any]]) -> Tuple[str, Set[str]]:
    """将查询结果转为集合：优先 ?result / ?room / 第一列"""
    if not rows:
        return ("", set())
    keys = list(rows[0].keys())
    preferred = [k for k in PREF_COLS if k in keys]
    col = preferred[0] if preferred else keys[0]
    return (col, {str(r[col]).strip() for r in rows if col in r})

def rows_preview(rows: List[Dict[str, Any]], limit: int = 10) -> str:
    """把前 limit 行转为 JSON 字符串（便于塞进 CSV）"""
    return json.dumps(rows[:limit], ensure_ascii=False)

def _norm_raw(s: str) -> str:
    return s.strip().lower()

def _norm_localname(s: str) -> str:
    """URI 或 前缀名 -> 取局部名；普通字符串原样"""
    if ":" in s or "/" in s or "#" in s:
        tail = re.split(r"[#/:]", s)[-1]
        return tail.strip().lower()
    return s.strip().lower()

def _norm_simplify_token(s: str) -> str:
    """
    强化简化：去掉 room/space 等通用词；统一下划线/连字符/空格；移除非字母数字
    """
    x = _norm_localname(s)
    x = re.sub(r"\b(room|space|rm|area)\b", "", x)
    x = re.sub(r"[_\-\s]+", "", x)
    x = re.sub(r"[^a-z0-9]", "", x)
    return x

def _to_norm_sets(vals: Set[str]) -> Dict[str, Set[str]]:
    return {
        "raw": {_norm_raw(v) for v in vals},
        "local": {_norm_localname(v) for v in vals},
        "simple": {_norm_simplify_token(v) for v in vals},
    }

def _f1(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float, int]:
    inter = pred & gold
    if not pred and not gold:
        return (1.0, 1.0, 1.0, 0)
    if not pred or not gold:
        return (0.0, 0.0, 0.0, 0)
    prec = len(inter)/len(pred)
    rec = len(inter)/len(gold)
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return (prec, rec, f1, len(inter))

# —— 指标计算 —— #

def metrics_set_based(pred_rows: List[Dict[str, Any]], expected: Any) -> Dict[str, Any]:
    """
    expected 支持：list[str] / {"count": int} / {"set": [...]}
    返回同时包含：
      - 严格字符串指标：exact_set_match / f1
      - 局部名指标：local_f1
      - 简化 token 指标：simple_f1
      - URI 维度：uri_exact_set_match / uri_overlap（仅当任一侧像 URI 时）
      - 尺寸关系：pred_size / gold_size / size_eq
    """
    col, pred_set_raw = to_setlike(pred_rows)

    # 计数题
    if isinstance(expected, dict) and "count" in expected:
        acc = 1.0 if len(pred_set_raw) == int(expected["count"]) else 0.0
        return {
            "mode": "count",
            "pred_size": len(pred_set_raw),
            "gold_size": int(expected["count"]),
            "acc": acc
        }

    # 解析 gold 集合
    if isinstance(expected, dict) and "set" in expected:
        gold_vals = {str(x).strip() for x in expected["set"]}
    elif isinstance(expected, list):
        gold_vals = {str(x).strip() for x in expected}
    else:
        return {"mode": "unknown", "pred_size": len(pred_set_raw), "gold_size": None}

    # 多重归一化
    pred_sets = _to_norm_sets(pred_set_raw)
    gold_sets = _to_norm_sets(gold_vals)

    # —— 严格字符串
    p_strict, r_strict, f1_strict, _ = _f1(pred_sets["raw"], gold_sets["raw"])
    exact = 1.0 if pred_sets["raw"] == gold_sets["raw"] else 0.0

    # —— 局部名（URI 尾部等价）
    p_local, r_local, f1_local, _ = _f1(pred_sets["local"], gold_sets["local"])

    # —— 简化 token
    p_simple, r_simple, f1_simple, _ = _f1(pred_sets["simple"], gold_sets["simple"])

    # —— 纯 URI 维度
    def looks_uri(x: str) -> bool:
        return (":" in x) or ("/" in x) or ("#" in x)

    pred_uri = {v for v in pred_sets["raw"] if looks_uri(v)}
    gold_uri = {v for v in gold_sets["raw"] if looks_uri(v)}
    uri_exact = None
    uri_overlap = None
    if pred_uri or gold_uri:
        uri_exact   = 1.0 if (pred_uri == gold_uri and pred_uri) else 0.0 if (pred_uri or gold_uri) else None
        uri_overlap = 1.0 if (pred_uri & gold_uri) else 0.0

    size_eq = 1.0 if len(pred_set_raw) == len(gold_vals) else 0.0
    any_overlap = 1.0 if (pred_sets["local"] & gold_sets["local"]) else 0.0

    return {
        "mode": "set",
        "pred_size": len(pred_set_raw),
        "gold_size": len(gold_vals),
        # 严格
        "exact_set_match": exact,
        "prec": p_strict, "recall": r_strict, "f1": f1_strict,
        # 局部名 / 简化
        "local_prec": p_local, "local_recall": r_local, "local_f1": f1_local,
        "simple_prec": p_simple, "simple_recall": r_simple, "simple_f1": f1_simple,
        # URI 维度
        "uri_exact_set_match": uri_exact,
        "uri_overlap": uri_overlap,
        # 规模/重合
        "size_eq": size_eq,
        "any_overlap": any_overlap,
        # 供摘要选最优
        "best_f1": max(f1_strict, f1_local, f1_simple),
    }

def metrics_against_rows(pred_rows: List[Dict[str, Any]], gold_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    直接把“预测结果 vs gold 查询的结果”做集合对齐评测。
    返回键名已带 vs_goldq_ 前缀，方便直接 merge 到 CSV 行里。
    """
    # 从 gold_rows 里自动抽列 -> 集合（内部会优先用 ?result）
    _, gold_set = to_setlike(gold_rows)
    met = metrics_set_based(pred_rows, list(gold_set))
    # 改键名
    out = {}
    for k, v in met.items():
        out[f"vs_goldq_{k}"] = v
    return out
