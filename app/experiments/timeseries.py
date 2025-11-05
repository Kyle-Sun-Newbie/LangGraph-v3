# app/experiments/timeseries_all.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import json
import math
from statistics import mean

import pandas as pd
import numpy as np

# === 解析 TTL：对象名 <-> site_id ===================================================
from rdflib import Graph, URIRef, BNode, Literal, Namespace
from rdflib.namespace import RDFS

BRICK = Namespace("https://brickschema.org/schema/Brick#")

def _localname(iri: URIRef) -> str:
    s = str(iri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rstrip("/").rsplit("/", 1)[-1]
    return s

def _norm(s: str) -> str:
    return "".join(ch for ch in s.strip().lower() if ch.isalnum())

@dataclass
class SiteResolver:
    """对象名 <-> 站点编号（site_id）解析器"""
    name2id: Dict[str, str]
    id2name: Dict[str, str]

    def resolve_id(self, name: str) -> Optional[str]:
        if name is None:
            return None
        return self.name2id.get(_norm(name))

    def best_name(self, site_id: str) -> Optional[str]:
        if site_id is None:
            return None
        return self.id2name.get(str(site_id))

def build_site_resolver(ttl_path: Path) -> SiteResolver:
    """
    在图中寻找每个主体 s 满足：
      s (p=任何前缀:siteID) o .
      o brick:value idLiteral .
    其中 s 的展示名优先用 rdfs:label，否则用 s 的本地名。
    """
    g = Graph()
    g.parse(ttl_path.as_posix(), format="turtle")

    name2id: Dict[str, str] = {}
    id2name: Dict[str, str] = {}

    # 遍历所有 (s,p,o)，挑选本地名为 siteID 的谓词
    for s, p, o in g.triples((None, None, None)):
        if not isinstance(p, URIRef):
            continue
        if _localname(p) != "siteID":
            continue
        if not isinstance(o, (BNode, URIRef)):
            continue

        # 取 brick:value
        site_id = None
        for _, _, v in g.triples((o, BRICK.value, None)):
            if isinstance(v, Literal):
                site_id = str(v)
                break
        if not site_id:
            continue

        # 展示名
        display_name = None
        for _, _, lab in g.triples((s, RDFS.label, None)):
            if isinstance(lab, Literal):
                display_name = str(lab)
                break
        if not display_name:
            display_name = _localname(s)

        id2name.setdefault(site_id, display_name)
        name2id.setdefault(_norm(display_name), site_id)
        name2id.setdefault(_norm(_localname(s)), site_id)

    return SiteResolver(name2id=name2id, id2name=id2name)

# === 加载器：dataset_xxxx.json -> TSCase ==========================================

# 业务字段到列名
DATATYPE_COLUMN = {
    "efficiency": "Efficiency(%)",
    "energy": "Energy(Wh)",
}

# 统计函数名映射（支持 cal_* 前缀与常见别名）
STAT_NAME_MAP = {
    # 基础统计
    "median": "median",
    "cal_median": "median",
    "mean": "mean",
    "average": "mean",
    "avg": "mean",
    "cal_mean": "mean",
    "cal_average": "mean",
    "sum": "sum",
    "cal_sum": "sum",
    "min": "min",
    "cal_min": "min",
    "max": "max",
    "cal_max": "max",
    # 极差（range = max - min）
    "range": "range",
    "cal_range": "range",
    # 百分位
    "p95": "p95",
    "pct95": "p95",
    "quantile95": "p95",
    "cal_p95": "p95",
    "p90": "p90",
    "pct90": "p90",
    "quantile90": "p90",
    "cal_p90": "p90",
}

@dataclass
class TSCase:
    """标准化后的时序评测样本"""
    case_id: int
    question: str
    params: Dict[str, Any]
    series_by_site: Dict[str, pd.DataFrame]
    column: str                 # e.g. "Efficiency(%)" / "Energy(Wh)"
    stat_func: str              # e.g. "median"/"mean"/"sum"/"min"/"max"/"range"/"p95"/"p90"
    compare_mode: str           # "lowest" / "highest" / ...
    groundtruth: Dict[str, Any]
    requested_site_names: List[str]
    resolved_site_ids: List[str]
    unresolved_site_names: List[str]

def _coerce_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "Time" in df.columns:
        try:
            df["Time"] = pd.to_datetime(df["Time"])
        except Exception:
            pass
    return df

def _detect_column(datatype: str) -> str:
    key = (datatype or "").strip().lower()
    col = DATATYPE_COLUMN.get(key)
    if not col:
        raise ValueError(
            f"Unsupported datatype={datatype!r}. "
            f"Please extend DATATYPE_COLUMN."
        )
    return col

def _normalize_statistic_name(statistic: str) -> str:
    key = (statistic or "").strip().lower()
    if key.startswith("cal_"):
        key = key[4:]
    return key

def _detect_stat(statistic: str) -> str:
    raw = (statistic or "").strip().lower()
    if raw in STAT_NAME_MAP:
        return STAT_NAME_MAP[raw]
    key = _normalize_statistic_name(raw)
    if key in STAT_NAME_MAP:
        return STAT_NAME_MAP[key]
    # 百分位通配
    if key.startswith("p") and key[1:].isdigit():
        return f"p{key[1:]}"
    if key.startswith("pct") and key[3:].isdigit():
        return f"p{key[3:]}"
    if key.startswith("quantile") and key[8:].isdigit():
        return f"p{key[8:]}"
    raise ValueError(
        f"Unsupported statistic={statistic!r}. "
        f"Add a mapping in STAT_NAME_MAP or handle it in _detect_stat()."
    )

def load_timeseries_dataset(json_path: Path,
                            resolver: Optional[SiteResolver] = None) -> List[TSCase]:
    """
    读取 dataset_xxxx.json，转为 TSCase 列表；若提供 resolver 则解析 parameter.siteName -> site_id
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cases: List[TSCase] = []

    for i, ex in enumerate(data):
        params = ex.get("parameter", {}) or {}
        question = ex.get("question", "") or ""
        series_by_site: Dict[str, pd.DataFrame] = {
            str(site_id): _coerce_dataframe(rows)
            for site_id, rows in (ex.get("query_func") or {}).items()
        }
        column = _detect_column(params.get("datatype", ""))
        stat_func = _detect_stat(params.get("statistic", "median"))
        compare_mode = (params.get("comparison") or "").strip().lower()
        gt = ex.get("groundtruth") or {}

        # 解析对象名 -> site_id（可选）
        requested_names = [str(x) for x in (params.get("siteName") or [])]
        resolved_ids: List[str] = []
        unresolved_names: List[str] = []
        if resolver and requested_names:
            for nm in requested_names:
                sid = resolver.resolve_id(nm)
                (resolved_ids.append(str(sid)) if sid else unresolved_names.append(nm))

        cases.append(TSCase(
            case_id=i,
            question=question,
            params=params,
            series_by_site=series_by_site,
            column=column,
            stat_func=stat_func,
            compare_mode=compare_mode,
            groundtruth=gt,
            requested_site_names=requested_names,
            resolved_site_ids=resolved_ids,
            unresolved_site_names=unresolved_names,
        ))
    return cases

# === 评测：统计/比较/答案文本/汇总 ================================================

DEFAULT_EPS = 1e-6

def _compute_stat_series(df: pd.DataFrame, column: str, stat: str) -> float:
    if column not in df.columns:
        return float("nan")
    s = df[column].dropna()
    if s.empty:
        return float("nan")

    stat = stat.lower()
    if stat == "median": return float(s.median())
    if stat == "mean":   return float(s.mean())
    if stat == "sum":    return float(s.sum())
    if stat == "min":    return float(s.min())
    if stat == "max":    return float(s.max())
    # 极差：max - min
    if stat == "range":  return float(s.max() - s.min())
    # 百分位
    if stat in ("p95", "pct95", "quantile95"): return float(s.quantile(0.95))
    if stat in ("p90", "pct90", "quantile90"): return float(s.quantile(0.90))

    raise ValueError(f"Unsupported stat={stat!r}")

def compute_stats_for_sites(series_by_site: Dict[str, pd.DataFrame],
                            column: str,
                            stat: str) -> Dict[str, float]:
    """返回 {site_id: 统计值}，自动跳过全 NaN"""
    out: Dict[str, float] = {}
    for sid, df in series_by_site.items():
        v = _compute_stat_series(df, column, stat)
        if not (isinstance(v, float) and math.isnan(v)):
            out[sid] = v
    return out

def choose_by_comparison(stats: Dict[str, float], mode: str) -> Tuple[str, float]:
    """根据比较模式选出站点（返回 site_id, value）"""
    if not stats:
        return "", float("nan")
    mode = (mode or "").lower()
    items = list(stats.items())
    if mode in ("lowest", "min", "minimum"):
        sid, val = min(items, key=lambda kv: kv[1])
    else:
        sid, val = max(items, key=lambda kv: kv[1])
    return sid, float(val)

def build_answer_text(site_id: str, value: float, site_name: str | None = None) -> str:
    name = site_name or site_id
    try:
        vf = float(value)
    except Exception:
        vf = value
    return (
        f"Site: {name} | Value: {vf:.3f}"
        if isinstance(vf, float) and not math.isnan(vf)
        else f"Site: {name} | Value: NaN"
    )

def summarize_rows(rows: List[Dict[str, Any]], eps: float = DEFAULT_EPS) -> Dict[str, Any]:
    """计算汇总指标"""
    if not rows:
        return {
            "n": 0,
            "site_accuracy": None,
            "value_mae": None,
            "within_eps_ratio": None,
            "overall_accuracy_strict": None,
            "overall_accuracy_relaxed": None,
        }

    site_acc_vals = [
        1.0 if r.get("site_correct") else 0.0
        for r in rows
        if r.get("site_correct") is not None
    ]
    mae_vals = [abs(r.get("abs_err")) for r in rows if r.get("abs_err") is not None]
    within_vals = [
        1.0 if (r.get("abs_err") is not None and abs(r["abs_err"]) <= eps) else 0.0
        for r in rows
    ]

    def _avg(x): return round(mean(x), 6) if x else None

    strict_flags = [
        1.0 if (
            (r.get("site_correct") is True) and
            (r.get("abs_err") is not None) and
            (abs(r["abs_err"]) <= eps)
        ) else 0.0
        for r in rows
    ]
    relaxed_flags = [1.0 if (r.get("site_correct") is True) else 0.0 for r in rows]

    return {
        "n": len(rows),
        "site_accuracy": _avg(site_acc_vals),
        "value_mae": _avg(mae_vals),
        "within_eps_ratio": _avg(within_vals),
        "overall_accuracy_strict": _avg(strict_flags),
        "overall_accuracy_relaxed": _avg(relaxed_flags),
        "eps": eps,
    }

__all__ = [
    # 解析
    "SiteResolver", "build_site_resolver",
    # 加载
    "TSCase", "load_timeseries_dataset",
    # 评测
    "DEFAULT_EPS", "compute_stats_for_sites", "choose_by_comparison",
    "build_answer_text", "summarize_rows",
]
