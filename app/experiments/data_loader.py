from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json, re
from .config import ALL_DATASET_DIR

def _read_json_or_jsonl(p: Path) -> List[Dict[str, Any]]:
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # JSON 数组
    if text.lstrip().startswith("["):
        try:
            arr = json.loads(text)
            return arr if isinstance(arr, list) else []
        except Exception:
            pass
    # JSONL
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows

# 解析 train/test 里的 "result" 为行列表：[{"result": "..."}]
_URIREF_RE = re.compile(r"URIRef\('([^']+)'\)")
_TUP_STR_RE = re.compile(r"\('([^']+)'\)")
_QUOTED1_RE = re.compile(r"'([^']+)'")
_QUOTED2_RE = re.compile(r"\"([^\"]+)\"")

def _coerce_gold_rows_from_result(result_field: Any) -> List[Dict[str, Any]]:
    if not result_field:
        return []
    rows = []
    def _extract_one(s: str) -> str:
        for pat in (_URIREF_RE, _TUP_STR_RE, _QUOTED2_RE, _QUOTED1_RE):
            m = pat.search(s)
            if m:
                return m.group(1)
        return str(s)
    if isinstance(result_field, list):
        for it in result_field:
            if isinstance(it, dict) and "result" in it:
                rows.append({"result": str(it["result"])})
            else:
                rows.append({"result": _extract_one(str(it))})
    elif isinstance(result_field, str):
        rows.append({"result": _extract_one(result_field)})
    return rows

def _normalize_record(r: Dict[str, Any]) -> Dict[str, Any]:
    # 统一键：question / answer_sparql / brick_model / expected(list[str]) / gold_rows(list[dict])
    q = r.get("question") or r.get("input") or r.get("query") or ""
    sp = r.get("answer_sparql") or r.get("output") or r.get("sparql") or ""
    b = r.get("brick_model") or r.get("building_model") or r.get("ttl") or ""
    gold_rows = []
    expected = r.get("expected")

    # 如果数据集中提供了 result，则直接作为金标行
    if r.get("result") is not None:
        gold_rows = _coerce_gold_rows_from_result(r.get("result"))
        # 兼容旧评测口径：提供 expected = list[str]
        expected = [row["result"] for row in gold_rows]

    # 若 expected 是 {"set":[...]} 也转成 list[str]
    if isinstance(expected, dict) and "set" in expected:
        expected = [str(x) for x in expected.get("set", [])]
    elif isinstance(expected, list):
        expected = [str(x) for x in expected]
    elif expected is None:
        expected = None
    else:
        expected = [str(expected)]

    return {
        "question": q,
        "answer_sparql": sp,
        "brick_model": b,
        "expected": expected,
        "gold_rows": gold_rows,     # 若数据里有 result，这里就是解析后的行
        "_raw": r,                  # 保留原始字段便于排查
    }

def load_dataset(prefix: str) -> List[Dict[str, Any]]:
    base = Path(ALL_DATASET_DIR)
    files = sorted(list(base.glob(f"{prefix}*.json")))
    out: List[Dict[str, Any]] = []
    for f in files:
        for r in _read_json_or_jsonl(f):
            out.append(_normalize_record(r))
    return out
