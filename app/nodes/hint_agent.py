# hint_agent.py
from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# ============== 配置和初始化 ==============
ABS_PROMPT = Path(r"F:\Task\RAG-LangGraph-Demo-bcp\prompt\prompt.md")
REL_PROMPT = Path(__file__).resolve().parents[2] / "prompt" / "prompt.md"
_prompt_cache: Optional[str] = None
_LLM = None

def _load_prompt() -> Optional[str]:
    """按绝对/相对路径加载提示词模板并缓存。"""
    global _prompt_cache
    if _prompt_cache is not None:
        return _prompt_cache
    for path in (ABS_PROMPT, REL_PROMPT):
        if path.exists():
            _prompt_cache = path.read_text(encoding="utf-8")
            return _prompt_cache
    return None

def _get_llm():
    """初始化并缓存 LLM（DeepSeek，经 LangChain init_chat_model）。"""
    global _LLM
    if _LLM is not None:
        return _LLM
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    try:
        from langchain.chat_models import init_chat_model
        _LLM = init_chat_model("deepseek:deepseek-chat", temperature=0, api_key=api_key)
        return _LLM
    except Exception:
        return None

# ============== 工具函数 ==============
def _safe_json_from_text(s: str) -> Optional[Dict]:
    """从 LLM 输出中“宽松地”抽取 JSON 对象（容忍多余文本/代码块）。"""
    if not s:
        return None
    m = re.search(r"\{.*\}", s, flags=re.S)
    txt = m.group(0) if m else s
    try:
        val = json.loads(txt)
        return val if isinstance(val, dict) else None
    except Exception:
        return None

def _neutral(question: str, source_reason: str) -> Dict:
    """无 prompt/无 llm/解析失败时的安全回退。"""
    return {
        "question_type": "other", "topology_intent": None, "need_stats": False,
        "need": None, "room": None, "metric": None, "time_range": None,
        "uncertain": True, "ambiguities": [], "_source": source_reason,
        "_prompt": str(ABS_PROMPT if ABS_PROMPT.exists() else REL_PROMPT),
    }

# ============== 语义解析核心 ==============
ALLOWED_NEEDS = {"avg", "max", "min", "trend"}
ALLOWED_QTYPE = {"timeseries", "topology", "other"}
ALLOWED_TOPO_INTENT = {"count_rooms", "list_rooms", "sensor_existence"}

def _normalize_need(x: Any) -> Optional[List[str]]:
    if not x:
        return None
    if isinstance(x, str):
        x = [x]
    if not isinstance(x, list):
        return None
    return [n.strip().lower() for n in x
            if isinstance(n, str) and n.strip().lower() in ALLOWED_NEEDS]

def parse(question: str) -> Dict:
    """主解析：返回结构化 hints（供 L0/L1/L2/L3 使用）。"""
    prompt = _load_prompt()
    if not prompt:
        return _neutral(question, "no-prompt")

    llm = _get_llm()
    if llm is None:
        return _neutral(question, "no-llm")

    try:
        current_time = datetime.now().strftime("%Y年%m月%d日")
        dynamic_prompt = f"**重要：当前系统时间是 {current_time}。**\n\n{prompt}"
        full_prompt = f"{dynamic_prompt}\n\n用户问题：{question or ''}"

        try:
            resp = llm.invoke(full_prompt)
            text = getattr(resp, "content", None) or (resp if isinstance(resp, str) else "")
        except Exception:
            text = llm.invoke(full_prompt)

        data = _safe_json_from_text(text)
        if not data:
            return _neutral(question, "parse-error")

        # 字段清洗/校正
        qtype = str(data.get("question_type", "other")).lower()
        qtype = qtype if qtype in ALLOWED_QTYPE else "other"

        topo_intent = data.get("topology_intent")
        if isinstance(topo_intent, str):
            topo_intent = topo_intent.lower().strip()
            topo_intent = topo_intent if topo_intent in ALLOWED_TOPO_INTENT else None

        need = _normalize_need(data.get("need"))
        need_stats = bool(data.get("need_stats")) or bool(need)

        room = data.get("room")
        if isinstance(room, str):
            mm = re.search(r"(?<!\d)(\d{1,4})(?!\d)", room)
            room = mm.group(1) if mm else None

        metric = data.get("metric")
        metric_allow = ("temp", "rh", "lux", "co2", "pm25")
        metric = metric if metric in metric_allow else None

        time_range = data.get("time_range")
        if not isinstance(time_range, dict) or "kind" not in time_range:
            time_range = None

        uncertain = bool(data.get("uncertain", False))
        ambiguities = data.get("ambiguities") or []
        if not isinstance(ambiguities, list):
            ambiguities = [str(ambiguities)]

        return {
            "question_type": qtype, "topology_intent": topo_intent, "need_stats": need_stats,
            "need": need, "room": room, "metric": metric, "time_range": time_range,
            "uncertain": uncertain, "ambiguities": ambiguities, "_source": "llm",
            "_prompt": str(ABS_PROMPT if ABS_PROMPT.exists() else REL_PROMPT),
        }
    except Exception:
        return _neutral(question, "llm-error")

# ============== Graph 暴露接口（保持兼容原 API） ==============
def get_hints(question: str) -> Dict:
    return parse(question)

def need_stats(question: str) -> bool:
    try:
        return bool(parse(question).get("need_stats"))
    except Exception:
        return False
