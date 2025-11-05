# app/experiments/schemes/langgraph_like.py
"""
LangGraph-like schemes for the test project (topology-only; no timeseries analysis).

三种模式（剔除时序分析，只做拓扑类 SPARQL）：
  - mode="template": 模板直出 SPARQL（无回退）                -> Scheme 3
  - mode="llm":      LLM 直接生成（等价 Level-1 回退路径）     -> Scheme 4
  - mode="rag_llm":  RAG(从TTL抽取片段)+LLM（等价 Level-2）    -> Scheme 5
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# ---------- 延迟导入：使用包内相对导入 ----------
def _lazy_imports():
    # 你的 config 和 llm 都在 app.experiments 包下
    from ..config import EMBED_MODEL
    from ..llm import LLMClient
    from ..sparql_eval import run_sparql
    return EMBED_MODEL, LLMClient, run_sparql


PREFIXES = """PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX bf:    <https://brickschema.org/schema/BrickFrame#>
PREFIX tag:  <https://brickschema.org/schema/BrickTag#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX sh:   <http://www.w3.org/ns/shacl#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""

def _normalize_ws(s: str) -> str:
    s = s.strip()
    return re.sub(r"\s+", " ", s)

# -----------------------------
# 模板：仅拓扑类（常见问法覆盖）
# -----------------------------
def template_topology(question: str) -> Optional[str]:
    q = question.lower().strip()

    # 1) 房间数量
    if any(k in q for k in ["多少个房间", "房间数", "房间数量", "how many rooms", "number of rooms"]):
        return f"""{PREFIXES}
SELECT (COUNT(DISTINCT ?room) AS ?count)
WHERE {{
  ?room rdf:type brick:Room .
}}"""

    # 2) 列出房间
    if any(k in q for k in ["列出房间", "有哪些房间", "所有房间", "list rooms"]):
        return f"""{PREFIXES}
SELECT DISTINCT ?room
WHERE {{
  ?room rdf:type brick:Room .
}}
ORDER BY ?room"""

    # 3) 列出某类 Brick 实例（Temperature_Sensor / *Point / *Equipment 等）
    m = re.search(r"(?:brick:)?([A-Za-z_]+_Sensor|[A-Za-z_]+_Point|[A-Za-z_]+_Equipment)", q)
    if m:
        brick_class = m.group(1)
        if not brick_class.startswith("brick:"):
            brick_class = "brick:" + brick_class
        return f"""{PREFIXES}
SELECT DISTINCT ?x
WHERE {{
  ?x rdf:type {brick_class} .
}}
ORDER BY ?x"""

    # 4) 某房间内的点位/传感器
    m2 = re.search(r"(?:房间|room)\s*([0-9A-Za-z_-]+)", q)
    if m2 and any(k in q for k in ["sensor", "传感器", "point", "点位"]):
        room = m2.group(1)
        return f"""{PREFIXES}
SELECT DISTINCT ?pt ?pt_type
WHERE {{
  ?room rdf:type brick:Room .
  FILTER(CONTAINS(STR(?room), "{room}"))
  ?pt bf:isPointOf ?room .
  ?pt rdf:type ?pt_type .
}}
ORDER BY ?pt"""

    return None

# -----------------------------
# LLM 生成器（改为使用 LLMClient）
# -----------------------------
_SYS_GEN = """You translate a natural-language question about a Brick building model into a single valid SPARQL query.
- Only output the SPARQL; no prose and no code fences.
- Always include the required Brick/BrickFrame prefixes provided below.
- Prefer simple SELECT queries.
""" + PREFIXES

def _llm_generate_sparql(question: str, context: Optional[str] = None) -> str:
    _, LLMClient, _ = _lazy_imports()
    llm = LLMClient()  # 统一走你的 LLM 后端选择逻辑
    user = "Question:\n" + question.strip()
    if context:
        user += "\n\nContext (TTL facts):\n" + context.strip()[:4000]

    # LLMClient.chat 接受 messages 列表，这里把 system 放第一条
    text = llm.chat([
        {"role": "system", "content": _SYS_GEN},
        {"role": "user", "content": user}
    ]).strip()

    # 清理代码块围栏，并补 PREFIX
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    if "PREFIX brick:" not in text:
        text = PREFIXES + "\n" + text
    return text

# -----------------------------
# RAG（从 TTL 文本抽片段）
# -----------------------------
def _ttl_to_text_corpus(ttl_path: Path, limit: int = 800):
    if not ttl_path.exists():
        return []
    lines = []
    for raw in ttl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("@prefix") or line.startswith("#"):
            continue
        if ";" in line or " a " in line or line.endswith("."):
            lines.append(_normalize_ws(line))
        if len(lines) >= limit:
            break
    return lines

def _ensure_rag_index(ttl_path: Path, model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import faiss  # type: ignore
    except Exception:
        return None, None, None, None
    corpus = _ttl_to_text_corpus(ttl_path)
    if not corpus:
        return None, None, None, None
    model = SentenceTransformer(model_name)
    embs = model.encode(corpus, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    import numpy as np
    vecs = np.array(embs, dtype="float32")
    import faiss  # type: ignore
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, corpus, model, faiss

def _rag_retrieve(question: str, index, corpus, model, faiss, top_k: int = 12) -> str:
    import numpy as np
    q_emb = model.encode([question], normalize_embeddings=True)
    vec = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(vec)
    D, I = index.search(vec, top_k)
    snippets = [corpus[i] for i in I[0] if 0 <= i < len(corpus)]
    return "\n".join(snippets)

# -----------------------------
# 三个对外模式（Scheme 3/4/5）
# -----------------------------
def scheme3_template_only(question: str, ttl_path: Path):
    _, _, run_sparql = _lazy_imports()
    sparql = template_topology(question)
    if sparql is None:
        return {"sparql": "", "rows": [], "fallback_used": False, "note": "no_template_match"}
    rows = []
    try:
        rows = run_sparql(ttl_path, sparql).rows or []
    except Exception:
        rows = []
    return {"sparql": sparql, "rows": rows, "fallback_used": False}

def scheme4_llm_direct(question: str, ttl_path: Path):
    _, _, run_sparql = _lazy_imports()
    sparql = _llm_generate_sparql(question)
    try:
        rows = run_sparql(ttl_path, sparql).rows or []
    except Exception:
        rows = []
    return {"sparql": sparql, "rows": rows, "fallback_used": True, "fallback": "llm"}

def scheme5_rag_llm(question: str, ttl_path: Path, model_name: Optional[str] = None):
    EMBED_MODEL, _, run_sparql = _lazy_imports()
    model_name = model_name or EMBED_MODEL
    index, corpus, model, faiss = _ensure_rag_index(Path(ttl_path), model_name)
    context = None
    if index is not None:
        context = _rag_retrieve(question, index, corpus, model, faiss, top_k=12)

    sparql = _llm_generate_sparql(question, context=context)
    try:
        rows = run_sparql(ttl_path, sparql).rows or []
    except Exception:
        rows = []
    return {
        "sparql": sparql,
        "rows": rows,
        "fallback_used": True,
        "fallback": "rag+llm",
        "rag_context": context or ""
    }

def run_langgraph_like_scheme(
    samples: List[Dict[str, Any]],
    ttl_path: str,
    mode: str = "template",   # "template" | "llm" | "rag_llm"
) -> List[Dict[str, Any]]:
    ttl = Path(ttl_path)
    outs = []
    for s in samples:
        q = s.get("question", "")
        if mode == "template":
            res = scheme3_template_only(q, ttl)
        elif mode == "llm":
            res = scheme4_llm_direct(q, ttl)
        elif mode == "rag_llm":
            res = scheme5_rag_llm(q, ttl)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        outs.append({
            "qid": s.get("qid"),
            "question": q,
            "pred_sparql": res.get("sparql", ""),
            "pred_rows": res.get("rows", []),
            "meta": {k: v for k, v in res.items() if k not in ("rows", "sparql")},
        })
    return outs
