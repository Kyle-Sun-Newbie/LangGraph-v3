# rag_agent.py
from __future__ import annotations
import os, json
from typing import Dict, List, Optional
from pathlib import Path

# ============== 仅保留：LLM 初始化（供 L2 回退使用） ==============
_LLM = None
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

# ============== RAG：SBERT + FAISS，从拓扑派生语料检索 ==============
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from rdflib import Graph

_FAISS_INDEX: Optional[faiss.IndexFlatIP] = None
_FAISS_TEXTS: List[str] = []
_SBERT_MODEL: Optional[SentenceTransformer] = None

def _load_sbert_model() -> SentenceTransformer:
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _SBERT_MODEL

def _auto_corpus_from_topology(limit: int = 500) -> List[str]:
    """把 topology.ttl 的 (s,p,o) 三元组转为简易文本语料；若无文件，用内置占位。"""
    ttl_path = Path(__file__).resolve().parents[2] / "data" / "topology.ttl"
    if not ttl_path.exists():
        return [
            "Room 1205 has three temperature sensors.",
            "Room 2201 has two humidity sensors.",
            "Illuminance sensors measure light intensity."
        ]
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    return [f"{s} {p} {o}" for s, p, o in g][:limit]

def _load_faiss_index() -> faiss.IndexFlatIP:
    """懒加载/构建 FAISS 内积索引，并缓存到磁盘。"""
    global _FAISS_INDEX, _FAISS_TEXTS
    if _FAISS_INDEX is not None:
        return _FAISS_INDEX

    base_dir = Path(__file__).resolve().parents[2] / "data" / "faiss_index"
    index_path = base_dir / "index.faiss"
    text_path = base_dir / "texts.json"
    base_dir.mkdir(parents=True, exist_ok=True)

    model = _load_sbert_model()
    if index_path.exists() and text_path.exists():
        _FAISS_TEXTS = json.loads(text_path.read_text(encoding="utf-8"))
        _FAISS_INDEX = faiss.read_index(str(index_path))
        return _FAISS_INDEX

    corpus = _auto_corpus_from_topology()
    _FAISS_TEXTS = corpus
    embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    _FAISS_INDEX = faiss.IndexFlatIP(embeddings.shape[1])
    _FAISS_INDEX.add(embeddings)

    faiss.write_index(_FAISS_INDEX, str(index_path))
    text_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    return _FAISS_INDEX

def search(question: str, k: int = 5) -> List[Dict]:
    """向量检索最相关的语料片段。"""
    index = _load_faiss_index()
    model = _load_sbert_model()
    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    return [{"text": _FAISS_TEXTS[idx], "score": float(score)}
            for idx, score in zip(I[0], D[0]) if 0 <= idx < len(_FAISS_TEXTS)]

def build_context(chunks: List[Dict]) -> str:
    """把检索结果拼成可读上下文，便于 LLM 条件生成。"""
    if not chunks:
        return ""
    parts = [f"[{i + 1}] {c['text']} (score={c['score']:.3f})" for i, c in enumerate(chunks)]
    return "以下是与问题相关的建筑知识片段：\n" + "\n".join(parts)

# ============== L2 回退：RAG + LLM → SPARQL ==============
def advanced_text_to_sparql(
    question: str,
    retrieved_context: str = "",
    hints: Dict | None = None
) -> str:
    """
    二级回退：把 hints 与检索到的 retrieved_context 一并交给 LLM 生成 SPARQL。
    - question: 原始用户问题
    - retrieved_context: 通过 search()/build_context() 得到的上下文字符串
    - hints: 来自 hint_agent.parse()/get_hints() 的结构化字段
    """
    hints = hints or {}
    prompt = f"""
你是一个建筑领域SPARQL专家。基于以下信息生成精确的SPARQL查询：

知识图谱模式：
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ref:   <https://brickschema.org/schema/Brick/ref#>
PREFIX bldg:  <urn:demo-building#>

核心关系：
- 房间：?room a brick:Room .
- 传感器：?sensor a ?sensorType ; brick:isPointOf ?room .
- 时序数据：?sensor ref:hasTimeseriesReference [ ref:hasTimeseriesId ?tsid ] .

传感器类型：
- 温度：brick:Air_Temperature_Sensor
- 湿度：brick:Relative_Humidity_Sensor
- 照度：brick:Illuminance_Sensor
- CO2：brick:CO2_Level_Sensor
- PM2.5：brick:PM2.5_Sensor

检索到的建筑知识：
{retrieved_context}

用户问题：{question}
问题类型：{hints.get('question_type', 'unknown')}
房间号：{hints.get('room', '未指定')}
监测指标：{hints.get('metric', '未指定')}
时间范围：{json.dumps(hints.get('time_range', {}), ensure_ascii=False)}
统计需求：{hints.get('need', [])}

请只输出SPARQL查询：
""".strip()

    llm = _get_llm()
    if llm is None:
        return """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ref:   <https://brickschema.org/schema/Brick/ref#>
PREFIX bldg:  <urn:demo-building#>

SELECT ?room ?pt ?ptType ?tsid WHERE {
  ?room a brick:Room .
  ?pt a ?ptType ;
      brick:isPointOf ?room ;
      ref:hasTimeseriesReference [ ref:hasTimeseriesId ?tsid ] .
} LIMIT 50
""".strip()

    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or (resp if isinstance(resp, str) else "")
        return _clean_sparql_response(text)
    except Exception:
        return "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 20"

def _clean_sparql_response(sparql: str) -> str:
    """去掉代码块、补齐必要 PREFIX。"""
    s = (sparql or "").strip()
    for block in ("```sparql", "```sql", "```"):
        if block in s:
            parts = s.split(block)
            if len(parts) >= 2:
                s = parts[1].split("```")[0].strip()
                break
    if "PREFIX brick:" not in s:
        basic_prefixes = (
            "PREFIX brick: <https://brickschema.org/schema/Brick#>\n"
            "PREFIX ref:   <https://brickschema.org/schema/Brick/ref#>\n"
            "PREFIX bldg:  <urn:demo-building#>"
        )
        s = f"{basic_prefixes}\n\n{s}"
    return s
