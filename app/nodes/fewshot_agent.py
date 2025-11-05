# app/nodes/fewshot_agent.py
from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# 可选依赖（都不存在时退回纯 numpy 检索）
try:
    import hnswlib  # type: ignore
    _HAS_HNSWLIB = True
except Exception:
    hnswlib = None  # type: ignore
    _HAS_HNSWLIB = False

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    _HAS_FAISS = False

from sentence_transformers import SentenceTransformer  # type: ignore

# ========= 配置 =========
# 训练集：优先读目录下所有 train_data_*.json；未设目录则从单文件路径推断其所在目录
_TRAIN_JSON_PATH_DEFAULT = r"F:\\Task\\RAG-LangGraph-Demo-bcp\\data\\All-dataset\\train_data_1.json"
TRAIN_JSON_PATH = os.getenv("FS_TRAIN_JSON", _TRAIN_JSON_PATH_DEFAULT)
TRAIN_DIR = os.getenv("FS_TRAIN_DIR")
TRAIN_GLOB = os.getenv("FS_TRAIN_GLOB", "train_data_*.json")

# 嵌入与召回
_EMB_MODEL_NAME = os.getenv("FS_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_TOPK = int(os.getenv("FS_TOPK", "6"))
_BACKEND = os.getenv("FS_BACKEND", "hnswlib")   # hnswlib | faiss_hnsw | faiss_flat | naive
_HNSW_M = int(os.getenv("FS_HNSW_M", "32"))
_HNSW_EFC = int(os.getenv("FS_HNSW_EFC", "300"))   # efConstruction
_HNSW_EFS = int(os.getenv("FS_HNSW_EFS", "256"))   # efSearch

# ========= 进程内缓存 =========
_EMB_MODEL: Optional[SentenceTransformer] = None
_DIM: Optional[int] = None
_INDEX: Optional[tuple] = None  # (backend, index, text_count)
_CORPUS_QA: List[Tuple[str, str]] = []  # (question, sparql)
_CORPUS_FILES: List[str] = []

# ========= 数据加载 =========

def _get_train_files() -> List[Path]:
    if TRAIN_DIR:
        return [p for p in sorted(Path(TRAIN_DIR).glob(TRAIN_GLOB)) if p.is_file()]
    p = Path(TRAIN_JSON_PATH)
    if p.is_file():
        return [f for f in sorted(p.parent.glob(TRAIN_GLOB)) if f.is_file()]
    if p.exists() and p.is_dir():
        return [f for f in sorted(p.glob(TRAIN_GLOB)) if f.is_file()]
    raise FileNotFoundError(f"[fewshot] 找不到训练集：{TRAIN_JSON_PATH} 或目录未设置")


def _load_corpus() -> List[Tuple[str, str]]:
    global _CORPUS_QA, _CORPUS_FILES
    if _CORPUS_QA:
        return _CORPUS_QA
    qa: List[Tuple[str, str]] = []
    files = _get_train_files()
    _CORPUS_FILES = [str(x) for x in files]
    for fp in files:
        try:
            data = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data:
            q = item.get("question") or item.get("query") or item.get("input") or ""
            s = item.get("sparql") or item.get("target") or item.get("label") or item.get("output") or ""
            if q and s:
                qa.append((q.strip(), s.strip()))
    if not qa:
        raise ValueError("[fewshot] 训练集中没有找到 (question, sparql) 配对")
    _CORPUS_QA = qa
    return qa

# ========= 嵌入与索引 =========

def _get_embedder():
    global _EMB_MODEL, _DIM
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer(_EMB_MODEL_NAME)
        _DIM = int(_EMB_MODEL.get_sentence_embedding_dimension())
    return _EMB_MODEL


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype("float32", copy=False)


class _NaiveIndex:
    """纯 numpy 余弦检索"""
    def __init__(self, mat: np.ndarray):
        self.mat = mat  # (N, D) 已归一化
    def search(self, q: np.ndarray, k: int):
        sims = q @ self.mat.T           # (B, N)
        order = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, order, axis=1)
        return scores, order


def _build_index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    qa = _load_corpus()
    model = _get_embedder()
    texts = [q for q, _ in qa]
    vecs = model.encode(texts, normalize_embeddings=False).astype("float32")
    vecs = _l2_normalize(vecs)
    dim = vecs.shape[1]

    backend = _BACKEND
    if backend == "hnswlib" and _HAS_HNSWLIB:
        idx = hnswlib.Index(space="ip", dim=dim)
        idx.init_index(max_elements=len(texts), ef_construction=_HNSW_EFC, M=_HNSW_M)
        idx.add_items(vecs, np.arange(len(texts), dtype=np.int32))
        idx.set_num_threads(0)
        idx.set_ef(_HNSW_EFS)
        _INDEX = (backend, idx, len(texts))
    elif backend == "faiss_hnsw" and _HAS_FAISS:
        idx = faiss.IndexHNSWFlat(dim, _HNSW_M)
        try:
            idx.hnsw.efConstruction = _HNSW_EFC  # type: ignore
            idx.hnsw.efSearch = _HNSW_EFS        # type: ignore
        except Exception:
            pass
        idx.add(vecs)
        _INDEX = (backend, idx, len(texts))
    elif backend == "faiss_flat" and _HAS_FAISS:
        idx = faiss.IndexFlatIP(dim)
        idx.add(vecs)
        _INDEX = (backend, idx, len(texts))
    else:
        idx = _NaiveIndex(vecs)
        _INDEX = ("naive", idx, len(texts))
    return _INDEX


def _search(qv: np.ndarray, k: int):
    backend, index, n = _build_index()
    k = min(k, n)
    if backend == "hnswlib":
        try:
            index.set_ef(_HNSW_EFS)
        except Exception:
            pass
        labels, dists = index.knn_query(qv, k=k)
        return dists, labels
    return index.search(qv, k)


def _retrieve_examples(query: str, k: int) -> List[Dict[str, str]]:
    qa = _load_corpus()
    model = _get_embedder()
    qv = model.encode([query], normalize_embeddings=False).astype("float32")
    qv = _l2_normalize(qv)
    _, idx = _search(qv, k)
    return [{"question": qa[i][0], "sparql": qa[i][1]} for i in idx[0].tolist()]

# ========= LLM =========

def _call_llm(messages: List[Dict[str, str]]) -> str:
    from openai import OpenAI
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("[fewshot] 未配置 DEEPSEEK_API_KEY / OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content or ""

# ========= Prompt & 前缀 =========
_PREFIX = (
    "PREFIX brick: <https://brickschema.org/schema/Brick#>\n"
    "PREFIX ref:   <https://brickschema.org/schema/Brick/ref#>\n"
    "PREFIX bldg:  <urn:demo-building#>\n"
)


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:sparql)?(.*?)```", text, flags=re.S)
    return m.group(1).strip() if m else text.strip()


def _build_fewshot_prompt(examples: List[Dict[str, str]], question: str) -> str:
    parts = ["你是 SPARQL 专家，请参考示例生成查询。"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"### 示例 {i}\n用户问题：{ex['question']}\nSPARQL：\n```\n{ex['sparql']}\n```")
    parts.append(
        "### 现在的任务\n"
        f"用户问题：{question}\n"
        "请只输出可执行的 SPARQL，需包含必要 PREFIX，不要解释。"
    )
    return "\n\n".join(parts)

# ========= 对外接口 =========

def fewshot_generate_sparql(question: str) -> Dict[str, Any]:
    examples = _retrieve_examples(question, _TOPK)

    copied = examples[0]["sparql"] if examples else ""
    cand1 = copied if copied.strip().upper().startswith("PREFIX") else _PREFIX + copied

    prompt = _build_fewshot_prompt(examples, question)
    messages = [
        {"role": "system", "content": "You are a strict SPARQL generator."},
        {"role": "user", "content": prompt},
    ]

    diag: Dict[str, Any] = {
        "backend": _BACKEND,
        "emb_model": _EMB_MODEL_NAME,
        "topk": _TOPK,
        "train_files": _CORPUS_FILES or [str(p) for p in _get_train_files()],
    }

    try:
        out = _call_llm(messages)
        cand2 = _strip_fences(out)
    except Exception as e:
        diag["llm_error"] = str(e)
        cand2 = ""

    if cand2 and not cand2.strip().upper().startswith("PREFIX"):
        cand2 = _PREFIX + cand2

    return {
        "sparql_copied": cand1,
        "sparql_llm": cand2,
        "examples": examples,
        "diag": diag,
    }
