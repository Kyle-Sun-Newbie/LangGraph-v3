# app/experiments/schemes/fewshot_faiss.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import os, json, hashlib
import numpy as np

from ..config import EMBED_MODEL, TOP_K, EMBED_BATCH_SIZE, CACHE_DIR

# 仅保留必要依赖
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore
try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore
try:
    import hnswlib  # type: ignore
except Exception:
    hnswlib = None  # type: ignore

# ---- 可调但保持简洁的几个环境变量（都有合理默认值） ----
TOPN_BASE  = int(os.getenv("TOPN_BASE", 16))  # 候选池下限
TOPN_MULT  = int(os.getenv("TOPN_MULT", 4))   # 候选池 = max(BASE, MULT*k)
DUP_THRESH = float(os.getenv("DUP_COS_THRESHOLD", "0.92"))  # few-shot 去重阈值（余弦）

# ---------------- dataclass ----------------
@dataclass
class FewshotIndex:
    backend: str
    dim: int
    index: Any
    texts: List[str]
    outs: List[str]
    metas: List[Dict[str, Any]]
    emb_model_name: str
    efS: int = 128
    q_encoder: Any = None
    index_sig: str = ""
    vecs_norm: Optional[np.ndarray] = None  # 归一化后的库向量（去重/相似度）

# ---------------- utils ----------------
def _hash_sig(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

def _texts_hash(texts: List[str]) -> str:
    h = hashlib.md5()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype("float32", copy=False)

def _load_embedder():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    return SentenceTransformer(EMBED_MODEL)

def _encode_in_batches(model, texts: List[str], bs: int = 64) -> np.ndarray:
    arrs: List[np.ndarray] = []
    for i in range(0, len(texts), bs):
        part = model.encode(texts[i:i+bs], show_progress_bar=False, normalize_embeddings=False)
        arrs.append(part.astype("float32"))
    if arrs:
        return np.vstack(arrs)
    return np.zeros((0, 384), dtype="float32")

def _emb_cache_paths(sig: str) -> Tuple[Path, Path]:
    return CACHE_DIR / f"emb_{sig}.npz", CACHE_DIR / f"emb_{sig}.meta.json"

def _index_cache_path(sig: str, backend: str) -> Path:
    ext = ".faiss" if backend.startswith("faiss") else ".hnsw"
    return CACHE_DIR / f"index_{backend}_{sig}{ext}"

def _try_load_emb_cache(sig: str):
    pz, pm = _emb_cache_paths(sig)
    if pz.exists() and pm.exists():
        data = np.load(pz)["mat"]
        meta = json.loads(pm.read_text(encoding="utf-8"))
        return data, meta["texts"], meta["outs"], meta["metas"]
    return None

def _save_emb_cache(sig: str, mat: np.ndarray, texts: List[str], outs: List[str], metas: List[Dict[str, Any]]):
    pz, pm = _emb_cache_paths(sig)
    np.savez_compressed(pz, mat=mat)
    pm.write_text(json.dumps({"texts": texts, "outs": outs, "metas": metas}, ensure_ascii=False), encoding="utf-8")

def _try_load_index(sig: str, backend: str, dim: int):
    p = _index_cache_path(sig, backend)
    if not p.exists():
        return None
    if backend in ("faiss_flat", "faiss_hnsw"):
        if faiss is None:
            return None
        return faiss.read_index(str(p))
    elif backend == "hnswlib":
        if hnswlib is None:
            return None
        idx = hnswlib.Index(space="ip", dim=dim)
        idx.load_index(str(p), max_elements=10**9)
        return idx
    return None

def _save_index(sig: str, backend: str, index: Any):
    p = _index_cache_path(sig, backend)
    try:
        if backend in ("faiss_flat", "faiss_hnsw"):
            faiss.write_index(index, str(p))  # type: ignore
        elif backend == "hnswlib":
            index.save_index(str(p))          # type: ignore
    except Exception:
        pass

# ---------------- build index ----------------
def build_index(
    train_examples: List[Dict[str, Any]],
    backend: str = "faiss_flat",
    hnsw_params: Optional[Dict[str, Any]] = None,
) -> FewshotIndex:
    texts = [(ex.get("question") or ex.get("input") or "") for ex in train_examples]
    outs  = [(ex.get("answer_sparql") or ex.get("output") or "") for ex in train_examples]
    metas = [{"brick_model": (ex.get("brick_model") or "")} for ex in train_examples]

    sig = _hash_sig(f"{EMBED_MODEL}|N={len(texts)}|H={_texts_hash(texts)}")

    cached = _try_load_emb_cache(sig)
    if cached:
        mat, texts_c, outs_c, metas_c = cached
        if len(texts_c) == len(texts):
            texts, outs, metas = texts_c, outs_c, metas_c
        else:
            cached = None

    if not cached:
        q_encoder = _load_embedder()
        mat = _encode_in_batches(q_encoder, texts, bs=EMBED_BATCH_SIZE)
        _save_emb_cache(sig, mat, texts, outs, metas)
    else:
        q_encoder = _load_embedder()

    dim = mat.shape[1] if mat.size else 384
    mat_norm = _l2_normalize(mat)

    idx_obj = _try_load_index(sig, backend, dim)

    if backend == "faiss_flat":
        if faiss is None:
            raise RuntimeError("faiss is required for faiss_flat")
        if idx_obj is None:
            try:
                faiss.omp_set_num_threads(os.cpu_count() or 1)  # type: ignore
            except Exception:
                pass
            index = faiss.IndexFlatIP(dim)
            index.add(mat_norm)
            _save_index(sig, backend, index)
        else:
            index = idx_obj
        efS_default = 128

    elif backend == "faiss_hnsw":
        if faiss is None:
            raise RuntimeError("faiss is required for faiss_hnsw")
        M  = int(hnsw_params.get("M", 24)) if hnsw_params else 24
        efC = int(hnsw_params.get("efC", 200)) if hnsw_params else 200
        efS = int(hnsw_params.get("efS", 96))  if hnsw_params else 96
        if idx_obj is None:
            try:
                faiss.omp_set_num_threads(os.cpu_count() or 1)  # type: ignore
            except Exception:
                pass
            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = efC
            index.add(mat_norm)
            index.hnsw.efSearch = efS
            _save_index(sig, backend, index)
        else:
            index = idx_obj
        efS_default = efS

    elif backend == "hnswlib":
        if hnswlib is None:
            raise RuntimeError("hnswlib is required for hnswlib backend")
        M  = int(hnsw_params.get("M", 24)) if hnsw_params else 24
        efC = int(hnsw_params.get("efC", 200)) if hnsw_params else 200
        efS = int(hnsw_params.get("efS", 96))  if hnsw_params else 96
        if idx_obj is None:
            index = hnswlib.Index(space="ip", dim=dim)
            index.init_index(max_elements=len(texts), ef_construction=efC, M=M)
            index.add_items(mat_norm, np.arange(len(texts), dtype=np.int32))
            index.set_num_threads(0)
            index.set_ef(efS)
            _save_index(sig, backend, index)
        else:
            index = idx_obj
            index.set_num_threads(0)
        efS_default = efS

    else:
        raise ValueError(f"Unknown backend: {backend}")

    return FewshotIndex(
        backend=backend, dim=dim, index=index, texts=texts, outs=outs, metas=metas,
        emb_model_name=EMBED_MODEL, efS=efS_default, q_encoder=q_encoder,
        index_sig=sig, vecs_norm=mat_norm
    )

# ---------------- retrieval ----------------
def _embed_query(q: str, model) -> np.ndarray:
    vec = model.encode([q], show_progress_bar=False, normalize_embeddings=False).astype("float32")
    return _l2_normalize(vec)

def retrieve(
    idx: FewshotIndex,
    question: str,
    k: int = TOP_K,
    prefer_brick_model: Optional[str] = None,
) -> List[int]:
    # 复用编码器
    if idx.q_encoder is None:
        idx.q_encoder = _load_embedder()
    qv = _embed_query(question, idx.q_encoder)  # (1, dim)

    # 候选池规模（更轻、更稳）
    topN = min(max(TOPN_BASE, TOPN_MULT * k), len(idx.texts))

    # 动态 efSearch（随 k 放大）
    if idx.backend == "faiss_hnsw":
        try:
            idx.index.hnsw.efSearch = max(idx.efS, 10 * k)
        except Exception:
            pass
    elif idx.backend == "hnswlib":
        try:
            idx.index.set_ef(max(idx.efS, 10 * k))
        except Exception:
            pass

    # ANN 检索
    if idx.backend in ("faiss_flat", "faiss_hnsw"):
        D, I = idx.index.search(qv, topN)  # inner-product（≈余弦）
        cand = I[0].tolist()
        dense_scores = D[0].astype("float32")
    elif idx.backend == "hnswlib":
        labels, distances = idx.index.knn_query(qv, k=topN)
        cand = labels[0].tolist()
        dense_scores = distances[0].astype("float32")
    else:
        raise ValueError("Unknown backend")

    # 轻量“同楼宇”小幅加权（避免强制硬排序）
    if prefer_brick_model:
        bm = Path(prefer_brick_model).name
        inc = []
        for r, i in enumerate(cand):
            m = (idx.metas[i] or {}).get("brick_model") or ""
            bonus = 0.05 if (bm and Path(m).name == bm) else 0.0
            inc.append((i, dense_scores[r] + bonus))
        inc.sort(key=lambda x: x[1], reverse=True)
        cand = [i for i, _ in inc]

    # 多样性去重（使用库向量余弦）
    chosen: List[int] = []
    for i in cand:
        if not chosen:
            chosen.append(i)
        else:
            vi = idx.vecs_norm[i]
            if all(float(np.dot(vi, idx.vecs_norm[j])) < DUP_THRESH for j in chosen):
                chosen.append(i)
        if len(chosen) >= k:
            break

    # 兜底：不足 k 则顺序补齐
    if len(chosen) < k:
        for i in cand:
            if i not in chosen:
                chosen.append(i)
                if len(chosen) >= k:
                    break

    return chosen

# ---------------- prompting + generation ----------------
BRICK_PREFIX = """PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
"""

def _clean_sparql(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    for fence in ("```sparql", "```sql", "```", "~~~"):
        if s.startswith(fence):
            s = s[len(fence):].strip()
    if s.endswith("```") or s.endswith("~~~"):
        s = s[:-3].strip()
    if "PREFIX brick:" not in s:
        s = BRICK_PREFIX + "\n" + s
    return s

def build_prompt(user_question: str, shots: List[Dict[str, str]], brick_model_name: Optional[str] = None):
    sys = (
        "You are an expert assistant that writes valid SPARQL 1.1 queries for Brick/Mortar building graphs.\n"
        "Output ONLY the query, no explanations, no code fences. Use given prefixes if present.\n"
        "Prefer concise SELECT queries with variables bound to meaningful URIs."
    )
    if brick_model_name:
        sys += f"\nTarget topology file: {brick_model_name}"

    ex_lines: List[str] = []
    for s in shots:
        q = (s.get('q') or "").strip()
        a = (s.get('a') or "").strip()
        if not q or not a:
            continue
        ex_lines.append(f"Q: {q}\nA:\n{a}")

    content = BRICK_PREFIX + "\n" + "\n\n".join(ex_lines) + f"\n\nUser question: {user_question}\nReturn only the query."
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": content},
    ]

def generate_sparql(
    idx: FewshotIndex,
    question: str,
    llm=None,
    mode: str = "llm_generate",   # or "copy_nearest"
    k: int = TOP_K,
    brick_model: Optional[str] = None,
) -> str:
    cand_ids = retrieve(idx, question, k=k, prefer_brick_model=brick_model)
    shots = [{"q": idx.texts[i], "a": idx.outs[i]} for i in cand_ids if idx.outs[i]]

    if mode == "copy_nearest" or not llm:
        return _clean_sparql(shots[0]["a"]) if shots else ""

    msgs = build_prompt(question, shots, Path(brick_model).name if brick_model else None)
    try:
        out = llm.chat(msgs)
        text = out.strip() if isinstance(out, str) else (out.text if hasattr(out, "text") else str(out))
    except Exception:
        return _clean_sparql(shots[0]["a"]) if shots else ""
    return _clean_sparql(text)
