from pathlib import Path
import os

# 推断项目根目录：.../RAG-LangGraph-Demo-bcp
ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.getenv("DATA_ROOT", ROOT / "data"))
ALL_DATASET_DIR = Path(os.getenv("ALL_DATASET_DIR", DATA_ROOT / "All-dataset"))
MORTAR_DIR = Path(os.getenv("MORTAR_DIR", DATA_ROOT / "mortar"))
DEFAULT_TTL = MORTAR_DIR / "topology.ttl"

OUT_DIR = Path(os.getenv("OUT_DIR", ROOT / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Few-shot 检索参数
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "3"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# 缓存目录（嵌入/索引缓存）
CACHE_DIR = Path(os.getenv("CACHE_DIR", OUT_DIR / "exp_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
