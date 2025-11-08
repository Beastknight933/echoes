"""
Utility functions: loading era embeddings, similarity search, drift/shift computation.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from typing import List, Dict, Any

BASE_EMBED_DIR = Path("embeddings")

def load_era_items(concept: str, era_file: str) -> List[Dict[str, Any]]:
    """
    Load items from embeddings/<concept>/<era_file>.json
    era_file should include .json suffix (e.g., '1900s.json')
    """
    p = BASE_EMBED_DIR / concept / era_file
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf8") as f:
        js = json.load(f)
    return js.get("items", [])

@lru_cache(maxsize=128)
def load_all_eras(concept: str) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: { era: '1900s', items: [...] }
    """
    base = BASE_EMBED_DIR / concept
    if not base.exists():
        raise FileNotFoundError(base)
    files = sorted([x.name for x in base.iterdir() if x.suffix == ".json"])
    result = []
    for fn in files:
        items = load_era_items(concept, fn)
        era_name = fn.replace(".json", "")
        result.append({"era": era_name, "items": items})
    return result

def _to_numpy_embeddings(items: List[Dict[str, Any]]) -> np.ndarray:
    embs = [it["embedding"] for it in items]
    return np.array(embs, dtype=float)

def top_similar_in_era(query_emb: List[float], items: List[Dict[str, Any]], top_n: int = 6):
    """
    Return top_n items most similar to query_emb from items.
    Each returned item: {id, text, score}
    """
    if not items:
        return []
    X = _to_numpy_embeddings(items)
    sims = cosine_similarity([query_emb], X)[0]
    idx = np.argsort(sims)[::-1][:top_n]
    out = []
    for i in idx:
        out.append({
            "id": items[i]["id"],
            "text": items[i]["text"],
            "score": float(sims[i])
        })
    return out

def compute_centroid(items: List[Dict[str, Any]]):
    if not items:
        return None
    X = _to_numpy_embeddings(items)
    return np.mean(X, axis=0)

def centroid_shift_score(centroid_a, centroid_b) -> float:
    """
    Return 1 - cosine_similarity(centroid_a, centroid_b) to indicate shift magnitude.
    Larger => more shift.
    """
    if centroid_a is None or centroid_b is None:
        return 0.0
    a = centroid_a.reshape(1, -1)
    b = centroid_b.reshape(1, -1)
    sim = cosine_similarity(a, b)[0][0]
    return float(1.0 - sim)

def build_timeline_for_query(concept: str, query_emb: List[float], top_n: int = 6):
    """
    Returns:
      { concept, timeline: [ { era, top: [{id,text,score}], centroid_shift_from_prev }, ... ] }
    """
    eras = load_all_eras(concept)
    result = []
    prev_centroid = None
    for era in eras:
        items = era["items"]
        top = top_similar_in_era(query_emb, items, top_n=top_n)
        centroid = compute_centroid(items)
        shift = centroid_shift_score(prev_centroid, centroid) if prev_centroid is not None else 0.0
        result.append({"era": era["era"], "top": top, "centroid_shift_from_prev": shift})
        prev_centroid = centroid
    return {"concept": concept, "timeline": result}
