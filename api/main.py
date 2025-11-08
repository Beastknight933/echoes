"""
FastAPI app exposing endpoints:
- GET /health
- POST /embed  (generate embedding for arbitrary text)
- GET /timeline?concept=...&top_n=...
- GET /era?concept=...&era=1900s&top_n=...
- GET /symbol-pairs?symbol=eye  (served from assets/)
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List
import os

from api import utils
from api.models import TimelineResponse

MODEL_NAME = "paraphrase-MiniLM-L6-v2"  # same as build script
EMBED_BASE = Path("embeddings")
ASSETS_BASE = Path("assets")

app = FastAPI(title="Echoes - Backend (v1.0.0)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Lazy model load to speed up import; will load when first used.
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

@app.get("/health")
def health():
    return {"status": "ok", "msg": "Echoes backend alive"}

@app.post("/embed")
def embed_text(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    model = get_model()
    emb = model.encode(text)
    return {"embedding": emb.tolist()}

@app.get("/timeline", response_model=TimelineResponse)
def timeline(concept: str = Query(..., description="concept name directory under embeddings, e.g. 'freedom'"),
             top_n: int = Query(6, ge=1, le=50)):
    # validate
    concept_dir = EMBED_BASE / concept
    if not concept_dir.exists():
        raise HTTPException(status_code=404, detail=f"Concept '{concept}' not found under {EMBED_BASE}")
    # by default use the *concept* string as query (so query emb of the word)
    model = get_model()
    q_emb = model.encode(concept)
    timeline = utils.build_timeline_for_query(concept, q_emb.tolist(), top_n=top_n)
    return timeline

@app.get("/era")
def era(concept: str = Query(...), era: str = Query(...), top_n: int = Query(10, ge=1, le=100)):
    fn = f"{era}.json" if not era.endswith(".json") else era
    try:
        items = utils.load_era_items(concept, fn)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Era '{era}' not found for concept '{concept}'")
    # use concept as query by default
    model = get_model()
    q_emb = model.encode(concept)
    top = utils.top_similar_in_era(q_emb.tolist(), items, top_n=top_n)
    return {"concept": concept, "era": era, "top": top}

@app.get("/symbol-pairs")
def symbol_pairs(symbol: str = Query(..., description="symbol name; returns matched image pairs from assets if available.")):
    """
    Very simple static implementation: expects assets/symbols/<symbol>/ancient.png and modern.png
    """
    sym_dir = ASSETS_BASE / "symbols" / symbol
    if not sym_dir.exists():
        return {"symbol": symbol, "pairs": [], "note": "No assets found for this symbol (place images under assets/symbols/<symbol>/)"}
    pairs = []
    ancient = sym_dir / "ancient.png"
    modern = sym_dir / "modern.png"
    # we return relative paths; frontend can request them from server static hosting or the hosting solution you pick.
    if ancient.exists() or modern.exists():
        pairs.append({
            "ancient": str(ancient) if ancient.exists() else None,
            "modern": str(modern) if modern.exists() else None
        })
    return {"symbol": symbol, "pairs": pairs}
