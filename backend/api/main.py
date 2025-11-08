"""
FastAPI app exposing endpoints:
- GET /health
- POST /embed  (generate embedding for arbitrary text)
- GET /timeline?concept=...&top_n=...
- GET /era?concept=...&era=1900s&top_n=...
- GET /symbol-pairs?symbol=eye  (served from assets/)
- POST /generate-evolution (NEW: generate word evolution using LLM/APIs)
"""
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Optional
import logging

from api import settings
from api import utils
from api.models import TimelineResponse
from api.etymology_service import etymology_service

# Setup logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Echoes - Backend",
    version="1.0.0",
    description="Track semantic evolution of concepts across time periods"
)

# CORS configuration from settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy model load to speed up import; will load when first used.
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading sentence transformer model: {settings.sentence_transformer_model}")
        _model = SentenceTransformer(settings.sentence_transformer_model)
        logger.info("Model loaded successfully")
    return _model


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Echoes Backend...")
    logger.info(f"Active LLM provider: {settings.active_llm_provider or 'None'}")
    logger.info(f"LLM etymology enabled: {settings.use_llm_etymology}")
    logger.info(f"External APIs enabled: {settings.use_external_apis}")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "msg": "Echoes backend alive",
        "version": "1.0.0",
        "llm_provider": settings.active_llm_provider,
        "llm_enabled": settings.use_llm_etymology
    }


@app.post("/embed")
def embed_text(text: str = Body(..., embed=True)):
    """Generate embedding for arbitrary text."""
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    
    try:
        model = get_model()
        emb = model.encode(text)
        return {"embedding": emb.tolist(), "text": text}
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")


@app.get("/timeline", response_model=TimelineResponse)
def timeline(
    concept: str = Query(..., description="concept name directory under embeddings, e.g. 'freedom'"),
    top_n: int = Query(settings.DEFAULT_TOP_N, ge=1, le=settings.MAX_TOP_N)
):
    """Get timeline of concept evolution across eras."""
    # validate
    concept_dir = settings.embeddings_path / concept
    if not concept_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Concept '{concept}' not found. Generate embeddings first."
        )
    
    try:
        # by default use the *concept* string as query (so query emb of the word)
        model = get_model()
        q_emb = model.encode(concept)
        timeline_data = utils.build_timeline_for_query(concept, q_emb.tolist(), top_n=top_n)
        return timeline_data
    except Exception as e:
        logger.error(f"Error building timeline for '{concept}': {e}")
        raise HTTPException(status_code=500, detail="Failed to build timeline")


@app.get("/era")
def era(
    concept: str = Query(...),
    era: str = Query(...),
    top_n: int = Query(10, ge=1, le=100)
):
    """Get top items for a specific era."""
    fn = f"{era}.json" if not era.endswith(".json") else era
    
    try:
        items = utils.load_era_items(concept, fn)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Era '{era}' not found for concept '{concept}'"
        )
    except Exception as e:
        logger.error(f"Error loading era {era} for {concept}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load era data")
    
    # use concept as query by default
    model = get_model()
    q_emb = model.encode(concept)
    top = utils.top_similar_in_era(q_emb.tolist(), items, top_n=top_n)
    return {"concept": concept, "era": era, "top": top}


@app.get("/symbol-pairs")
def symbol_pairs(
    symbol: str = Query(..., description="symbol name; returns matched image pairs from assets if available.")
):
    """
    Get symbol image pairs (ancient vs modern).
    Very simple static implementation: expects assets/symbols/<symbol>/ancient.png and modern.png
    """
    sym_dir = settings.assets_path / "symbols" / symbol
    
    if not sym_dir.exists():
        return {
            "symbol": symbol,
            "pairs": [],
            "note": "No assets found for this symbol (place images under assets/symbols/<symbol>/)"
        }
    
    pairs = []
    ancient = sym_dir / "ancient.png"
    modern = sym_dir / "modern.png"
    
    # we return relative paths; frontend can request them from server static hosting
    if ancient.exists() or modern.exists():
        pairs.append({
            "ancient": str(ancient) if ancient.exists() else None,
            "modern": str(modern) if modern.exists() else None
        })
    
    return {"symbol": symbol, "pairs": pairs}


@app.post("/generate-evolution")
async def generate_evolution(
    word: str = Body(..., description="Word to analyze"),
    eras: List[str] = Body(..., description="List of eras (e.g., ['1900s', '2020s'])"),
    num_examples: int = Body(5, ge=1, le=20, description="Number of examples per era")
):
    """
    Generate word evolution data across eras using LLM/APIs.
    
    This endpoint will:
    1. Try to use LLM to generate contextual examples
    2. Fallback to Wiktionary API if available
    3. Fallback to CSV files if configured
    
    Returns examples showing how the word's meaning evolved.
    """
    if not word:
        raise HTTPException(status_code=400, detail="word is required")
    
    if not eras:
        raise HTTPException(status_code=400, detail="eras list is required")
    
    try:
        logger.info(f"Generating evolution data for '{word}' across {len(eras)} eras")
        
        evolution_data = await etymology_service.get_word_evolution(
            word=word,
            eras=eras,
            num_examples=num_examples
        )
        
        if not evolution_data:
            raise HTTPException(
                status_code=404,
                detail=f"Could not generate evolution data for '{word}'. "
                       f"Check API keys and fallback options."
            )
        
        return {
            "word": word,
            "eras": eras,
            "evolution": evolution_data,
            "source": settings.active_llm_provider or "csv"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating evolution for '{word}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate evolution data: {str(e)}"
        )


@app.post("/build-embeddings")
async def build_embeddings_endpoint(
    word: str = Body(...),
    eras: List[str] = Body(...),
    num_examples: int = Body(5, ge=1, le=20)
):
    """
    Complete pipeline: Generate evolution data AND create embeddings.
    
    This combines /generate-evolution with the embedding creation process.
    """
    # First, generate the evolution data
    evolution_result = await generate_evolution(word, eras, num_examples)
    evolution_data = evolution_result["evolution"]
    
    # Now create embeddings for each era
    model = get_model()
    concept_dir = settings.embeddings_path / word
    concept_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_created = []
    
    for era in eras:
        if era not in evolution_data:
            logger.warning(f"No data for era {era}, skipping")
            continue
        
        texts = evolution_data[era]
        if not texts:
            continue
        
        # Generate embeddings
        embs = model.encode(texts, show_progress_bar=False)
        
        # Create items
        items = []
        for i, (text, emb) in enumerate(zip(texts, embs)):
            items.append({
                "id": f"{word}_{era}_{i}",
                "text": text,
                "era": era,
                "embedding": emb.tolist()
            })
        
        # Save to JSON
        output_path = concept_dir / f"{era}.json"
        import json
        with output_path.open("w", encoding="utf8") as f:
            json.dump({
                "items": items,
                "meta": {
                    "concept": word,
                    "era": era,
                    "count": len(items),
                    "source": evolution_result["source"]
                }
            }, f, ensure_ascii=False, indent=2)
        
        embeddings_created.append(str(output_path))
        logger.info(f"Created embeddings file: {output_path}")
    
    return {
        "word": word,
        "eras": eras,
        "embeddings_files": embeddings_created,
        "source": evolution_result["source"],
        "message": f"Successfully created embeddings for {len(embeddings_created)} eras"
    }