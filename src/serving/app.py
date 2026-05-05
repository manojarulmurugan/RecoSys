"""FastAPI serving app for GRU4Rec V9 session-based recommendations.

Startup: port 8080 opens immediately; GCS artifact download + FAISS index
build happens in a background thread (~20-60s). Endpoints return HTTP 503
until loading finishes, then switch to normal operation.

Environment variables:
    GCS_CHECKPOINT_DIR  GCS directory with best_checkpoint.pt + hparams.json
    GCS_VOCABS_PATH     GCS path to vocabs.pkl
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.serving.model_loader import EVENT_TYPE_MAP, ServingArtifacts, load_artifacts

_GCS_CHECKPOINT_DIR = os.environ.get(
    "GCS_CHECKPOINT_DIR",
    "gs://recosys-data-bucket/models/gru4rec_session_v9_1M",
)
_GCS_VOCABS_PATH = os.environ.get(
    "GCS_VOCABS_PATH",
    "gs://recosys-data-bucket/data/1M/vocabs.pkl",
)

_MAX_SEQ_LEN = 20
_MAX_TOP_K   = 50

_artifacts:     ServingArtifacts | None = None
_loading_error: str | None              = None

# Drift report is baked into the Docker image at /app/drift_report.json;
# fall back to the local repo path when running outside Docker.
_DRIFT_PATHS = [
    Path("/app/drift_report.json"),
    Path("reports/drift_report.json"),
]

limiter = Limiter(key_func=get_remote_address)


def _load_in_background() -> None:
    global _artifacts, _loading_error
    print("Loading serving artifacts from GCS …")
    try:
        _artifacts = load_artifacts(_GCS_CHECKPOINT_DIR, _GCS_VOCABS_PATH)
        print("Artifacts ready.")
    except Exception as exc:
        _loading_error = str(exc)
        print(f"ERROR loading artifacts: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start loading in a background thread so port 8080 opens immediately.
    # Cloud Run considers the container started once the port is listening,
    # regardless of whether loading is complete yet.
    t = threading.Thread(target=_load_in_background, daemon=True)
    t.start()
    yield


app = FastAPI(
    title="RecoSys — GRU4Rec V9 Recommender",
    description="Session-based next-item recommendations. NDCG@20=0.2676 on REES46 1M-user dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class SessionEvent(BaseModel):
    item_id:    str
    event_type: str = "view"


class RecommendRequest(BaseModel):
    session: list[SessionEvent] = Field(..., min_length=1)
    top_k:   Annotated[int, Field(ge=1, le=_MAX_TOP_K)] = 20


class RecommendResponse(BaseModel):
    recommendations:  list[str]
    session_length:   int
    known_items:      int
    model_version:    str = "gru4rec_v9_1m"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if _loading_error:
        raise HTTPException(500, f"Artifact loading failed: {_loading_error}")
    if _artifacts is None:
        raise HTTPException(503, "Model loading in progress — retry in ~60s")
    return {
        "status":          "ok",
        "model_version":   "gru4rec_v9_1m",
        "n_items_indexed": int(_artifacts.index.ntotal),
        "ndcg_20":         float(_artifacts.hparams.get("best_val_ndcg_20", 0.2676)),
        "embed_dim":       int(_artifacts.hparams["embed_dim"]),
    }


@app.get("/recommend/example")
def recommend_example():
    return {
        "session": [
            {"item_id": "4209538", "event_type": "view"},
            {"item_id": "4209538", "event_type": "cart"},
            {"item_id": "3622698", "event_type": "view"},
            {"item_id": "3622698", "event_type": "view"},
            {"item_id": "4244526", "event_type": "view"},
        ],
        "top_k": 20,
    }


@app.get("/drift")
def drift_report():
    for path in _DRIFT_PATHS:
        if path.exists():
            return json.loads(path.read_text())
    raise HTTPException(404, "Drift report not available")


@app.post("/recommend", response_model=RecommendResponse)
@limiter.limit("10/minute")
def recommend(request: Request, req: RecommendRequest):
    if _artifacts is None:
        raise HTTPException(503, "Model not loaded")

    item2idx = _artifacts.vocabs["item2idx"]
    idx2item = _artifacts.vocabs["idx2item"]

    # Map raw item_ids → item_idxs; drop OOV silently
    known: list[tuple[int, int]] = []
    for ev in req.session:
        idx = item2idx.get(int(ev.item_id) if ev.item_id.isdigit() else ev.item_id)
        if idx is not None:
            etype = EVENT_TYPE_MAP.get(ev.event_type.lower(), 1)
            known.append((int(idx), etype))

    if not known:
        raise HTTPException(
            422,
            detail="No recognized items in session. "
                   "All item_ids are out-of-vocabulary for this model.",
        )

    # Left-pad to max_seq_len
    item_seq  = [0] * _MAX_SEQ_LEN
    event_seq = [0] * _MAX_SEQ_LEN
    start = max(0, _MAX_SEQ_LEN - len(known))
    for i, (iidx, etype) in enumerate(known[-_MAX_SEQ_LEN:]):
        item_seq [start + i] = iidx
        event_seq[start + i] = etype

    item_t  = torch.tensor([item_seq],  dtype=torch.long)
    event_t = torch.tensor([event_seq], dtype=torch.long)

    with torch.no_grad():
        user_emb = _artifacts.model.encode_sequence(item_t, event_t)

    user_np = user_emb.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(user_np)

    n_candidates = min(req.top_k + 10, _artifacts.index.ntotal)
    _, faiss_indices = _artifacts.index.search(user_np, n_candidates)

    rec_item_ids: list[str] = []
    for pos in faiss_indices[0]:
        if pos < 0:
            continue
        iidx = int(_artifacts.item_idx_array[pos])
        pid = idx2item.get(iidx)
        if pid is not None:
            rec_item_ids.append(str(pid))
        if len(rec_item_ids) >= req.top_k:
            break

    return RecommendResponse(
        recommendations = rec_item_ids,
        session_length  = len(req.session),
        known_items     = len(known),
    )
