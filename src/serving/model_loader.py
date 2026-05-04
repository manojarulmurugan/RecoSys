"""Load GRU4Rec V9 artifacts and build the FAISS serving index.

Artifact resolution order (first match wins):
  1. /app/model/  — baked into Docker image (instant load, preferred)
  2. GCS          — downloaded at runtime (fallback if image has no baked artifacts)

Called once at FastAPI lifespan startup via a background thread so the HTTP
server binds to port 8080 immediately.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import faiss
import numpy as np
import torch

from src.sequence.models.gru4rec import GRU4RecModel


# Baked artifact paths (populated when COPY model/ model/ is in Dockerfile)
_BAKED_DIR = Path("/app/model")

EVENT_TYPE_MAP: dict[str, int] = {
    "view":     1,
    "cart":     2,
    "purchase": 3,
}


class ServingArtifacts(NamedTuple):
    model:          GRU4RecModel
    index:          faiss.Index
    item_idx_array: np.ndarray   # (n_indexed,) int64 — FAISS row → item_idx
    vocabs:         dict
    hparams:        dict


def _gcs_download(gcs_uri: str, local_path: Path) -> None:
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    from google.cloud import storage  # type: ignore
    print(f"  Downloading {gcs_uri} ...")
    bucket_name, blob_name = gcs_uri.removeprefix("gs://").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(local_path), timeout=300)
    mb = local_path.stat().st_size / 1024 / 1024
    print(f"  Downloaded {gcs_uri} ({mb:.1f} MB)")


def load_artifacts(
    gcs_checkpoint_dir: str,
    gcs_vocabs_path: str,
) -> ServingArtifacts:
    """Resolve artifacts (baked image → GCS fallback), build model + FAISS index."""

    # ── Resolve paths ─────────────────────────────────────────────────────────
    baked_ckpt   = _BAKED_DIR / "model_inference.pt"
    baked_vocabs = _BAKED_DIR / "vocabs.pkl"

    if baked_ckpt.exists() and baked_vocabs.exists():
        print("  Using baked model artifacts from /app/model/")
        ckpt_path   = baked_ckpt
        vocabs_path = baked_vocabs
    else:
        print("  Baked artifacts not found — downloading from GCS ...")
        tmp = Path("/tmp/recosys_serving")
        tmp.mkdir(parents=True, exist_ok=True)
        ckpt_path   = tmp / "model_inference.pt"
        vocabs_path = tmp / "vocabs.pkl"
        gcs_checkpoint_dir = gcs_checkpoint_dir.rstrip("/")
        _gcs_download(f"{gcs_checkpoint_dir}/model_inference.pt", ckpt_path)
        _gcs_download(gcs_vocabs_path, vocabs_path)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("  Loading checkpoint ...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams: dict = ckpt["hparams"]

    model = GRU4RecModel(
        n_items       = int(hparams["n_items"]),
        n_event_types = 4,
        embed_dim     = int(hparams["embed_dim"]),
        gru_hidden    = int(hparams["gru_hidden"]),
        n_layers      = int(hparams["n_layers"]),
        dropout       = float(hparams.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Model loaded  (n_items={hparams['n_items']:,}, embed_dim={hparams['embed_dim']})")

    # ── Build FAISS index — mirrors _build_item_faiss_index in evaluate_sequence.py
    n_items = int(hparams["n_items"])
    with torch.no_grad():
        all_embs = model.get_item_embeddings().cpu().numpy().astype(np.float32)

    keep_mask = np.ones(n_items, dtype=bool)
    keep_mask[0] = False  # exclude PAD token

    item_idx_array = np.where(keep_mask)[0].astype(np.int64)
    embeddings = all_embs[item_idx_array].copy()
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"  FAISS index : {index.ntotal:,} items x {embeddings.shape[1]} dims")

    # ── Load vocabs ───────────────────────────────────────────────────────────
    with open(vocabs_path, "rb") as f:
        vocabs = pickle.load(f)
    print(f"  Vocabs loaded  (items={len(vocabs['item2idx']):,})")

    return ServingArtifacts(
        model          = model,
        index          = index,
        item_idx_array = item_idx_array,
        vocabs         = vocabs,
        hparams        = hparams,
    )
