"""
Shared utility: convert retrieved LangChain Documents back into the plain
dict shape that full_text_fetch.augment_with_full_text() and
generate_report.generate_report() already expect:
    {"title", "abstract", "date", "subject", "source_id"}

Keeping this in one place means ingest.py and the future query/retrieval
script both produce/consume the same shape, so nothing breaks silently
if one side changes field names.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
INDEX_DIR = "faiss_index"


def get_embeddings():
    """Must be used identically at ingestion time AND query time —
    mismatched embedding models silently produce garbage similarity scores.

    Uses GPU (CUDA) automatically if available — critical for reasonable
    embedding speed on the full 64K-record corpus. Falls back to CPU if
    no GPU/CUDA-enabled torch is detected, so this still works on any
    machine, just slower.

    batch_size is deliberately conservative (16) for 4GB-class GPUs like
    the RTX 2050 — bge-base plus CUDA/cuDNN overhead can eat 2GB+ before
    encoding even starts, leaving limited headroom. Raise this if you
    have a bigger GPU and want more throughput.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[retrieval_utils] Embedding device: {device}"
          + ("" if device == "cuda" else
             " (no CUDA GPU detected — install a CUDA-enabled torch build "
             "if you have an NVIDIA GPU, this will be much slower on CPU)"))

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


def load_vector_store():
    embeddings = get_embeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def docs_to_records(docs):
    """docs: list of LangChain Document objects returned by a similarity search.
    Returns: list of plain dicts matching the shape full_text_fetch.py and
    generate_report.py expect. Deduplicates by source_id, since multiple
    chunks from the same paper can both show up in top-k results."""
    seen_sources = set()
    records = []
    for doc in docs:
        meta = doc.metadata
        source_id = meta.get("source_id")
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        records.append({
            "title": meta.get("title"),
            "abstract": meta.get("abstract"),  # full original abstract, not just this chunk
            "date": meta.get("date"),
            "subject": meta.get("subject"),
            "source_id": source_id,
        })
    return records