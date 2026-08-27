"""
Phase 2 — Chunking & embedding.

Loads agris_filtered.json, chunks each record's title+abstract for
embedding, embeds with a local Hugging Face model (no API rate limits,
no cost), and builds + persists a FAISS index to disk.

Each chunk's metadata preserves the FULL original abstract (not just that
chunk's text) plus title/date/subject/source_id — so that after retrieval,
downstream code (full_text_fetch.py, generate_report.py) always has the
complete abstract to work with, even though matching happened at chunk
granularity. See retrieval_utils.py for the shared conversion logic.

Usage:
    uv run python ingest.py --input agris_filtered.json --index_dir faiss_index

Install deps (add to requirements.txt if not already there):
    faiss-cpu
    langchain-huggingface
    sentence-transformers
    langchain-text-splitters
"""

import argparse
import json
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from retrieval_utils import get_embeddings, INDEX_DIR

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def records_to_documents(records):
    """One Document per record BEFORE splitting — the splitter below
    handles breaking longer abstracts into multiple chunks itself."""
    docs = []
    skipped = 0
    for r in records:
        if not r.get("title") or not r.get("abstract"):
            skipped += 1
            continue
        content = f"{r['title']}\n\n{r['abstract']}"
        metadata = {
            "title": r["title"],
            "abstract": r["abstract"],  # full abstract kept here regardless of chunking
            "date": r.get("date"),
            "subject": r.get("subject"),
            "source_id": r.get("source_id"),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    if skipped:
        print(f"Skipped {skipped} records missing title/abstract")
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to agris_filtered.json")
    parser.add_argument("--index_dir", default=INDEX_DIR, help="Where to save the FAISS index")
    args = parser.parse_args()

    print(f"Loading records from {args.input}...")
    records = load_records(args.input)
    print(f"Loaded {len(records)} records")

    docs = records_to_documents(records)
    print(f"Built {len(docs)} documents (pre-chunking)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"Loading embedding model (this downloads the model on first run, may take a minute)...")
    embeddings = get_embeddings()

    print(f"Embedding {len(chunks)} chunks and building FAISS index — this is the slow step, "
          f"expect real time on CPU, more if you don't have a GPU...")
    start = time.time()
    vector_store = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - start
    print(f"Done embedding in {elapsed:.1f}s")

    vector_store.save_local(args.index_dir)
    print(f"Saved FAISS index to ./{args.index_dir}")

    # Quick sanity-check retrieval on a fixed test query
    print("\nSanity check — running a test query against the freshly built index:")
    test_query = "irrigation water management for crop yield"
    results = vector_store.similarity_search_with_score(test_query, k=3)
    for doc, score in results:
        print(f"  score={score:.4f} | {doc.metadata['title'][:80]}")


if __name__ == "__main__":
    main()
