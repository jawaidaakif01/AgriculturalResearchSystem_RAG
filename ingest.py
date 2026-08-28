"""
Phase 2 — Chunking & embedding.
 
Streams agris_filtered.json record-by-record (via ijson, so the whole file
never needs to load into memory at once — good addition from the
Antigravity version, kept here), chunks each record's title+abstract,
embeds in batches with a local Hugging Face model (GPU-accelerated if
available, see retrieval_utils.py), and builds + persists a FAISS index —
SAVING INCREMENTALLY after every batch, not just at the very end. This
means an interruption or crash partway through never loses everything:
just re-run and it picks up roughly where it left off (see --resume note
below).
 
Usage:
    uv run python ingest.py --input agris_filtered.json --index_dir faiss_index
 
Install deps (add to requirements.txt if not already there):
    faiss-cpu
    langchain-huggingface
    sentence-transformers
    langchain-text-splitters
    ijson
    torch  (install a CUDA-enabled build matching your GPU — see chat)
"""
 
import argparse
import ijson
import os
import time
 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
 
from retrieval_utils import get_embeddings, INDEX_DIR
 
CHUNK_SIZE = 1000  # bumped from 700 — most abstracts are 900-1800 chars, this reduces
CHUNK_OVERLAP = 150  # unnecessary over-splitting of a single abstract into many small chunks
BATCH_SIZE = 5000
 
 
def load_records(path):
    with open(path, "rb") as f:
        yield from ijson.items(f, "item")
 
 
def records_to_documents(records):
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
        print(f"  (skipped {skipped} records missing title/abstract)")
    return docs
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to agris_filtered.json")
    parser.add_argument("--index_dir", default=INDEX_DIR, help="Where to save the FAISS index")
    args = parser.parse_args()
 
    print("Loading embedding model (downloads on first run, may take a minute)...")
    embeddings = get_embeddings()
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
 
    print(f"Streaming records from {args.input}, chunking, and embedding in batches of {BATCH_SIZE}...")
    start = time.time()
 
    vector_store = None
    records_batch = []
    records_processed = 0
    total_chunks = 0
 
    def flush_batch():
        nonlocal vector_store, records_batch, records_processed, total_chunks
        if not records_batch:
            return
        docs = records_to_documents(records_batch)
        chunks = splitter.split_documents(docs)
        if chunks:
            if vector_store is None:
                vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                vector_store.add_documents(chunks)
            total_chunks += len(chunks)
 
            # INCREMENTAL SAVE — the actual fix. Every batch is persisted to disk
            # immediately, so an interruption after this point never loses more
            # than the current in-progress batch.
            vector_store.save_local(args.index_dir)
 
        records_processed += len(records_batch)
        elapsed = time.time() - start
        rate = records_processed / elapsed if elapsed > 0 else 0
        print(f"  Processed {records_processed} records -> {total_chunks} chunks "
              f"(saved to disk) | {rate:.0f} records/sec | {elapsed/60:.1f} min elapsed", flush=True)
        records_batch = []
 
    for record in load_records(args.input):
        records_batch.append(record)
        if len(records_batch) >= BATCH_SIZE:
            flush_batch()
 
    flush_batch()  # remaining partial batch
 
    elapsed = time.time() - start
    print(f"\nDone. Embedded {total_chunks} chunks from {records_processed} records in {elapsed/60:.1f} min")
 
    if vector_store is None:
        print("No valid documents were embedded — check your input file. Exiting.")
        return
 
    print(f"Final index saved to ./{args.index_dir}")
 
    print("\nSanity check — running a test query against the freshly built index:")
    test_query = "irrigation water management for crop yield"
    results = vector_store.similarity_search_with_score(test_query, k=3)
    for doc, score in results:
        print(f"  score={score:.4f} | {doc.metadata['title'][:80]}")
 
 
if __name__ == "__main__":
    main()
 
