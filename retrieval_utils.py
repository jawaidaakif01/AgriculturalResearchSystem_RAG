"""
Shared utility: convert retrieved LangChain Documents back into the plain
dict shape that full_text_fetch.augment_with_full_text() and
generate_report.generate_report() already expect:
    {"title", "abstract", "date", "subject", "source_id"}

Keeping this in one place means ingest.py and the future query/retrieval
script both produce/consume the same shape, so nothing breaks silently
if one side changes field names.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
INDEX_DIR = "faiss_index"


def get_embeddings():
    """Must be used identically at ingestion time AND query time —
    mismatched embedding models silently produce garbage similarity scores."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


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
