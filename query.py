import argparse
import sys
from retrieval_utils import load_vector_store, docs_to_records
from full_text_fetch import augment_with_full_text
from generate_report import generate_report

def main():
    parser = argparse.ArgumentParser(description="Query the Agricultural RAG Assistant.")
    parser.add_argument("query", type=str, help="The research question to ask.")
    parser.add_argument("--k", type=int, default=8, help="Number of chunks to retrieve from FAISS.")
    parser.add_argument("--index_dir", type=str, default="faiss_index", help="Path to FAISS index directory.")
    args = parser.parse_args()

    print(f"\n[1/4] Loading FAISS vector store...")
    try:
        vector_store = load_vector_store()
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print(f"Make sure you have run 'uv run python ingest.py' to generate the '{args.index_dir}' directory first.")
        sys.exit(1)
        
    print(f"[2/4] Searching for top {args.k} semantic matches...")
    # similarity_search_with_score returns a list of (Document, score) tuples
    results = vector_store.similarity_search_with_score(args.query, k=args.k)
    
    # Extract just the LangChain Document objects to pass to docs_to_records
    docs = [doc for doc, score in results]
    
    records = docs_to_records(docs)
    print(f"      -> Retrieved {len(records)} unique sources after deduplication.")
    
    print("[3/4] Attempting to augment sources with full text (skipping known blocked domains)...")
    augmented_records = augment_with_full_text(records)
    
    print("[4/4] Generating synthesized report via Gemini...")
    report = generate_report(args.query, augmented_records)
    
    print("\n" + "="*80)
    print("📋 FINAL RESEARCH REPORT")
    print("="*80 + "\n")
    print(report)
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
