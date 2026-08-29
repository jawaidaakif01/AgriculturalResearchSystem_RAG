import argparse
import sys
import io
import traceback

# Ensure UTF-8 output formatting
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from retrieval_utils import load_vector_store, docs_to_records
from full_text_fetch import augment_with_full_text
from generate_report import generate_report
from web_search import search_web_fallback

# L2 distance cutoff: lower is closer. Matches > 0.75 trigger web fallback.
SIMILARITY_THRESHOLD = 0.75


def main():
    parser = argparse.ArgumentParser(description="Query the Agricultural RAG Assistant.")
    parser.add_argument("query", type=str, help="The research question to ask.")
    parser.add_argument("--k", type=int, default=8, help="Number of chunks to retrieve from FAISS.")
    parser.add_argument("--index_dir", type=str, default="faiss_index", help="Path to FAISS index directory.")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="L2 score threshold for fallback.")
    args = parser.parse_args()

    try:
        print(f"\n[1/4] Loading FAISS vector store from '{args.index_dir}'...", flush=True)
        vector_store = load_vector_store(args.index_dir)
            
        print(f"\n[2/4] Searching top {args.k} matches...", flush=True)
        results = vector_store.similarity_search_with_score(args.query, k=args.k)
        
        use_fallback = False
        if not results:
            print("      -> No documents returned from local index.", flush=True)
            use_fallback = True
        else:
            best_score = results[0][1]
            print(f"      -> Best local match L2 score: {best_score:.4f} (Threshold: {args.threshold:.2f})", flush=True)
            if best_score > args.threshold:
                print(f"      -> Best match exceeds threshold ({best_score:.4f} > {args.threshold:.2f}). Triggering Tavily fallback.", flush=True)
                use_fallback = True

        if use_fallback:
            print("\n[3/4] Fetching web search results via Tavily fallback...", flush=True)
            augmented_records = search_web_fallback(args.query, max_results=5)
            print(f"      -> Retrieved {len(augmented_records)} web search sources.", flush=True)
        else:
            docs = [doc for doc, score in results]
            records = docs_to_records(docs)
            print(f"      -> Converted to {len(records)} unique local papers.", flush=True)
            
            print("\n[3/4] Fetching full text for retrieved papers...", flush=True)
            augmented_records = augment_with_full_text(records)
            full_text_count = sum(1 for r in augmented_records if r.get("full_text"))
            print(f"      -> Full text obtained for {full_text_count}/{len(augmented_records)} sources.", flush=True)

        print("\n[4/4] Generating synthesized report via Gemini...", flush=True)
        report = generate_report(args.query, augmented_records)
        
        mode = "WEB FALLBACK (Tavily)" if use_fallback else "LOCAL CORPUS (AGRIS)"
        print("\n" + "="*80)
        print(f"FINAL RESEARCH REPORT [{mode}]")
        print("="*80 + "\n")
        print(report)
        print("\n" + "="*80)

    except Exception as e:
        print(f"\n[ERROR OCCURRED]: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()