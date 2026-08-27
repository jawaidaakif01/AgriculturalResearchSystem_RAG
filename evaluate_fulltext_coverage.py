"""
Full-text fetch success rate — batch evaluation script.

Randomly samples N records from your actual agris_filtered.json, attempts
full-text fetch on each via full_text_fetch.get_full_text(), and reports:
  - overall success rate (a real metric for your project report)
  - per-domain breakdown (which sources tend to work vs. not)
  - a results CSV you can reference/include as evidence in your evaluation section

Usage:
    python evaluate_fulltext_coverage.py --input agris_filtered.json --sample_size 50

Run this from the same folder as full_text_fetch.py (it imports from it directly).
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from urllib.parse import urlparse

from full_text_fetch import get_full_text


def domain_of(url):
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to agris_filtered.json")
    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--output_csv", default="fulltext_eval_results.csv")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducible sampling")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    # Only sample records that actually have a source_id — no point testing what can't be fetched at all
    candidates = [r for r in records if r.get("source_id", "").startswith("http")]
    print(f"Total records: {len(records)} | with a fetchable-looking source_id: {len(candidates)}")

    random.seed(args.seed)
    sample = random.sample(candidates, min(args.sample_size, len(candidates)))
    print(f"Testing a random sample of {len(sample)} records...\n")

    results = []
    domain_stats = defaultdict(lambda: {"success": 0, "total": 0})

    for i, record in enumerate(sample, 1):
        url = record["source_id"]
        domain = domain_of(url)
        text = get_full_text(url)  # debug=False — keep this run's output clean
        success = text is not None

        domain_stats[domain]["total"] += 1
        if success:
            domain_stats[domain]["success"] += 1

        results.append({
            "title": record["title"][:80],
            "source_id": url,
            "domain": domain,
            "success": success,
            "chars_extracted": len(text) if text else 0,
        })

        status = "OK" if success else "failed"
        print(f"[{i}/{len(sample)}] {domain} -> {status}")

    # Write results to CSV for inclusion in your report
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "source_id", "domain", "success", "chars_extracted"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total_success = sum(1 for r in results if r["success"])
    overall_rate = 100 * total_success / len(results) if results else 0

    print(f"\n{'='*50}")
    print(f"OVERALL: {total_success}/{len(results)} succeeded ({overall_rate:.1f}%)")
    print(f"{'='*50}")
    print("\nPer-domain breakdown (sorted by volume):")
    for domain, stats in sorted(domain_stats.items(), key=lambda x: -x[1]["total"]):
        rate = 100 * stats["success"] / stats["total"]
        print(f"  {domain}: {stats['success']}/{stats['total']} ({rate:.0f}%)")

    print(f"\nDetailed results written to {args.output_csv}")


if __name__ == "__main__":
    main()
