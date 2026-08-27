"""
Phase 1 — AGRIS ODS crawler & parser (corrected for actual DCAT structure)

AGRIS ODS is NOT a single downloadable zip. It's a DCAT catalog:
  1. A root XML file lists ~2000 data providers, each with a link to their own XML file.
  2. Each provider's XML file contains the actual bibliographic records (Dublin Core fields).

This script:
  1. Downloads and parses the root catalog file.
  2. Extracts each provider's download URL.
  3. Downloads a capped number of provider files (politely rate-limited).
  4. Parses bibliographic records from each, filters by keyword/year.
  5. Writes the combined, filtered result to JSON.

Usage:
    python crawl_agris.py --output ./agris_filtered.json \
        --keywords crop irrigation soil pest yield fertilizer farming livestock \
        --min_year 2015 --max_records 20000 --max_providers 150

Increase --max_providers if you need more records and have time/bandwidth;
150 providers is a reasonable starting point for a first pass.
"""

import argparse
import json
import time
from xml.etree import ElementTree as ET

import requests

ROOT_URL = "https://agris.fao.org/ods/AGRIS.ODS.xml"

NAMESPACES = {
    "dcat": "http://www.w3.org/ns/dcat#",
    "dct": "http://purl.org/dc/terms/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
}

HEADERS = {"User-Agent": "student-research-project/1.0 (educational RAG capstone)"}


def fetch_xml(url, timeout=30):
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return ET.fromstring(resp.content)


def get_provider_download_urls(root_elem, max_providers):
    """Root file: each dcat:Dataset has a dcat:distribution/dcat:Distribution/dcat:downloadURL."""
    urls = []
    datasets = root_elem.findall(".//dcat:Dataset", NAMESPACES)
    print(f"Root catalog lists {len(datasets)} data provider datasets")

    for ds in datasets:
        dist = ds.find(".//dcat:Distribution", NAMESPACES)
        if dist is None:
            continue
        download_url_elem = dist.find("dcat:downloadURL", NAMESPACES)
        if download_url_elem is None:
            continue
        # downloadURL may be an attribute (rdf:resource) or element text
        url = download_url_elem.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource") \
            or download_url_elem.text
        if url:
            urls.append(url.strip())
        if len(urls) >= max_providers:
            break

    return urls


def extract_text(elem, tag_options):
    for tag in tag_options:
        found = elem.find(tag, NAMESPACES)
        if found is not None and found.text:
            return found.text.strip()
    return None


def parse_provider_file(root_elem):
    """Each provider file contains BibliographicResource elements with DC fields."""
    records = []
    # BibliographicResource elements may or may not be namespace-prefixed depending on file
    resources = root_elem.findall(".//dct:BibliographicResource", NAMESPACES)
    if not resources:
        resources = list(root_elem.iter())  # fallback: scan everything

    for elem in resources:
        title = extract_text(elem, ["dc:title", "dct:title"])
        abstract = extract_text(elem, ["dc:description", "dct:abstract"])
        date = extract_text(elem, ["dc:date", "dct:issued"])
        subject = extract_text(elem, ["dc:subject", "dct:subject"])
        identifier = elem.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}id") or \
            extract_text(elem, ["dc:identifier", "dct:identifier"])

        if not title or not abstract:
            continue

        records.append({
            "title": title,
            "abstract": abstract,
            "date": date,
            "subject": subject,
            "source_id": identifier,
        })

    return records


def year_from_date(date_str):
    if not date_str:
        return None
    for token in date_str.replace("-", " ").replace("/", " ").split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def matches_keywords(record, keywords):
    if not keywords:
        return True
    haystack = f"{record['title']} {record['abstract']} {record.get('subject') or ''}".lower()
    return any(kw.lower() in haystack for kw in keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--keywords", nargs="*", default=[])
    parser.add_argument("--min_year", type=int, default=None)
    parser.add_argument("--max_records", type=int, default=20000)
    parser.add_argument("--max_providers", type=int, default=150,
                         help="Cap on how many data-provider files to crawl (there are ~2000 total)")
    parser.add_argument("--delay", type=float, default=0.5,
                         help="Seconds to wait between provider file downloads (be polite to FAO's servers)")
    args = parser.parse_args()

    print(f"Fetching root catalog: {ROOT_URL}")
    root = fetch_xml(ROOT_URL)

    provider_urls = get_provider_download_urls(root, args.max_providers)
    print(f"Will crawl {len(provider_urls)} provider files")

    kept = []
    for i, url in enumerate(provider_urls, 1):
        if len(kept) >= args.max_records:
            break
        try:
            provider_root = fetch_xml(url)
            records = parse_provider_file(provider_root)
        except Exception as e:
            print(f"  [{i}/{len(provider_urls)}] Skipping {url}: {e}")
            time.sleep(args.delay)
            continue

        added = 0
        for record in records:
            year = year_from_date(record["date"])
            if args.min_year and year and year < args.min_year:
                continue
            if not matches_keywords(record, args.keywords):
                continue
            kept.append(record)
            added += 1
            if len(kept) >= args.max_records:
                break

        print(f"  [{i}/{len(provider_urls)}] {url} -> {len(records)} records, {added} kept (total kept: {len(kept)})")
        time.sleep(args.delay)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Wrote {len(kept)} filtered records to {args.output}")


if __name__ == "__main__":
    main()
