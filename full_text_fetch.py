"""
Full-text fetch-and-extract stage.

Runs at QUERY TIME (not during ingestion) on only the top-k records the
retriever already selected as most relevant. For each one, tries to fetch
the actual paper's full text using its source_id URL, so generation can
draw on methods/results detail, not just the abstract.

Handles three common patterns seen in AGRIS source_id URLs:
  1. Direct PDF link (source_id IS the PDF, or ends in .pdf)
  2. Journal abstract page with a discoverable PDF link (citation_pdf_url
     meta tag, or an anchor tagged "PDF"/"Download"/"Full text")
  3. Repository item page with multiple downloadable files (e.g. thesis
     split into chapters) — picks the file most likely to be the main text

Falls back to None (caller should use the abstract instead) if nothing
usable is found — never raises out to the caller for a single bad source.

Usage (as a library, called from the query pipeline):
    from full_text_fetch import get_full_text

    text = get_full_text("https://zenodo.org/records/3865420")
    if text:
        # use text (truncated to max_chars) in the generation prompt
    else:
        # fall back to the record's abstract

Install deps:
    pip install requests beautifulsoup4 pdfplumber
"""

import io
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import pdfplumber

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = 25
RETRY_TIMEOUT = 45  # longer timeout for the single retry attempt
MAX_CHARS_DEFAULT = 15000  # cap per-source text to keep prompts manageable

# Domains measured (via evaluate_fulltext_coverage.py, n=50 real samples) to
# reliably block non-browser requests (403 Forbidden regardless of headers/retries).
# Skipping these immediately avoids burning a 25-45s timeout on a doomed request —
# pure latency savings at query time, no accuracy cost since they never succeed anyway.
KNOWN_BLOCKED_DOMAINS = {
    "www.scielo.cl",
    "www.scielo.org.co",
    "www.scielo.org.ar",
    "scielo.sld.cu",
    "www.gvaa.com.br",
}


def _is_known_blocked(url):
    return urlparse(url).netloc.lower() in KNOWN_BLOCKED_DOMAINS


def _log(debug, message):
    if debug:
        print(f"    [debug] {message}")


def _get_with_retry(url, timeout, debug):
    """One attempt at normal timeout; on a timeout specifically, one retry with a
    longer timeout (many academic servers are just slow, not broken)."""
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.exceptions.Timeout:
        _log(debug, f"timed out at {timeout}s, retrying once with {RETRY_TIMEOUT}s timeout")
        return requests.get(url, headers=HEADERS, timeout=RETRY_TIMEOUT)


# Words that suggest an anchor/link leads to the actual full text
FULLTEXT_HINTS = ["pdf", "download", "full text", "fulltext", "full-text", "view/open"]
# Prefer these filename patterns when a repository lists multiple files
PREFERRED_FILENAME_HINTS = ["full text", "fulltext", "full_text", "artikel", "article", "manuscript"]


def _looks_like_pdf(response):
    ctype = response.headers.get("Content-Type", "").lower()
    return "application/pdf" in ctype


def _extract_pdf_text(pdf_bytes, max_chars):
    text_parts = []
    total_len = 0
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                total_len += len(page_text)
                if total_len >= max_chars:
                    break
    except Exception:
        return None
    full_text = "\n".join(text_parts).strip()
    return full_text[:max_chars] if full_text else None


def _find_pdf_links_in_html(html, base_url):
    """Returns a ranked list of candidate PDF URLs (best guess first), not just one —
    so the caller can fall through to the next if one turns out to be restricted."""
    soup = BeautifulSoup(html, "html.parser")
    ranked = []

    # 1. citation_pdf_url meta tag is the most reliable signal (standard in OJS journals)
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta and meta.get("content"):
        ranked.append(urljoin(base_url, meta["content"]))

    # 2. Anchors whose text explicitly says "full text" / "fulltext" etc — high priority
    #    (important for multi-file repository pages like theses split into chapters,
    #    where we want the whole-document file, not an individual chapter/cover/etc.)
    preferred, other = [], []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = (a.get_text() or "").strip().lower()
        combined = f"{text} {href}".lower()
        if not any(hint in combined for hint in FULLTEXT_HINTS):
            continue
        url = urljoin(base_url, href)
        if url in ranked:
            continue
        if any(hint in text for hint in PREFERRED_FILENAME_HINTS):
            preferred.append(url)
        else:
            other.append(url)

    ranked.extend(preferred)
    ranked.extend(other)
    return ranked


def _try_download_pdf(pdf_url, max_chars, timeout, debug):
    """Attempt to download and extract one candidate PDF URL. Returns text or None —
    None means 'try the next candidate', not necessarily a hard failure."""
    try:
        pdf_resp = _get_with_retry(pdf_url, timeout, debug)
        _log(debug, f"GET {pdf_url} -> status {pdf_resp.status_code}, "
                     f"content-type={pdf_resp.headers.get('Content-Type')}")
        pdf_resp.raise_for_status()
    except Exception as e:
        _log(debug, f"  candidate failed: {e}")
        return None

    if not _looks_like_pdf(pdf_resp) and not pdf_url.lower().endswith(".pdf"):
        _log(debug, "  candidate isn't actually a PDF (likely an access-restricted "
                     "or redirected page) — trying next candidate")
        return None

    text = _extract_pdf_text(pdf_resp.content, max_chars)
    if text:
        _log(debug, f"  extracted {len(text)} chars")
    else:
        _log(debug, "  PDF extraction returned nothing — trying next candidate")
    return text


def get_full_text(source_url, max_chars=MAX_CHARS_DEFAULT, timeout=TIMEOUT, debug=False):
    """
    Attempt to fetch and extract full text for a single source.
    Returns extracted text (str) on success, or None if unavailable.
    Never raises — any failure just results in a None return.
    Pass debug=True to print step-by-step diagnostics to stdout.
    """
    if not source_url or not source_url.startswith("http"):
        _log(debug, "no valid source_url given")
        return None

    if _is_known_blocked(source_url):
        _log(debug, f"domain {urlparse(source_url).netloc} is known to block non-browser "
                     "requests — skipping straight to abstract fallback")
        return None

    try:
        resp = _get_with_retry(source_url, timeout, debug)
        _log(debug, f"GET {source_url} -> status {resp.status_code}, "
                     f"content-type={resp.headers.get('Content-Type')}")
        resp.raise_for_status()
    except Exception as e:
        _log(debug, f"request failed: {e}")
        return None

    # Case 1: the URL itself resolved directly to a PDF
    if _looks_like_pdf(resp) or source_url.lower().endswith(".pdf"):
        _log(debug, "source URL is a direct PDF, extracting text")
        text = _extract_pdf_text(resp.content, max_chars)
        _log(debug, f"extracted {len(text) if text else 0} chars" if text else "PDF extraction returned nothing")
        return text

    # Case 2: it's an HTML page — try to find PDF link candidates on it
    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        _log(debug, f"not HTML or PDF (content-type={content_type}), giving up")
        return None

    candidates = _find_pdf_links_in_html(resp.text, source_url)
    if not candidates:
        _log(debug, "no PDF link found in page (no citation_pdf_url meta tag, "
                     "no matching anchor text) — page may need JS rendering, "
                     "or block non-browser requests")
        return None
    _log(debug, f"found {len(candidates)} candidate PDF link(s): {candidates}")

    # Try each candidate in priority order; fall through if one is restricted/broken
    # (this is exactly the case that broke on the IPB thesis repository — the first
    # file matched happened to be access-restricted, so we now try the next one)
    for i, pdf_url in enumerate(candidates, 1):
        _log(debug, f"trying candidate {i}/{len(candidates)}: {pdf_url}")
        text = _try_download_pdf(pdf_url, max_chars, timeout, debug)
        if text:
            return text

    _log(debug, "all candidates failed")
    return None


def augment_with_full_text(retrieved_records, max_chars=MAX_CHARS_DEFAULT):
    """
    Takes the list of top-k records returned by your FAISS retriever
    (each a dict with at least 'title', 'abstract', 'source_id'), and
    returns the same list with an added 'full_text' field: the fetched
    text if successful, or None if unavailable (caller should fall back
    to 'abstract' for that record in the generation prompt).

    This should be called at query time on the small top-k set only,
    never on the whole corpus.
    """
    augmented = []
    for record in retrieved_records:
        full_text = get_full_text(record.get("source_id"), max_chars=max_chars)
        augmented.append({**record, "full_text": full_text})
    return augmented


if __name__ == "__main__":
    # Quick manual test against the 6 URLs we already spot-checked
    test_urls = [
        "https://zenodo.org/records/3865420",
        "https://revistas.uepg.br/index.php/conexao/en/article/view/6916",
        "https://www.gvaa.com.br/revista/index.php/RBGA/article/view/5273",
        "https://www.gvaa.com.br/revista/index.php/CVADS/article/view/7177",
        "https://repository.ipb.ac.id/handle/123456789/55983",
        "https://journals.ukim.mk/index.php/jafes/en/article/view/1797",
    ]
    for url in test_urls:
        print(url)
        text = get_full_text(url, debug=True)
        status = f"OK ({len(text)} chars)" if text else "FAILED / no full text found"
        print(f"  -> {status}\n")