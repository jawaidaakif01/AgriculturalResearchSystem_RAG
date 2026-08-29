"""
Web fallback module using Tavily search.
Converts search results into the standard record shape expected by generate_report.
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def search_web_fallback(query, max_results=5):
    """Searches the web via Tavily and returns a list of records matching

    the shared data contract.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key":
        print("[web_search] WARNING: TAVILY_API_KEY not set in .env. Falling back with empty context.")
        return []

    client = TavilyClient(api_key=api_key)
    print(f"[web_search] Executing Tavily web search for: '{query}'...", flush=True)

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=False,
        )
    except Exception as e:
        print(f"[web_search] Tavily search error: {e}", flush=True)
        return []

    records = []
    for item in response.get("results", []):
        records.append({
            "title": item.get("title", "Web Source"),
            "abstract": item.get("content", ""),
            "date": "Recent Web Result",
            "subject": "Web Search",
            "source_id": item.get("url", "unknown"),
            "full_text": item.get("content", ""),  # Tavily snippets provide direct content
        })

    return records