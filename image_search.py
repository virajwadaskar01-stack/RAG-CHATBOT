"""
image_search.py
------------------
Fetches relevant real photos from Unsplash's free API to accompany chat answers.

Get a free Access Key at: https://unsplash.com/developers
(Create an app on their dashboard - the free tier allows 50 requests/hour,
 which is plenty for a personal/portfolio project.)
"""

import requests

UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"


def search_images(query: str, api_key: str, count: int = 3, page: int = 1) -> list:
    """
    Returns a list of dicts: [{"url": ..., "credit": ..., "credit_link": ...}, ...]
    Returns an empty list if the request fails or no key is provided.

    `page` lets the caller fetch a different batch of results for the same
    query - useful for "show me more photos" style follow-ups, since Unsplash
    otherwise returns the identical top results every time for the same query.
    """
    if not api_key or not query.strip():
        return []

    try:
        response = requests.get(
            UNSPLASH_SEARCH_URL,
            params={"query": query, "per_page": count, "page": page},
            headers={"Authorization": f"Client-ID {api_key}"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", [])[:count]:
            results.append({
                "url": item["urls"]["small"],
                "credit": item["user"]["name"],
                "credit_link": item["user"]["links"]["html"],
            })
        return results
    except (requests.RequestException, KeyError, ValueError):
        # Fail quietly - a missing image is not worth crashing the chat over
        return []
