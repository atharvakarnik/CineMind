"""Fetches movie data from TMDb and Wikipedia, normalizes it into RAG documents, and caches everything to data/."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests
import wikipediaapi

from config import TMDB_API_KEY

TMDB_API_BASE = "https://api.themoviedb.org/3"
WIKIPEDIA_USER_AGENT = "CineMind-RAG-Agent/0.1 (educational project)"
REQUEST_DELAY_SECONDS = 0.3
TOP_CAST_COUNT = 5

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
DOCUMENTS_PATH = DATA_DIR / "documents.json"

SEED_MOVIES: list[dict[str, Any]] = [
    {"title": "The Godfather", "year": 1972},
    {"title": "Pulp Fiction", "year": 1994},
    {"title": "The Shawshank Redemption", "year": 1994},
    {"title": "Inception", "year": 2010},
    {"title": "The Dark Knight", "year": 2008},
    {"title": "Forrest Gump", "year": 1994},
    {"title": "The Matrix", "year": 1999},
    {"title": "Fight Club", "year": 1999},
    {"title": "Interstellar", "year": 2014},
    {"title": "Parasite", "year": 2019},
    {"title": "Spirited Away", "year": 2001},
    {"title": "The Lord of the Rings: The Fellowship of the Ring", "year": 2001},
    {"title": "Goodfellas", "year": 1990},
    {"title": "Schindler's List", "year": 1993},
    {"title": "The Silence of the Lambs", "year": 1991},
    {"title": "Gladiator", "year": 2000},
    {"title": "Titanic", "year": 1997},
    {"title": "Jurassic Park", "year": 1993},
    {"title": "The Lion King", "year": 1994},
    {"title": "Toy Story", "year": 1995},
    {"title": "Whiplash", "year": 2014},
    {"title": "La La Land", "year": 2016},
    {"title": "Get Out", "year": 2017},
    {"title": "Mad Max: Fury Road", "year": 2015},
    {"title": "The Social Network", "year": 2010},
]


def slugify(title: str, year: int) -> str:
    """Build a filesystem-safe cache key for a movie."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"{slug}_{year}"


def cache_or_fetch(
    path: Path, fetch_fn: Callable[[], dict[str, Any] | None], refresh: bool
) -> dict[str, Any] | None:
    """Load a cached JSON response, or call fetch_fn, cache the result, and rate-limit."""
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    data = fetch_fn()
    time.sleep(REQUEST_DELAY_SECONDS)
    if data is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    return data


def tmdb_get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Call a TMDb API endpoint and return its JSON body, or None on failure."""
    try:
        response = requests.get(
            f"{TMDB_API_BASE}{endpoint}",
            params={"api_key": TMDB_API_KEY, **(params or {})},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        print(f"  ! TMDb request failed for {endpoint}: {exc}")
        return None


def fetch_wikipedia_plot(title: str, year: int) -> dict[str, Any] | None:
    """Fetch a movie's Wikipedia article, trying disambiguated titles if the plain title misses."""
    wiki = wikipediaapi.Wikipedia(user_agent=WIKIPEDIA_USER_AGENT, language="en")
    candidates = [title, f"{title} (film)", f"{title} ({year} film)"]
    for candidate in candidates:
        page = wiki.page(candidate)
        if page.exists():
            return {"title": page.title, "summary": page.summary, "text": page.text}
    print(f"  ! No Wikipedia page found for {title}")
    return None


def normalize_documents(
    title: str,
    year: int,
    details: dict[str, Any] | None,
    credits: dict[str, Any] | None,
    reviews: dict[str, Any] | None,
    wiki_data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Turn raw TMDb/Wikipedia responses into normalized {text, metadata} documents."""
    documents: list[dict[str, Any]] = []

    def metadata(content_type: str) -> dict[str, Any]:
        return {"title": title, "year": year, "content_type": content_type}

    if details:
        genres = ", ".join(g["name"] for g in details.get("genres", []))
        runtime = details.get("runtime")
        overview = details.get("overview", "")
        if overview.strip():
            overview_text = f"{title} ({year}). Genres: {genres}. Runtime: {runtime} minutes.\n\n{overview}"
            documents.append({"text": overview_text, "metadata": metadata("overview")})

    if wiki_data and wiki_data.get("text"):
        documents.append({"text": wiki_data["text"], "metadata": metadata("plot")})

    if credits:
        director = next((c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"), None)
        top_cast = credits.get("cast", [])[:TOP_CAST_COUNT]
        cast_lines = [f"{c['name']} as {c.get('character', 'Unknown role')}" for c in top_cast]
        if director or cast_lines:
            cast_text = f"Director: {director or 'Unknown'}.\nCast: " + "; ".join(cast_lines)
            documents.append({"text": cast_text, "metadata": metadata("cast")})

    if reviews:
        for review in reviews.get("results", []):
            content = review.get("content", "")
            if content.strip():
                author = review.get("author", "Anonymous")
                documents.append({"text": f"Review by {author}: {content}", "metadata": metadata("review")})

    return documents


def ingest_movie(title: str, year: int, refresh: bool) -> list[dict[str, Any]]:
    """Fetch, cache, and normalize all documents for a single movie."""
    movie_dir = RAW_DIR / slugify(title, year)

    search = cache_or_fetch(
        movie_dir / "search.json",
        lambda: tmdb_get("/search/movie", {"query": title, "primary_release_year": year}),
        refresh,
    )
    if not search or not search.get("results"):
        print(f"  ! Could not find '{title}' ({year}) on TMDb")
        return []
    movie_id = search["results"][0]["id"]

    details = cache_or_fetch(movie_dir / "details.json", lambda: tmdb_get(f"/movie/{movie_id}"), refresh)
    credits = cache_or_fetch(movie_dir / "credits.json", lambda: tmdb_get(f"/movie/{movie_id}/credits"), refresh)
    reviews = cache_or_fetch(movie_dir / "reviews.json", lambda: tmdb_get(f"/movie/{movie_id}/reviews"), refresh)
    wiki_data = cache_or_fetch(movie_dir / "wikipedia.json", lambda: fetch_wikipedia_plot(title, year), refresh)

    return normalize_documents(title, year, details, credits, reviews, wiki_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest movie data from TMDb and Wikipedia.")
    parser.add_argument("--refresh", action="store_true", help="Force refetching even if cached data exists.")
    args = parser.parse_args()

    all_documents: list[dict[str, Any]] = []
    movies_fetched = 0

    for movie in SEED_MOVIES:
        title, year = movie["title"], movie["year"]
        print(f"Ingesting {title} ({year})...")
        try:
            docs = ingest_movie(title, year, args.refresh)
        except Exception as exc:
            print(f"  ! Unexpected error for {title}: {exc}")
            continue
        if docs:
            all_documents.extend(docs)
            movies_fetched += 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_PATH.write_text(json.dumps(all_documents, indent=2))

    counts_by_type: dict[str, int] = {}
    for doc in all_documents:
        content_type = doc["metadata"]["content_type"]
        counts_by_type[content_type] = counts_by_type.get(content_type, 0) + 1

    print("\n--- Ingestion summary ---")
    print(f"Movies fetched: {movies_fetched}/{len(SEED_MOVIES)}")
    print(f"Total documents: {len(all_documents)}")
    for content_type, count in sorted(counts_by_type.items()):
        print(f"  {content_type}: {count}")


if __name__ == "__main__":
    main()
