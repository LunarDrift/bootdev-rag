import os

from dotenv import load_dotenv
from openai import OpenAI

from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    OPENROUTER_MODEL,
    OPENROUTER_URL,
    RRF_K,
    SEARCH_MULTIPLIER,
    Movie,
    load_movies,
)

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url=OPENROUTER_URL, api_key=api_key)


def generate_answer(search_results: list[Movie], query: str, limit: int = 5) -> str:
    context = ""
    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {context}

    Answer:"""

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def rag(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = generate_answer(search_results, query, limit)
    return {
        "query": query,
        "search_results": search_results,
        "answer": answer,
    }


def rag_command(query: str):
    return rag(query)
