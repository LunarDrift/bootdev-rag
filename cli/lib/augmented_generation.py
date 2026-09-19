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


def generate_summary(search_results: list[Movie], query: str, limit: int = 5) -> str:
    doc_text = ""
    for i, result in enumerate(search_results[:limit], 1):
        doc_text += f"Document {i}: {result['title']}; {result['document']}\n\n"

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

    Query: {query}

    Search results:
    {doc_text}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def summarize(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query, k=RRF_K, limit=limit)
    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    summary = generate_summary(search_results, query, limit)
    return {
        "query": query,
        "summary": summary,
        "search_results": search_results[:limit],
    }


def generate_answer_with_citations(
    search_results: list[Movie], query: str, limit: int = 5
) -> str:
    doc_text = ""
    for i, result in enumerate(search_results[:limit], 1):
        doc_text += f"Document {i}: {result['title']}; {result['document']}\n\n"

    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Webflyx, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {doc_text}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def cite_sources(query: str, limit: int = 5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query, k=RRF_K, limit=limit)
    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    result = generate_answer_with_citations(search_results, query, limit)
    return {
        "query": query,
        "answer": result,
        "search_results": search_results,
    }


def generate_conversational_answer(
    search_results: list[Movie], query: str, limit: int = 5
) -> str:
    doc_text = ""
    for i, result in enumerate(search_results[:limit], 1):
        doc_text += f"Document {i}: {result['title']}; {result['document']}\n\n"

    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

    Question: {query}

    Documents:
    {doc_text}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def conversational_answer(query: str, limit: int = 5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query, k=RRF_K, limit=limit)
    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    result = generate_conversational_answer(search_results, query, limit)
    return {
        "query": query,
        "answer": result,
        "search_results": search_results,
    }


def rag_command(query: str):
    return rag(query)


def summarize_command(query: str):
    return summarize(query)


def citations_command(query: str):
    return cite_sources(query)


def question_command(query: str):
    return conversational_answer(query)
