import os
from typing import TypedDict

from lib.search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    SearchResult,
    format_search_result,
    load_movies,
)

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class CombinedScoreData(TypedDict):
    title: str
    document: str
    bm25_score: float
    semantic_score: float


class WeightedSearchCommandResult(TypedDict):
    original_query: str
    query: str
    alpha: float
    results: list[SearchResult]


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[Movie]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[Movie]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for score in scores:
        normalized = (score - min_score) / (max_score - min_score)
        normalized_scores.append(normalized)

    return normalized_scores


def normalize_search_results(
    results: list[SearchResult],
) -> list[tuple[SearchResult, float]]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    return list(zip(results, normalized))


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> list[SearchResult]:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    alpha: float = DEFAULT_ALPHA,
) -> list[SearchResult]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores: dict[int, CombinedScoreData] = {}

    for result, normalized_score in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if normalized_score > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = normalized_score

    for result, normalized_score in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if normalized_score > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = normalized_score

    hybrid_results: list[SearchResult] = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> WeightedSearchCommandResult:
    movies = load_movies()
    search = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = search.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }
