import argparse

from dotenv import load_dotenv
from lib.hybrid_search import normalize_scores, weighted_search_command
from lib.search_utils import DEFAULT_ALPHA, DEFAULT_SEARCH_LIMIT


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalizes a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="*", type=float, help="Scores to normalize"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search",
        help="Weighted hybrid search",
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
    )
    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return (default=5)",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_scores(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)
            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
