import argparse

from lib.augmented_generation import (
    citations_command,
    question_command,
    rag_command,
    summarize_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate multi-document summary"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max number of documents to summarize (default=5)",
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Answers include citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max number of documents to use (default=5)",
    )

    question_parser = subparsers.add_parser(
        "question", help="Conversational question-answering"
    )
    question_parser.add_argument("query", type=str, help="Search query")
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max number of documents to use (default=5)",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("RAG Response:")
            print(result["answer"])

        case "summarize":
            result = summarize_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Summary:")
            print(result["summary"])

        case "citations":
            result = citations_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Answer:")
            print(result["answer"])

        case "question":
            result = question_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("Answer:")
            print(result["answer"])

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
