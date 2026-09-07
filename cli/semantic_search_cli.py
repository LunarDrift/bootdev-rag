import argparse

from lib.semantic_search import embed_text, verify_embeddings, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(
        "verify", help="Verify the embedding model was loaded successfully"
    )

    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a text input"
    )
    embed_parser.add_argument("text", help="Text to get an embedding for")

    subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie dataset"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
