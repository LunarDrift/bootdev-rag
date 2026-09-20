import argparse

from dotenv import load_dotenv
from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Ensure image embeddings were generated successfully",
    )
    verify_image_embedding_parser.add_argument(
        "image", type=str, help="Path to image file"
    )

    image_search_parser = subparsers.add_parser(
        "image_search", help="Image-based search"
    )
    image_search_parser.add_argument("image", type=str, help="Path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)

        case "image_search":
            results = image_search_command(args.image)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (similarity: {result['score']:.3f})")
                print(f"   {result['document']}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
