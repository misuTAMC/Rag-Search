import argparse

from cli.lib.rag_llm_command import llm_summarization_command, rag_command
from lib.hybrid_search import HybridSearch
from lib.keyword_search import load_movie

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform multi-document summarization on search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize results for")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to synthesize (default: 5)"
    )

    args = parser.parse_args()
    docs = load_movie()
    searcher = HybridSearch(docs)
    match args.command:
        case "rag":
            rag_command(searcher,args)
        case "summarize":
            llm_summarization_command(searcher, args)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()