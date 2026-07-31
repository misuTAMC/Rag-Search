import argparse
import json
import os
from lib.keyword_search import load_movie
from lib.hybrid_search import HybridSearch, reciprocal_rank_fusion_search_command, weighted_search_command 

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- Đăng ký lệnh 'weighted' ----
    weighted_parser = subparsers.add_parser("weighted", help="Run a weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="The search query string")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic search (0.0 to 1.0)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    # ---- Đăng ký lệnh 'rrf' ----
    rrf_parser = subparsers.add_parser("rrf", help="Run a Reciprocal Rank Fusion (RRF) hybrid search")
    rrf_parser.add_argument("query", type=str, help="The search query string")
    rrf_parser.add_argument("-k", type=int, default=60, help="Smoothing constant for RRF calculation")
    rrf_parser.add_argument("--limit", type=int, default=10, help="Number of results to return")
    
    args = parser.parse_args()

    docs = load_movie()
    searcher = HybridSearch(docs)

    match args.command:
        case "weighted":
            weighted_search_command(args,searcher)
        case "rrf":
            reciprocal_rank_fusion_search_command(args,searcher)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
