import argparse
import json
import os
from lib.keyword_search import load_movie
from lib.hybrid_search import (
    HybridSearch, 
    reciprocal_rank_fusion_search_command, 
    weighted_search_command, 
    normalize_scores_command
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.lib.llm import correct_spelling

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    weighted_parser = subparsers.add_parser("weighted_search", help="Run a weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="The search query string")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic search (0.0 to 1.0)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rrf_parser = subparsers.add_parser("rrf_search", help="Run a Reciprocal Rank Fusion (RRF) hybrid search")
    rrf_parser.add_argument("query", type=str, help="The search query string")
    rrf_parser.add_argument("-k", type=int, default=60, help="Smoothing constant for RRF calculation")
    rrf_parser.add_argument("--limit", type=int, default=10, help="Number of results to return")
    rrf_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="Query enhancement method, e.g., spell correction")
    rrf_parser.add_argument("--rerank-method",type=str,choices=["individual","batch","cross_encoder"],help="Reranking method for RRF search, e.g., individual reranking,batch reranking")
    rrf_parser.add_argument("--evaluate",action="store_true",help="Use an LLM to evaluate the search results from 0 to 3")
    
    normalize_parser = subparsers.add_parser("normalize", help="Normalize the embeddings and save to a new file")
    normalize_parser.add_argument("input_list_score", type=float,nargs="*", help="List of scores to normalize")
    
    
    
    args = parser.parse_args()

    docs = load_movie()
    searcher = HybridSearch(docs)

    match args.command:
        case "weighted_search":
            weighted_search_command(args,searcher,)
            
        case "rrf_search":
            reciprocal_rank_fusion_search_command(args, searcher)

        case "normalize":
            normalize_scores_command(args)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
"""
Ex                              Alpha   Reason
Exact match	"The Revenant"	    0.8	    Title search needs keywords
Conceptual	"family movies"	    0.2	    Meaning matters more
Mixed	    "2015 comedies"	    0.5	    Both year AND concept
"""