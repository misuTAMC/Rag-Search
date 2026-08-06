import argparse
from lib.ensemple_consensus_search import ensemble_search_command

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble Search CLI"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands"
    )
    
    ensemble_search_parser = subparsers.add_parser(
        "ensemble_search",
        help="Search movie using Ensemble Search",   
    )
    
    ensemble_search_parser.add_argument(
        "query", type=str, help="The search query string"
    )
    ensemble_search_parser.add_argument(
        "--image", type=str, default=None, help="The path to the image file (Optional)"
    )
    ensemble_search_parser.add_argument(
        "--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method"
    )
    ensemble_search_parser.add_argument(
        "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method for final consensus results"
    )
    ensemble_search_parser.add_argument(
        "--evaluate", action="store_true", help="Use an LLM to evaluate the final results"
    )

    
    args = parser.parse_args()
    
    match args.command:
        case "ensemble_search":
            ensemble_search_command(args)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
