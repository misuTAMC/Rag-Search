import argparse
from lib.search_utils import text_processing
from lib.keyword_search import InvertedIndex
from lib.keyword_search import (
    search_command,
    build_command,
    load_command,
    tf_command,
    # tokenize_single_term,
    idf_command,
    tfidf_command,
)
import math 


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index and save it to disk")
    subparsers.add_parser("load",help="Load the movies list from a dist")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser=subparsers.add_parser("tf",help="Get term frequency for a given document term")
    tf_parser.add_argument("doc_id",type=int,help="The ID of the document")
    tf_parser.add_argument("term",type=str,help="The term to look up")
    
    idf_parser=subparsers.add_parser("idf",help="Get inverse document frequency for a given document term")
    idf_parser.add_argument("term",type=str,help="The term to look up")
    
    tfidf_parser=subparsers.add_parser("tfidf",help="Calculating the TF-IDF score of a term in a given document")
    tfidf_parser.add_argument("doc_id",type=int,help="The ID of the document")
    tfidf_parser.add_argument("term",type=str,help="The term to look up")
    
    
    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "load":
            load_command()
            
        case "search":
            search_command(args.query)
        
        case "tf":
            tf_score=tf_command(args.term,args.doc_id)
            print(f"The [{args.term}] appears {tf_score} times.")
        case "idf":
            idf_score=idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")
        case "tfidf":
            tfidf_score=tfidf_command(args.term,args.doc_id)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf_score:.2f}")
                
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
