import argparse
from lib.search_utils import text_processing
from lib.keyword_search import BM25_B, InvertedIndex
from lib.keyword_search import (
    search_command,
    build_command,
    load_command,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
    BM25_K1
)
import math 


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index and save it to disk")
    subparsers.add_parser("load", help="Load the movies list from a disk")
    
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a given document term")
    tf_parser.add_argument("doc_id", type=int, help="The ID of the document")
    tf_parser.add_argument("term", type=str, help="The term to look up")
    
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a given document term")
    idf_parser.add_argument("term", type=str, help="The term to look up")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Calculating the TF-IDF score of a term in a given document")
    tfidf_parser.add_argument("doc_id", type=int, help="The ID of the document")
    tfidf_parser.add_argument("term", type=str, help="The term to look up")
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
   
   
    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    
    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "load":
            load_command()
            
        case "search":
            search_command(args.query)
        
        case "tf":
            try:
                tf_score = tf_command(args.term, args.doc_id)
                print(f"The term '{args.term}' appears {tf_score} times in document '{args.doc_id}'.")
            except ValueError as e:
                print(f"Error: {e}")
            
        case "idf":
            try:
                idf_score = idf_command(args.term)
                print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")
            except ValueError as e:
                print(f"Error: {e}")
            
        case "tfidf":
            try:
                tfidf_score = tfidf_command(args.term, args.doc_id)
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf_score:.2f}")
            except ValueError as e:
                print(f"Error: {e}")
            
        case "bm25idf":
            try:
                bm25idf_score = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf_score:.2f}")
            except ValueError as e:
                print(f"Error: {e}")
        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except ValueError as e:
                print(f"Error: {e}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
