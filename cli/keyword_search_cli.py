import argparse
from search_utils import text_processing
from keyword_search import InvertedIndex

def build_command():
    print("Building inverted index...")
    indexer = InvertedIndex()
    indexer.build()
    indexer.save()
    print("Index saved to disk successfully.")

def tokenize_single_term(term:str)->str:
    tokens=text_processing(term)
    if len(tokens)!=1:
        raise ValueError(f"Expected exactly one token for term '{term}', got {len(tokens)}")
    return tokens[0]

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index and save it to disk")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser=subparsers.add_parser("tf",help="Get term frequency for a given document term")
    tf_parser.add_argument("doc_id",type=int,help="The ID of the document")
    tf_parser.add_argument("term",type=str,help="The term to look up")
    
    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
              
        case "search":
            print(f"Searching for: {args.query}")
            indexer = InvertedIndex()
            
            try:
                indexer.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return 
            match_movies = indexer.search(args.query, max_results=5)#list
            
            if not match_movies:
                print("No movies found matching your query.")
                return

            for index, movie in enumerate(match_movies, start=1):
                print(f"{index}. [{movie['id']}] {movie['title']}")
        
        case "tf":
            indexer=InvertedIndex()
            
            try:
                indexer.load()
                
                target_token=tokenize_single_term(args.term)
                
                frequency=indexer.get_tf(args.doc_id,target_token)
                print(f"The {target_token} appears {frequency} times.")
            except ValueError as e:
                print(f"Error : {e}")
            except FileNotFoundError as e:
                print(f"Error : {e}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
