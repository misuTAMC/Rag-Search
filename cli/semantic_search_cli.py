import argparse

from lib.keyword_search import load_movie
from lib.semantic_search import (
    ChunkedSemanticSearch,
    chunk_text,
    embed_query_text,
    embed_text,
    embeded_chunked_command, 
    search,
    search_chunked_command,
    semantic_chunk_text,
    verify_embeddings, 
    verify_model
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Verify that the embedding model loads properly")
    
    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a given text")
    embed_parser.add_argument("text", type=str, help="The text string to be embedded")
    
    subparsers.add_parser("verify_embeddings", help="Verify that the cached movie embeddings load properly")
    
    embed_query_parser = subparsers.add_parser("embed_query", help="Generate an embedding for a user query")
    embed_query_parser.add_argument("query", type=str, help="The search query string to embed")
    
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str, help="The search query string")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document into fixed-size segments")
    chunk_parser.add_argument("text", type=str, help="The document text to be chunked")
    chunk_parser.add_argument("--chunk_size", type=int, default=200, help="Number of words per fixed-size chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks")

    semantic_parser = subparsers.add_parser("semantic_chunk", help="Semantically chunk a document by sentences")    
    semantic_parser.add_argument("text", type=str, help="The document text to be chunked")
    semantic_parser.add_argument("--max_chunk_size", type=int, default=4, help="Maximum number of sentences per chunk")
    semantic_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences between chunks")

    subparsers.add_parser("embed_chunks", help="Generate and cache semantic embeddings for document chunks")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies using chunked semantic search")
    search_chunked_parser.add_argument("query", type=str, help="The search query string")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

    
    
    args=parser.parse_args()
    
    
    
    match args.command:
        case "search_chunked":
            search_chunked_command(args)
            
        case "embed_chunks":
            embeded_chunked_command()
        
        case "semantic_chunk":
            semantic_chunk_text(args.text,args.max_chunk_size,args.overlap)
        
        case "chunk":
            chunk_text(args.text,args.chunk_size,args.overlap)
        
        case "search":
            search(args.query,args.limit)
            
        case "embed_query":
            embed_query_text(args.query)
            
        case "verify_embeddings":
            verify_embeddings()
        
        case "verify":
            verify_model()
        
        case "embed_text":
            try:
                embed_text(args.text)
            except ValueError as e:
                print(f"Error : {e}")
                
        case _:
            parser.print_help()
            
            
            
            
if __name__ == "__main__":
    main()