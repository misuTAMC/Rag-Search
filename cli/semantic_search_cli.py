import argparse

from lib.keyword_search import load_movie
from lib.semantic_search import (
    ChunkedSemanticSearch,
    chunk_text,
    embed_query_text,
    embed_text, 
    search,
    semantic_chunk_text,
    verify_embeddings, 
    verify_model
)

def main()->None:
    parser=argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers=parser.add_subparsers(dest="command",help="Avalable commands")
    
    subparsers.add_parser("verify",help="Verify the embedding model loads property")
    
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_parser.add_argument("text", type=str, help="The string text to be embedded")
    
    subparsers.add_parser("verify_embeddings",help="Verify the embeddings model loads property")
    
    embed_query_parser=subparsers.add_parser("embed_query",help="Encode query with embedding model")
    embed_query_parser.add_argument("query",type=str,help="User query to be encoder")
    
    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit",type=int,default=5,help="Number of result to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document")
    chunk_parser.add_argument("text", type=str, help="Document to be chunked")
    chunk_parser.add_argument("--chunk_size",type=int,default=200,help="Number of words in each fixed size")
    chunk_parser.add_argument("--overlap",type=int,default=0,help="Number of words overlap")

    semantic_parser = subparsers.add_parser("semantic_chunk", help="Semantically chunk a document by sentences")    
    semantic_parser.add_argument("text", type=str, help="Document to be chunked")
    semantic_parser.add_argument("--max_chunk_size",type=int,default=4,help="Max number of sentences per chunk")
    semantic_parser.add_argument("--overlap",type=int,default=0,help="Number of sentences to overlap")


    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate and cache semantic embeddings for document chunks")

    args=parser.parse_args()
    
    
    
    match args.command:
        case "embed_chunks":
            documents=load_movie()
            chunked_search=ChunkedSemanticSearch()
            chunk_embeddings=chunked_search.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(chunk_embeddings)} chunked embeddings")
        
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