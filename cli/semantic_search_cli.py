import argparse

from lib.semantic_search import chunk_text, embed_query_text, embed_text, search, verify_embeddings, verify_model

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
    

    args=parser.parse_args()
    
    
    
    match args.command:
        case "chunk":
            chunk_text(args.text,args.chunk_size)
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