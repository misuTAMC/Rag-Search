import argparse
from lib.multimodal_search import image_search_command, verify_image_embedding
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

def main()->None:
    
    parser=argparse.ArgumentParser(description="Multimodal Search CLI - Verify Image Embedding")
    subparsers=parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands"
    )
    
    verify_parser=subparsers.add_parser(
        "verify_image_embedding",
        help="Generate and verify CLIP embedding shape for a given image"
    )
    
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file"
    )
    
    img_parser = subparsers.add_parser("image_search", help="Search movies using an image file")
    img_parser.add_argument("image_path", type=str, help="The path to the image file to search with")


    args = parser.parse_args()
    console=Console()
    
    
    match args.command:
        
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        
        case "image_search":
            console.print(f"\n[bold magenta]Image Search Mode:[/bold magenta] Analyzing image [italic white]'{args.image_path}'[/italic white]...")
            
            top_movies = image_search_command(args.image_path)
            
            console.print(f"\n[bold cyan]Top 5 Movie Matches found in Database:[/bold cyan]\n")
            
            for idx, movie in enumerate(top_movies, start=1):
                title_text = Text()
                title_text.append(f"{idx}. ", style="bold green")
                title_text.append(movie['title'], style="bold white")
                title_text.append(f" (Match Score: {movie['similarity_score']:.4f})", style="bold yellow")
                
                body_text = Text(f"{movie['description'][:150]}...", style="dim italic")
                
                panel = Panel(
                    body_text,
                    title=title_text,
                    title_align="left",
                    border_style="magenta",
                    padding=(0, 2)
                )
                console.print(panel)
            console.print("\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()