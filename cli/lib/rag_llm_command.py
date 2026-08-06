

from lib.llm import rag_answer,llm_summarization,llm_citations,llm_qa


from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

def rag_command(searcher, args):
    console = Console()
    
    console.print(f"\n[bold cyan]RAG Engine:[/bold cyan] Fetching context for [italic yellow]'{args.query}'[/italic yellow]...")
    
    results_list = searcher.rrf_search(args.query, limit=5)
    
    context_table = Table(
        title="Retrieved Context (Top 5 Matches)", 
        title_style="bold dim cyan",
        box=None,
        show_header=True,
        header_style="bold magenta"
    )
    context_table.add_column("Index", style="dim", width=6, justify="center")
    context_table.add_column("Movie Title", style="bold white")
    context_table.add_column("RRF Score", style="yellow", justify="right")
    
    for idx, film in enumerate(results_list, start=1):
        context_table.add_row(
            f"#{idx}", 
            film['title'], 
            f"{film.get('score', 0.0):.4f}"
        )
    
    console.print(context_table)
    console.print("\n" + "─" * 60 + "\n")

    with Live(
        Spinner("dots", text=Text(" RAG Agent is synthesizing the answer...", style="italic green")),
        console=console,
        transient=True
    ):
        rag_ans = rag_answer(args.query, results_list)
        
 
    rag_response_panel = Panel(
        Markdown(rag_ans),
        title="[bold green] Agent Response[/bold green]",
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False
    )
    
    console.print(rag_response_panel)
    console.print("\n")


from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown

def llm_summarization_command(searcher, args):
    console = Console()
    
    limit_val = getattr(args, 'limit', 5)
    
    console.print(f"\n[bold cyan]Summarization Pipeline:[/bold cyan] Synthesizing top [bold yellow]{limit_val}[/bold yellow] results for [italic yellow]'{args.query}'[/italic yellow]...")
    
    results_list = searcher.rrf_search(args.query, limit=limit_val)
    
    if not results_list:
        console.print("[bold red] No search results found to summarize.[/bold red]")
        return

    context_table = Table(
        title="Source Documents To Synthesize", 
        title_style="bold dim cyan",
        box=None,
        show_header=True,
        header_style="bold magenta"
    )
    context_table.add_column("Source", style="dim", width=8, justify="center")
    context_table.add_column("Movie Title", style="bold white")
    
    for idx, film in enumerate(results_list, start=1):
        context_table.add_row(f"Doc #{idx}", film['title'])
        
    console.print(context_table)
    console.print("\n" + "─" * 60 + "\n") 

    with Live(
        Spinner("dots", text=Text(" LLM is scanning sources and synthesizing summary...", style="italic yellow")),
        console=console,
        transient=True
    ):
       
        summary_text = llm_summarization(args.query, results_list)

    summary_panel = Panel(
        Markdown(summary_text),
        title="[bold cyan]Summarization[/bold cyan]",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False
    )
    
    console.print(summary_panel)
    console.print("\n")


def llm_citations_command(searcher, args):
    console = Console()
    
    limit_val = getattr(args, 'limit', 5)
    
    console.print(f"\n[bold cyan]Citation RAG Pipeline:[/bold cyan] Generating verifiable answer for [italic yellow]'{args.query}'[/italic yellow]...")
    
    results_list = searcher.rrf_search(args.query, limit=limit_val)
    
    if not results_list:
        console.print("[bold red]No search results found to analyze.[/bold red]")
        return

    context_table = Table(
        title="Source Documents & Citation Map", 
        title_style="bold dim cyan",
        box=None,
        show_header=True,
        header_style="bold magenta"
    )
    context_table.add_column("Citation", style="bold yellow", width=12, justify="center")
    context_table.add_column("Movie Title", style="bold white")
    
    for idx, film in enumerate(results_list, start=1):
        context_table.add_row(f"[{idx}]", film['title'])
        
    console.print(context_table)
    console.print("\n" + "─" * 60 + "\n") 

    with Live(
        Spinner("dots", text=Text(" LLM is analyzing facts and mapping citations...", style="italic green")),
        console=console,
        transient=True
    ):
        citation_text = llm_citations(args.query, results_list)

    answer_panel = Panel(
        Markdown(citation_text),
        title="[bold green]Verified AI Answer[/bold green]",
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False
    )
    
    console.print(answer_panel)
    console.print("\n")


def llm_question_answering_command(searcher, args):
    console = Console()
    
    limit_val = getattr(args, 'limit', 5)
    
    console.print(f"\n[bold yellow]Chat Mode:[/bold yellow] Finding movies to answer: [italic white]'{args.query}'[/italic white]...")
    
    results_list = searcher.rrf_search(args.query, limit=limit_val)
    
    if not results_list:
        console.print("[bold red] No movie context found to answer this question.[/bold red]")
        return

    context_table = Table(
        title="HCMUS Movies Found for Chat Context", 
        title_style="bold dim yellow",
        box=None,
        show_header=True,
        header_style="bold magenta"
    )
    context_table.add_column("Source", style="dim", width=8, justify="center")
    context_table.add_column("Movie Title", style="bold white")
    
    for idx, film in enumerate(results_list, start=1):
        context_table.add_row(f"Film #{idx}", film['title'])
        
    console.print(context_table)
    console.print("\n" + "─" * 60 + "\n")

    with Live(
        Spinner("dots", text=Text(" Agent is typing a response...", style="italic yellow")),
        console=console,
        transient=True
    ):
        qa_answer = llm_qa(args.query, results_list)

    chat_panel = Panel(
        Markdown(qa_answer),
        title="[bold yellow]HCMUS Support Chat[/bold yellow]",
        title_align="left",
        border_style="yellow",
        padding=(1, 2),
        expand=False
    )
    
    console.print(chat_panel)
    console.print("\n")
