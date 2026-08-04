import json
from time import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import track

from lib.keyword_search import load_movie
from lib.search_utils import PROJECT_ROOT
from lib.hybrid_search import HybridSearch

def load_test_cases():
    with open(PROJECT_ROOT /'data'/'golden_dataset.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)['test_cases']
    return test_cases
'''Precision asks: "How much of what you found is relevant?"
Recall asks: "How much of what's relevant did you find?"
'''

def evaluate(limit: int):
    console = Console()
    test_cases = load_test_cases()
    movies = load_movie()
    
    searcher = HybridSearch(movies)
    
    total_precision = 0.0
    total_recall=0.0
    total_f1=0.0
    start_time = time()
    
    console.print(Panel(
        f"[bold green]SEARCH EVALUATION ENGINE[/bold green]\n"
        f"Evaluating [bold cyan]{len(test_cases)}[/bold cyan] Golden Test Cases with [bold yellow]K = {limit}[/bold yellow] (Precision@{limit})",
        border_style="green",
        expand=False
    ))
    print(f"k={limit}\n") 

    for test_case in track(test_cases, description="Processing evaluation cases..."):
        query = test_case['query']
        exp = test_case['relevant_docs']
        rrf_results = searcher.rrf_search(query, k=60, limit=limit)
        relevant_cnt = 0
        retrieved_titles = []
        
        for rrf_result in rrf_results:
            title = rrf_result['title']
            retrieved_titles.append(title)
            if title in exp:
                relevant_cnt += 1
                
        precision = relevant_cnt / limit if limit > 0 else 0.0
        recall = relevant_cnt/len(exp) if exp else 0
        f1 = 2 * (precision * recall) / (precision + recall)

        total_precision += precision
        total_recall+=recall
        total_f1+=f1
        
        retrieved_str = ", ".join(retrieved_titles)
        relevant_str = ", ".join(exp)
        
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  -F1@{limit}: {f1:.4f}")
        print(f"  - Retrieved: {retrieved_str}")
        print(f"  - Relevant: {relevant_str}\n")


    mean_precision = total_precision / len(test_cases) if test_cases else 0.0
    mean_recall = total_recall / len(test_cases) if test_cases else 0.0
    mean_f1=total_f1/len(test_cases) if test_cases else 0
    elapsed_time = time() - start_time
    
    summary_table = Table(title="Evaluation Summary Metrics", border_style="cyan", show_header=True)
    summary_table.add_column("Metric", style="bold white")  
    summary_table.add_column("Value", style="bold yellow", justify="right")
    
    summary_table.add_row(f"Mean Precision@{limit} (MP@{limit})", f"{mean_precision:.4f} ({mean_precision*100:.1f}%)")
    summary_table.add_row(f"Mean Recall@{limit} (MR@{limit})", f"{mean_recall:.4f} ({mean_recall*100:.1f}%)")
    summary_table.add_row(f"Mean F1@{limit} (MF1@{limit})", f"{mean_f1:.4f} ({mean_f1*100:.1f}%)")
    summary_table.add_row("Total Test Cases", str(len(test_cases)))
    summary_table.add_row("Evaluation Latency", f"{elapsed_time:.2f} seconds")
    
    console.print("\n")
    console.print(summary_table)
