
from lib.multimodal_search import MultimodalSearch
from lib.hybrid_search import HybridSearch
from lib.keyword_search import load_movie
from lib.hybrid_search import (
    correct_spelling, rewrite_query, expand_query, 
    batch_rerank_results, cross_encoder_rerank
)
def ensemble_consensus_search(query: str, 
                            image_path: str | None = None, 
                            enhance_method: str | None = None,
                            rerank_method: str | None = None,
                            top_k: int = 5,) -> list[dict]:
    query_to_use = query
    if enhance_method == "spell":
        query_to_use = correct_spelling(query)
    elif enhance_method == "rewrite":
        query_to_use = rewrite_query(query)
    elif enhance_method == "expand":
        query_to_use = expand_query(query)

    movies=load_movie()
    
    hybrid_searcher=HybridSearch(movies)
    
    all_discovered_docs={}
    
    consensus_registry={}
    
    score_registry={}
    
    fetch_limit = top_k * 5
    
    try:
        bm25_results=hybrid_searcher._bm25_search(
            query_to_use,limit=top_k*3,
        )
        for idx,doc in enumerate(bm25_results):
            doc_id=doc.get("id")
            
            norm_score=1.0/(1+idx)
            
            if doc_id:
                if doc_id not in all_discovered_docs:
                    all_discovered_docs[doc_id]=doc
                consensus_registry[doc_id]=consensus_registry.get(doc_id,0)+1
                score_registry[doc_id]=score_registry.get(doc_id,0.0)+norm_score
                    
    except Exception as e:
        print(f"Error in BM25 SEARCH : {e}")
        
    try:
        dense_results=hybrid_searcher._dense_search(query_to_use,limit=top_k*3)
        for idx, doc in enumerate(dense_results):
            doc_id = doc.get("id")
            norm_score = 1.0 / (1 + idx)
        
            if doc_id:
                if doc_id not in all_discovered_docs:
                    all_discovered_docs[doc_id] = doc
                consensus_registry[doc_id] = consensus_registry.get(doc_id, 0) + 1
                score_registry[doc_id] = score_registry.get(doc_id, 0.0) + norm_score
    except Exception as e:
        print(f"Error in SEMANTIC SEARCH : {e}")
            
    
    if image_path:
        try:
            multimodal_searcher=MultimodalSearch(movies)
            
            image_results=multimodal_searcher.search_with_image(image_path)
            
            for idx,doc in enumerate(image_results):
                doc_id = doc.get("id") or doc.get("doc_id")
                norm_score=1.0/(1+idx)
                
                if doc_id:
                    original_movie_obj = next((m for m in movies if m.get("id") == doc_id or m.get("doc_id") == doc_id), doc)
                    if doc_id not in all_discovered_docs:
                        all_discovered_docs[doc_id]=original_movie_obj
                    consensus_registry[doc_id]=consensus_registry.get(doc_id,0)+1
                    score_registry[doc_id] = score_registry.get(doc_id, 0.0) + norm_score
        except Exception as e:
            print(f"Error in MULTIMODAL SEARCH : {e}")
    raw_ensemble_results = []
    for doc_id, count in consensus_registry.items():
        original_doc = all_discovered_docs[doc_id]
        raw_ensemble_results.append({
            "id": doc_id,
            "title": original_doc.get("title", "Unknown"),
            "description": original_doc.get("description") or original_doc.get("document", ""),
            "consensus_methods_count": count,
            "ensemble_rank_score": score_registry[doc_id]
        })
        
    raw_ensemble_results.sort(key=lambda x: (-x["consensus_methods_count"], -x["ensemble_rank_score"]))
    candidate_pool = raw_ensemble_results[:top_k * 3]
    if not candidate_pool:
        return []
    if rerank_method == "batch":
        doc_list_strings = [f"ID: {doc['id']}, Title: {doc['title']}, Document: {doc['description']}" for doc in candidate_pool]
        ranked_ids = batch_rerank_results(query_to_use, doc_list_strings)
        
        final_results = [doc for doc in candidate_pool if doc['id'] in ranked_ids]
        final_results.sort(key=lambda x: ranked_ids.index(x['id']))
        for doc in final_results:
            doc['final_rerank_rank'] = ranked_ids.index(doc['id']) + 1
    elif rerank_method == "cross_encoder":
        pairs = [(query_to_use, f"{doc.get('title', '')} - {doc.get('description', '')}") for doc in candidate_pool]
        scores = cross_encoder_rerank(pairs)
        for idx, doc in enumerate(candidate_pool):
            doc['cross_encoder_score'] = scores[idx]
        final_results = sorted(candidate_pool, key=lambda x: x.get('cross_encoder_score', 0.0), reverse=True)
        for idx, doc in enumerate(final_results, start=1):
            doc['final_rerank_rank'] = idx
            
    else:
        final_results = candidate_pool
        for idx, doc in enumerate(final_results, start=1):
            doc['final_rerank_rank'] = idx

    return final_results[:top_k]

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
# Import hàm kiểm định từ file của em
from lib.llm import llm_judge_results
def ensemble_search_command(args):
    

    console = Console()
    img_path = getattr(args, 'image', None)
    enhance_method = getattr(args, 'enhance', None)
    rerank_method = getattr(args, 'rerank_method', None)
    
    console.print(f"\n[bold gold3]ACTIVATE MULTI-ENGINE ENSEMBLE CONSENSUS SEARCH[/bold gold3]")
    console.print(f"Query text: [italic yellow]'{args.query}'[/italic yellow]")
    if img_path: console.print(f"Query image: [italic magenta]'{img_path}'[/italic magenta]")
    if enhance_method: console.print(f"Enhancement Layer: [bold cyan]{enhance_method}[/bold cyan]")
    if rerank_method: console.print(f"Reranking Layer: [bold cyan]{rerank_method}[/bold cyan]")
        
    console.print("\n[bold cyan]Executing whole pipelines (Enhance ➔ Retrieve ➔ Consensus ➔ Rerank)...[/bold cyan]\n")
    
    top_matches = ensemble_consensus_search(
        args.query, 
        image_path=img_path, 
        enhance_method=enhance_method, 
        rerank_method=rerank_method,
        top_k=5
    )

    console.print(f"[bold green]Final Top 5 High-Precision Filtered Results:[/bold green]\n")
    
    for idx, movie in enumerate(top_matches, start=1):
        title_text = Text()
        title_text.append(f"{idx}. ", style="bold green")
        title_text.append(movie['title'], style="bold white")
        
        match_count = movie['consensus_methods_count']
        title_text.append(f" (Matched in {match_count} Engines)", style="bold cyan" if match_count > 1 else "dim white")
        
        body_text = Text(f"Accumulated Position Score: {movie['ensemble_rank_score']:.4f} | Final Pipeline Rank: #{movie.get('final_rerank_rank', idx)}\n", style="bold yellow")
        
        if 'cross_encoder_score' in movie:
            body_text.append(f"Cross-Encoder Score: {movie['cross_encoder_score']:.4f}\n", style="bold magenta")
            
        body_text.append(f"\n{movie['description'][:150]}...", style="dim italic")
        
        panel = Panel(
            body_text,
            title=title_text,
            title_align="left",
            border_style="gold3" if match_count >= 2 else "green",
            padding=(0, 2)
        )
        console.print(panel)

    if getattr(args, "evaluate", False) and top_matches:
        console.print("\n" + "─" * 60)
        
        # Hiển thị Spinner chuyển động tinh tế trong lúc LLM Judge chấm điểm
        with Live(
            Spinner("arc", text=Text(" LLM Judge is calculating absolute relevance scores...", style="bold italic yellow")),
            console=console,
            transient=True
        ):
            evaluation_scores = llm_judge_results(args.query, top_matches)
            
        relevant_top_film = -1
        best_movie = None
        
        # Duyệt qua bảng điểm để bốc ra bộ phim có score cao nhất của Giám khảo
        for movie, score_raw in zip(top_matches, evaluation_scores):
            score = int(score_raw)
            # Ưu tiên lấy bộ phim có thứ hạng tìm kiếm cao hơn nếu có các bộ phim trùng điểm nhau
            if score > relevant_top_film:
                relevant_top_film = score
                best_movie = movie
                
        # NẾU TÌM THẤY PHIM ĐẠT ĐIỂM SỐ LIÊN QUAN (> 0)
        if best_movie and relevant_top_film > 0:
            # Tự động thay đổi màu sắc Badge theo đẳng cấp điểm số (3/3: Cyan xanh ngọc, dưới 3: Vàng chanh)
            badge_color = "turquoise2" if relevant_top_film == 3 else "green_yellow"
            score_status = f"SCORE: {relevant_top_film}/3 EXCELLENT" if relevant_top_film == 3 else f"SCORE: {relevant_top_film}/3 RELEVANT"
            
            # Khởi tạo thanh tiêu đề Panel đa sắc Cyberpunk
            title_text = Text()
            title_text.append("DIRECT MATCHFOUND ", style="bold blink white on magenta")
            title_text.append(f" {score_status} ", style=f"bold black on {badge_color}")
            
            # Tạo nội dung text chi tiết bên trong hộp
            content_text = Text()
            content_text.append(f"Movie Title: ", style="bold cyan")
            content_text.append(f"{best_movie['title']}\n", style="bold white underline")
            
            content_text.append(f"Consensus Verification: ", style="bold dim white")
            content_text.append(f"Verified by {best_movie['consensus_methods_count']} Engines (Rank Score: {best_movie['ensemble_rank_score']:.4f})\n", style="italic dim yellow")
            
            if 'cross_encoder_score' in best_movie:
                content_text.append(f"Deep Attention Score: ", style="bold dim white")
                content_text.append(f"{best_movie['cross_encoder_score']:.4f} (Cross-Encoder Optimized)\n", style="italic dim magenta")
                
            # Đổ ra TOÀN BỘ MÔ TẢ (Full Description) theo đúng ý em
            content_text.append("\nFull Movie Description:\n", style="bold yellow")
            content_text.append(f"\"{best_movie['description']}\"", style="italic gray70")
            
            # Đóng gói sản phẩm vào hộp Panel Rich Neon cao cấp
            direct_answer_panel = Panel(
                content_text,
                title=title_text,
                title_align="center",
                border_style="turquoise2" if relevant_top_film == 3 else "green_yellow",
                padding=(1, 3),
                expand=False
            )
            console.print(direct_answer_panel)
        else:
            # Fallback an toàn nếu tất cả các phim đều bị chấm 0/3 (Không có phim nào khớp ngữ nghĩa)
            console.print(Panel(
                Text("⚠️ Sorry,Agent couldn't find any perfectly relevant movie in the database that matches your specific contextual request.", style="bold italic red", justify="center"),
                title="[bold red]NO MATCHING FOUND[/bold red]",
                border_style="red",
                padding=(1, 2)
            ))
            
    # 3. NẾU KHÔNG BẬT FLAG EVALUATE, IN THEO BẢNG DANH SÁCH TOP 5 THÔ CŨ CỦA EM
    else:
        if top_matches:
            console.print(f"\n[bold green]Final Top 5 High-Precision Filtered Results:[/bold green]\n")
            for idx, movie in enumerate(top_matches, start=1):
                title_text = Text()
                title_text.append(f"{idx}. ", style="bold green")
                title_text.append(movie['title'], style="bold white")
                
                match_count = movie['consensus_methods_count']
                title_text.append(f" (Matched in {match_count} Engines)", style="bold cyan" if match_count > 1 else "dim white")
                
                body_text = Text(f"Accumulated Position Score: {movie['ensemble_rank_score']:.4f} | Final Pipeline Rank: #{movie.get('final_rerank_rank', idx)}\n", style="bold yellow")
                
                if 'cross_encoder_score' in movie:
                    body_text.append(f"Cross-Encoder Score: {movie['cross_encoder_score']:.4f}\n", style="bold magenta")
                    
                body_text.append(f"\n{movie['description'][:150]}...", style="dim italic")
                
                panel = Panel(
                    body_text,
                    title=title_text,
                    title_align="left",
                    border_style="gold3" if match_count >= 2 else "green",
                    padding=(0, 2)
                )
                console.print(panel)
        else:
            console.print("[bold red]❌ No results found.[/bold red]")

    console.print("\n")
