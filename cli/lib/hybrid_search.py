import os
from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from lib.llm import (correct_spelling,
                     rewrite_query,
                     expand_query,
                     rerank_results)

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.inverted_index = InvertedIndex()
        if not os.path.exists(self.inverted_index.index_path):
            self.inverted_index.build()
            self.inverted_index.save()
        else:
            self.inverted_index.load()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        # self.idx.load()
        """
        {
            "doc_id":doc_id,
            "title":film_name,
            "score":score
        }
        """
        return self.inverted_index.bm25_search(query, limit)

    def _dense_search(self, query: str, limit: int) -> list[dict]:
        """
        {
            "id": doc_id,
            "title": title,
            "document": document[:100],
            "score": round(score, SCORE_PRECISION),
            "metadata": metadata or {},
        }
        """
        return self.semantic_search.search_chunks(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        """
        Tìm kiếm kết hợp theo trọng số dựa trên thứ hạng (Rank-based Linear Combination).
        alpha: Trọng số cho Semantic Search (0.0 <= alpha <= 1.0)
        (1 - alpha): Trọng số cho BM25 Keyword Search
        """
        # Lấy dư kết quả (limit * 3) để tăng khả năng intersection between 2 approachs
        fetch_limit = limit * 500
        bm25_res = self._bm25_search(query, limit=fetch_limit)
        dense_res = self._dense_search(query, limit=fetch_limit)
        if not bm25_res and not dense_res:
            return []
        # Tạo bản đồ id -> doc để lấy thông tin chi tiết tài liệu ở       
        all_docs = {}
        
        for doc in bm25_res:
            d_id=doc.get("doc_id") 
            movie_obj=self.inverted_index.docmap.get(d_id)
            description=movie_obj["description"] if movie_obj is not None else "" 
            metadata=movie_obj.get("metadata",{}) if movie_obj is not None else {}
    

            all_docs[d_id]={
                "id": d_id,
                "title": doc.get('title'),
                "document": description[:100] ,
                "score": doc.get('score'),
                "metadata": metadata,
            }
        for doc in dense_res:
            d_id = doc.get('id')
            all_docs[d_id] = {
                "id": d_id,
                "title": doc.get('title'),
                "document": doc.get('document'),
                "metadata": doc.get('metadata', {}),
            }

            

        def get_normalized_scores(results,doc_id):
            if not results:
                return {}
            # Lấy danh sách điểm
            scores = [doc.get('score', 0.0) for doc in results]
            min_score = min(scores)
            max_score = max(scores)
            if min_score == max_score:
                return {doc[doc_id]: 1.0 for doc in results}  # Nếu tất cả điểm giống nhau, trả về 1.0 cho tất cả
            normalized_scores = {
                doc[doc_id]:(doc.get('score', 0.0) - min_score) / (max_score - min_score)
                for doc in results
            }
            return normalized_scores

        bm25_scores = get_normalized_scores(bm25_res,"doc_id")
        dense_scores = get_normalized_scores(dense_res,"id")

        # Tính điểm tổng hợp có trọng số
        combined_scores = {}
        for doc_id in all_docs.keys():
            s_bm25 = bm25_scores.get(doc_id, 0.0)
            s_dense = dense_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = (alpha * s_bm25) + ((1 - alpha) * s_dense)

        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        final_results = []
        for doc_id, final_score in sorted_ids:
            final_doc = all_docs[doc_id].copy()
            final_doc['score'] = round(final_score, 4)
            final_results.append(final_doc)
        
        return final_results

    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:
        
        """
        Tìm kiếm kết hợp bằng thuật toán Reciprocal Rank Fusion (RRF).
        k: Hằng số làm mượt (ngăn các tài liệu đứng đầu thao túng quá mức kết quả, mặc định = 60)
        """
        fetch_limit = limit * 500
        bm25_res = self._bm25_search(query, limit=fetch_limit)
        dense_res = self._dense_search(query, limit=fetch_limit)

        all_docs = {}
                
        for doc in bm25_res:
            d_id=doc.get("doc_id") 
            movie_obj=self.inverted_index.docmap.get(d_id)
            description=movie_obj["description"] if movie_obj is not None else "" 
            metadata=movie_obj.get("metadata",{}) if movie_obj is not None else {}

            all_docs[d_id]={
                "id": d_id,
                "title": doc.get('title'),
                "document": description[:100] ,
                "metadata": metadata,
                "score":0.0
            }
        for doc in dense_res:
            d_id = doc.get('id')
            all_docs[d_id] = {
                "id": d_id,
                "title": doc.get('title'),
                "document": doc.get('document'),
                "metadata": doc.get('metadata', {}),
            }
            
        rrf_scores = {}

        for rank, doc in enumerate(bm25_res):
            doc_id = doc['doc_id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        for rank, doc in enumerate(dense_res):
            doc_id = doc['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        final_results = []
        for doc_id, final_score in sorted_ids:
            final_doc = all_docs[doc_id].copy()
            final_doc['score'] = round(final_score, 4) # Lưu điểm RRF đã làm tròn
            final_results.append(final_doc)
        
        return final_results
'''
[{'id': 1771, 'title': 'Paddington', 'document': 'Deep in the rainforests of Peru, a young bear lives peacefully with his Aunt Lucy and Uncle Pastuzo,', 'metadata': {}, 'score': 0.0333}, 
{'id': 1354, 'title': 'Murder She Said', 'document': 'This is based on the Agatha Christie book "4:50 from Paddington" and the opening locale is Paddingto', 'metadata': {}, 'score': 0.0325}, 
{'id': 2833, 'title': "It Couldn't Happen Here", 'document': 'In the early morning, dancers are warming up on an English beach (Clacton-On-Sea.Essex), and Neil Te', 'metadata': {}, 'score': 0.032}]
'''



#* ==================== CLI COMMAND FUNCTIONS ====================
def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)  # Nếu tất cả điểm giống nhau, trả về 1.0 cho tất cả
    return [round((score - min_score) / (max_score - min_score), 4) for score in scores]

def normalize_scores_command(args):
    console = Console()
    if args.input_list_score is None or len(args.input_list_score) == 0:
        console.print("[bold red]Error:[/bold red] No scores provided for normalization.")
        return
    input_list_score = args.input_list_score
    if not isinstance(input_list_score, list) or not all(isinstance(x, (int, float)) for x in input_list_score):
        console.print("[bold red]Error:[/bold red] The input must be a list of numeric scores.")
        return

    normalized_scores = normalize_scores(input_list_score)
    
    console.print(f"\n[bold cyan]Normalized Scores:[/bold cyan] {normalized_scores}\n")
    
def weighted_search_command(args,searcher):
    console = Console()
    results = searcher.weighted_search(args.query, alpha=args.alpha, limit=args.limit)    
    console.print(f"\n[bold cyan]🔍 Running Weighted Hybrid Search for:[/bold cyan] [italic yellow]'{args.query}'[/italic yellow] with alpha = [italic yellow]'{args.alpha}'[/italic yellow]and limit = [italic yellow]'{args.limit}'[/italic yellow]...\n")

    for i, res in enumerate(results, start=1):
        title_text = Text()
        title_text.append(f"{i}. ", style="bold green")
        title_text.append(res['title'], style="bold white")
        title_text.append(f" (Score: {res['score']:.4f})", style="bold yellow")
        
        body_text = Text(f"{res['document']}...", style="dim italic")
        
        panel = Panel(
            body_text,
            title=title_text,
            title_align="left",
            border_style="cyan",
            padding=(0, 2)
        )
        console.print(panel)
def reciprocal_rank_fusion_search_command(args,searcher,enhance=None,rerank_method=None):
    console = Console()
    match args.enhance:
        case "spell":
            
            enhanced_query = correct_spelling(args.query)
            print(f"Enhanced query (spell): '{args.query}' -> '{enhanced_query}'\n")
            args.query = enhanced_query
        case "rewrite":
            enhanced_query = rewrite_query(args.query)
            print(f"Enhanced query (rewrite): '{args.query}' -> '{enhanced_query}'\n")
            args.query = enhanced_query
        case "expand":
            enhanced_query = expand_query(args.query)
            print(f"Enhanced query (expand): '{args.query}' -> '{enhanced_query}'\n")
            args.query = enhanced_query
        case _:
            # No enhancement applied
            pass
    
    if getattr(args, 'rerank_method', None) == "individual":
        fetch_limit = args.limit * 5
        raw_results = searcher.rrf_search(args.query, k=args.k, limit=fetch_limit)
        results = rerank_results(args.query, raw_results)
    else:
        results = searcher.rrf_search(args.query, k=args.k, limit=args.limit)
            
            
            
    console.print(f"\n[bold cyan]🔍 Running Reciprocal Rank Fusion Hybrid Search for:[/bold cyan] [italic yellow]'{args.query}'[/italic yellow] with k = [italic yellow]'{args.k}'[/italic yellow]and limit = [italic yellow]'{args.limit}'[/italic yellow]...\n")

    final_results = results[:args.limit] 
    for i, res in enumerate(final_results, start=1):
        title_text = Text()
        title_text.append(f"{i}. ", style="bold green")
        title_text.append(res['title'], style="bold white")
        
        if 'rerank_score' in res:
            title_text.append(f" (RRF Score: {res['score']:.4f}, LLM Score: {res['rerank_score']:.1f})", style="bold yellow")
        else:
            title_text.append(f" (Score: {res['score']:.4f})", style="bold yellow")
        
        body_text = Text(f"{res['document']}...", style="dim italic")
        
        panel = Panel(
            body_text,
            title=title_text,
            title_align="left",
            border_style="cyan",
            padding=(0, 2)
        )
        console.print(panel)
        
