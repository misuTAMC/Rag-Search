import os
from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


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
        fetch_limit = limit * 3
        bm25_res = self._bm25_search(query, limit=fetch_limit)
        dense_res = self._dense_search(query, limit=fetch_limit)

        # Tạo bản đồ id -> doc để lấy thông tin chi tiết tài liệu ở       
        all_docs = {}
        
        for doc in bm25_res:
            d_id=doc.get("doc_id") 
            movie_obj=self.inverted_index.docmap.get(d_id)
            description=movie_obj["description"] if movie_obj is not None else "" 
            metadata=self.inverted_index.docmap.get("metadata",{})
    

            all_docs[d_id]={
                "id": d_id,
                "title": doc.get('title'),
                "document": description[:100] ,
                "score": doc.get('score'),
                "metadata": metadata,
            }
        for doc in dense_res:
            d_id=doc.get('id')
            all_docs[d_id]=doc
            

        # Normalization point from 0.0 đến 1.0 (Min-Max Scaling dựa trên Rank)
        def get_rank_scores(results,doc_id):
            scores = {}
            n = len(results)
            if n == 0: return scores
            if n == 1:
                scores[results[0][doc_id]] = 1.0
                return scores
            for rank, doc in enumerate(results):
                # Hạng đầu tiên (rank=0) -> score = 1.0; Hạng cuối (rank=n-1) -> score = 0.0
                scores[doc[doc_id]] = (n - 1 - rank) / (n - 1)
            return scores

        bm25_scores = get_rank_scores(bm25_res,"doc_id")
        dense_scores = get_rank_scores(dense_res,"id")

        # Tính điểm tổng hợp có trọng số
        combined_scores = {}
        for doc_id in all_docs.keys():
            s_bm25 = bm25_scores.get(doc_id, 0.0)
            s_dense = dense_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = (alpha * s_dense) + ((1 - alpha) * s_bm25)

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
        fetch_limit = limit * 3
        bm25_res = self._bm25_search(query, limit=fetch_limit)
        dense_res = self._dense_search(query, limit=fetch_limit)

        all_docs = {}
                
        for doc in bm25_res:
            d_id=doc.get("doc_id") 
            movie_obj=self.inverted_index.docmap.get(d_id)
            description=movie_obj["description"] if movie_obj is not None else "" 
            metadata=self.inverted_index.docmap.get("metadata",{})
    

            all_docs[d_id]={
                "id": d_id,
                "title": doc.get('title'),
                "document": description[:100] ,
                "metadata": metadata,
                "score":0.0
            }
        for doc in dense_res:
            d_id=doc.get('id')
            all_docs[d_id]=doc
            
        rrf_scores = {}

        # Cộng điểm RRF từ nhánh BM25
        for rank, doc in enumerate(bm25_res):
            doc_id = doc['doc_id']
            # Vị trí hạng thực tế là rank + 1 (vì rank xuất phát từ 0)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        # Cộng dồn điểm RRF từ nhánh Semantic (Dense)
        for rank, doc in enumerate(dense_res):
            doc_id = doc['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        final_results = []
        for doc_id, final_score in sorted_ids:
            final_doc = all_docs[doc_id].copy()
            final_doc['score'] = round(final_score, 4) # Lưu điểm RRF đã làm tròn
            final_results.append(final_doc)
            
        return final_results


#* ==================== CLI COMMAND FUNCTIONS ====================


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
def reciprocal_rank_fusion_search_command(args,searcher):
    console = Console()
    results = searcher.rrf_search(args.query, k=args.k, limit=args.limit)    
    console.print(f"\n[bold cyan]🔍 Running Reciprocal Rank Fusion Hybrid Search for:[/bold cyan] [italic yellow]'{args.query}'[/italic yellow] with k = [italic yellow]'{args.k}'[/italic yellow]and limit = [italic yellow]'{args.limit}'[/italic yellow]...\n")

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