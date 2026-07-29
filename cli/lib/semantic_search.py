import json
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
from lib.keyword_search import load_movie
from lib.search_utils import format_search_result

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


CACHE_DIR="cache"


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model=SentenceTransformer(model_name)
        self.embeddings=None
        self.documents=None
        self.document_map={}
    
    def generate_embedding(self,text):
        '''To check if an input string is empty or contains only whitespace 
        and raise a ValueError, use Python's built-in strip() method or isspace() method.'''
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or contain only whitespace.")
        text_embedding=self.model.encode([text],convert_to_numpy=True)#encode input is list or tup
        
        return text_embedding[0]
    def build_embeddings(self,documents):#documents list of dics,dic:movie
        self.documents=documents
        for doc in documents:
            self.document_map[doc["id"]]=doc
            
        movie_strings=[f"{doc['title']}: {doc['description']}" for doc in documents]
        
        print("Generating embeddings... This might take a while.")
        self.embeddings=self.model.encode(movie_strings,show_progress_bar=True,convert_to_numpy=True)
        
        os.makedirs(CACHE_DIR,exist_ok=True)
        np.save(os.path.join(CACHE_DIR, 'movie_embeddings.npy'), self.embeddings)
        with open(CACHE_DIR+"/chunk_metadata.json", "w",encoding='utf-8') as f:
                    json.dump(self.document_map, f,ensure_ascii=False)
        return self.embeddings
        
        
    def load_or_create_embeddings(self,documents):
        self.documents=documents
        
        for doc in documents:
            self.document_map[doc["id"]]=doc
        
        cache_path=CACHE_DIR+'/movie_embeddings.npy'
        
        if os.path.exists(cache_path):
            print("Loading embeddings from cache...")
            self.embeddings=np.load(cache_path)
            
            
            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                print("Cache mismatch detected. Rebuilding...")
        
        return self.build_embeddings(documents)
    def search(self,query,limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
            
        embed_query=self.generate_embedding(query)
        cosine_score_list=[]
        
        for doc, doc_emb in zip(self.documents, self.embeddings):
                cosine_score = cosine_similarity(embed_query, doc_emb)
                cosine_score_list.append((cosine_score, doc))
                
        
        cosine_score_list.sort(key=lambda x: x[0], reverse=True)
        
        top_cosine_score_list=[]
        for score,doc in cosine_score_list[:limit]:
            top_cosine_score_list.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })
        return top_cosine_score_list
    
    
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None        
        
    def build_chunk_embeddings(self,documents:list[dict])->np.ndarray:
        # 1. Populate self.documents and self.document_map
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
            
        all_chunks=[]
        chunk_metadata=[]
                
        for movie_idx,doc in enumerate(documents):
            if not doc.get("description"):
                continue
            chunks=semantic_chunking(
                doc["description"],
                max_chunk_size=4,
                overlap=1
            )
            
            all_chunks.extend(chunks)
            total_chunks=len(chunks)
            for chunk_idx,_ in enumerate(chunks):
                chunk_metadata.append(
                    {
                    "movie_idx": movie_idx,       
                    "chunk_idx": chunk_idx,       
                    "total_chunks": total_chunks   
                    }
                )
        if all_chunks:
            self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
        else:
            self.chunk_embeddings=np.empty((0, 0), dtype=np.float32)
        
        self.chunk_metadata=chunk_metadata
        
        
        # 5. Tạo thư mục cache và lưu file npy
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(os.path.join(CACHE_DIR, 'chunk_embeddings.npy'), self.chunk_embeddings)
            
        metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")
        with open(metadata_path, "w",encoding="utf-8") as f:
            json.dump({
                "chunks": self.chunk_metadata, 
                "total_chunks": len(all_chunks)
                }, f, indent=2)
        
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self,documents:list[dict])->np.ndarray:
            #Populate self.documents and self.document_map from the input documents 
            self.documents=documents
            self.document_map={doc["id"]:doc for doc in documents}
            
            chunk_embedding_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
            chunk_metadata_path  = os.path.join(CACHE_DIR, "chunk_metadata.json")
            
            
            if os.path.isfile(chunk_embedding_path) and os.path.isfile(chunk_metadata_path):
                self.chunk_embeddings=np.load(chunk_embedding_path)
                with open(chunk_metadata_path, 'r', encoding='utf-8') as file:
                    cached_data=json.load(file)
                    
                    self.chunk_metadata=cached_data.get("chunks",[])
            
                # ĐỌC FILE VECTOR LÊN ĐỂ KIỂM TRA ĐỘ DÀI MA TRẬN
                self.chunk_embeddings = np.load(chunk_embedding_path)
                
                # ĐIỀU KIỆN BẢO HIỂM: Kiểm tra xem số lượng vector đã lưu trong cache 
                # có TRÙNG KHỚP với trường total_chunks trong file JSON hay không.
                # Nếu thầy cô đổi file dữ liệu, hai con số này hoặc số lượng phim sẽ lệch nhau ngay!
                if len(self.chunk_embeddings) == cached_data.get("total_chunks", 0):
                    self.chunk_metadata = cached_data.get("chunks", [])
                    return self.chunk_embeddings
                else:
                    print("Cache mismatch detected (Dữ liệu đầu vào đã thay đổi). Rebuilding...")
            
            return self.build_chunk_embeddings(self.documents)
                
    def search_chunks(self,query:str,limit:int=10):
        
        
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        
        
        embed_query=self.generate_embedding(query)
        chunk_scores=[]
        
        for idx,chunk_embedding in enumerate(self.chunk_embeddings):
            cosine_score=cosine_similarity(embed_query,chunk_embedding)
            extract_metadata=self.chunk_metadata[idx]
            chunk_scores.append(
                {
                    'chunk_idx':extract_metadata['chunk_idx'],
                    'movie_idx':extract_metadata['movie_idx'],
                    'score':float(cosine_score) 
                }
            )
        # 4. GOM NHÓM (Aggregation): Tạo dictionary rỗng ánh xạ movie_idx -> best chunk score
        best_movie_scores={}
        for item in chunk_scores:
            m_idx=item['movie_idx']
            if m_idx not in best_movie_scores:
                best_movie_scores[m_idx]=item
            else:
                if item['score']>best_movie_scores[m_idx]['score']:
                    best_movie_scores[m_idx]=item
        # 5. SẮP XẾP: Chuyển dict thành list và xếp giảm dần theo trường 'score'
        best_movie_list=list(best_movie_scores.values())#.values() để lấy danh sách Object dict
        best_movie_list.sort(key=lambda x: x['score'],reverse=True)
        
        
        top_movie_scores=best_movie_list[:limit]
        final_results = []
        SCORE_PRECISION=4
        for item in top_movie_scores:
            movie_position=item['movie_idx']
            score=item['score']
            
            doc=self.documents[movie_position]
            
            doc_id=doc.get('id',-1)
            title=doc.get('title','')
            description=doc.get('description','')
            
            metadata=doc.get('metadata',{})
            
            final_results.append(format_search_result(
                doc_id,
                title,
                description,
                score,
                metadata,
                SCORE_PRECISION
            ))
            
        return final_results
                
                
                
                
            
            
            
                
 #*===============Function=======================

def search_chunked_command(args):
    console = Console()
    
    movies = load_movie()
    search_engine = ChunkedSemanticSearch()
    search_engine.load_or_create_chunk_embeddings(movies)
    results = search_engine.search_chunks(query=args.query, limit=args.limit) 
    
    console.print(f"\n[bold cyan]🔍 Semantic Search Results for:[/bold cyan] [italic yellow]'{args.query}'[/italic yellow]\n")

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

def search(query,limit=5):
    ss=SemanticSearch()
    movies=load_movie()
    ss.load_or_create_embeddings(movies)
    movie_lists=ss.search(query,limit)
    print("-"*50)
    for idx,movie in enumerate(movie_lists):
        
        print(f"{idx}. {movie['title']} (score: {movie['score']:.2f})")
        print(movie["description"][:100])
        print("-"*50)
        

def verify_embeddings():
    ss=SemanticSearch()
    
    documents=load_movie()
    
    embeddings=ss.load_or_create_embeddings(documents)
    
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
    
def verify_model():
    search_instance=SemanticSearch()
    
    print(f"Model loaded : {search_instance.model}")
    print(f"Max sequence length : {search_instance.model.max_seq_length}")
    
def embed_text(text):
    ss=SemanticSearch()
    embed_result=ss.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embed_result[:3]}")
    print(f"Dimensions: {embed_result.shape[0]}")
    
def embed_query_text(query):
    ss=SemanticSearch()
    vector_embedding=ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {vector_embedding[:3]}")
    print(f"Shape: {vector_embedding.shape}")
    
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def fixed_sized_chunking(
    text: str, chunk_size: int = 200, overlap: int = 0
):
    words = text.split()

    if not words:
        return []

    if overlap < 0:
        overlap = 0

    if overlap >= chunk_size:
        overlap = chunk_size - 1

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

        # Điều kiện dừng: Nếu vị trí kết thúc của chunk hiện tại đã bao phủ hết từ cuối cùng
        # thì dừng luôn, không để vòng lặp chạy tiếp tạo ra các chunk thừa phía sau
        if i + chunk_size >= len(words):
            break

    return chunks


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0):
    chunks = fixed_sized_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")

def semantic_chunking(text:str,max_chunk_size:int=4,overlap:int=0):
    sentences=re.split(r"(?<=[.!?])\s+", text)
    
    #*remove cac chuoi rong if exists
    sentences=[s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    if overlap<0:
        overlap=0
    if overlap >= max_chunk_size:
        overlap = max_chunk_size - 1
    chunks = []
    step = max_chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + max_chunk_size])
        chunks.append(chunk)

        # Điều kiện dừng: Nếu vị trí kết thúc của chunk hiện tại đã bao phủ hết từ cuối cùng
        # thì dừng luôn, không để vòng lặp chạy tiếp tạo ra các chunk thừa phía sau
        if i + max_chunk_size >= len(sentences):
            break

    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0):
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")
