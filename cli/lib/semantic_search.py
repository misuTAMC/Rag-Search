import json

from sentence_transformers import SentenceTransformer
import numpy as np
import os

from lib.keyword_search import load_movie
CACHE_DIR="cache"


class SemanticSearch:
    def __init__(self):
        self.model=SentenceTransformer("all-MiniLM-L6-v2")
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
        np.save(CACHE_DIR+'/movie_embeddings.npy',self.embeddings)
        
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

def fixed_sized_chunking(text:str,chunk_size:int=200):
    words = text.split()

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks
def chunk_text(text,chunk_size:int=200):
    chunks=fixed_sized_chunking(text,chunk_size)
    print(f"Chunking {len(text)} characters")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")

    