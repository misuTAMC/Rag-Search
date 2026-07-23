import os
import json
import pickle
from lib.search_utils import STOPWORDS, text_processing
from collections import Counter
import math

def load_movie():
    with open("data/movies.json", "r", encoding="utf-8") as file:
        movies_data = json.load(file)
    return movies_data.get("movies", []) if isinstance(movies_data, dict) else movies_data


BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR='cache'

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        
        self.term_frequencies = {}
        self.doc_lengths={}
        
        self.doc_lengths_path=os.path.join(CACHE_DIR,"doc_lengths.pkl")
        self.index_path=os.path.join(CACHE_DIR,"index.pkl")
        self.docmap_path=os.path.join(CACHE_DIR,"docmap.pkl")
        self.term_frequencies_path=os.path.join(CACHE_DIR,"term_frequencies.pkl")
    

    def __add_document(self, doc_id, text):
        token_text = text_processing(text)
        sequence_length = len(token_text)
        
        for token in token_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.term_frequencies[doc_id].update(token_text)
        
        self.doc_lengths[doc_id] = sequence_length

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        
        total_length = sum(self.doc_lengths.values())
        return float(total_length / len(self.doc_lengths))

        
    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def build(self):
        movies_list = load_movie()
        for movie in movies_list:
            doc_id = movie['id']
            self.docmap[doc_id] = movie
            
            title = movie.get('title', '')
            description = movie.get('description', '')
            full_text = f"{title} {description}"
            self.__add_document(doc_id, full_text)
            
    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if (not os.path.exists(self.index_path) or 
            not os.path.exists(self.docmap_path) or 
            not os.path.exists(self.term_frequencies_path) or
            not os.path.exists(self.doc_lengths_path)):
            raise FileNotFoundError("Index files do not exist. Please run 'build' first.")
            
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file) 
        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)
        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)


    def search(self, query: str, max_results: int = 5) -> list:
        query_tokens = text_processing(query)
        if not query_tokens:
            return []
            
        matched_ids = set()
        for token in query_tokens:
            token_key_from_film = self.index.get(token, set())
            matched_ids.update(token_key_from_film)
            
        results = []
        for doc_id in sorted(matched_ids):
            if len(results) >= max_results:
                break
            if doc_id in self.docmap:
                results.append(self.docmap[doc_id])
                
        return results

    def get_tf(self, doc_id, term) -> int:
        if doc_id in self.term_frequencies:
            return self.term_frequencies[doc_id].get(term, 0)
        return 0

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        if total_doc_count == 0:
            return 0.0
        term_match_doc_count = len(self.index.get(term, set()))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        if N == 0:
            return 0.0
        doc_frequency = len(self.index.get(term, set()))
        
        try:
            bm25_score = math.log((N - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1)
            return bm25_score
        except ValueError:   
            return 0.0
    def get_bm25_tf(self,doc_id,term,k1=BM25_K1,b=BM25_B):
        
        raw_tf=self.get_tf(doc_id,term)
        
        doc_length=self.doc_lengths.get(doc_id,0)
        
        avg_doc_length=self.__get_avg_doc_length()
        
        if avg_doc_length==0:
            length_norm=0
        else:
            length_norm=1-b+b*(doc_length/avg_doc_length)
        
        tf_component = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        
        return tf_component
        

# ==================== CLI COMMAND FUNCTIONS ====================

def bm25_idf_command(term: str) -> float:
    indexer = InvertedIndex()
    indexer.load()
    token_term = tokenize_single_term(term)
    return indexer.get_bm25_idf(token_term)
def bm25_tf_command(doc_id:int,term:str,k1:float=BM25_K1,b:float=BM25_K1)->float:
    indexer=InvertedIndex()
    indexer.load()
    
    token_term=tokenize_single_term(term)
    bm25_tf_score=indexer.get_bm25_tf(doc_id,token_term,k1,b)
    
    return bm25_tf_score
    
    
def build_command():
    print("Building inverted index...")
    indexer = InvertedIndex()
    indexer.build()
    indexer.save()
    print("Index saved to disk successfully.")

def load_command():
    indexer = InvertedIndex()
    indexer.load()
    print("The movie list successfully loaded")
    
def search_command(query):
    print(f"Searching for: {query}")
    indexer = InvertedIndex()
    try:
        indexer.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 
    
    match_movies = indexer.search(query, max_results=5)
    if not match_movies:
        print("No movies found matching your query.")
        return

    for index, movie in enumerate(match_movies, start=1):
        print(f"{index}. [{movie['id']}] {movie['title']}")

def tf_command(term, doc_id):
    indexer = InvertedIndex()
    try:
        indexer.load()
        target_token = tokenize_single_term(term)
        return indexer.get_tf(doc_id, target_token)
    except FileNotFoundError as e:
        print(f"Error : {e}")
    return 0
    
def idf_command(term):
    indexer = InvertedIndex()
    try:
        indexer.load()
        target_token = tokenize_single_term(term)
        return indexer.get_idf(target_token)
    except FileNotFoundError as e:
        print(f"Error : {e}")
    return 0  

def tfidf_command(term, doc_id):
    indexer = InvertedIndex()
    try:
        indexer.load()
        target_token = tokenize_single_term(term)
        
        tf_score = indexer.get_tf(doc_id, target_token)
        idf_score = indexer.get_idf(target_token)
        
        return tf_score * idf_score
    except FileNotFoundError as e:
        print(f"Error : {e}")
    except ValueError as e:
        print(f"Error : {e}")
    return 0
    
def tokenize_single_term(term: str) -> str:
    if term.lower() in STOPWORDS:
        raise ValueError(f"The term '{term}' is a stopword and has been filtered out by the system.")
        
    tokens = text_processing(term)
    if not tokens:
        raise ValueError(f"The term '{term}' did not produce any valid tokens after text processing.")
    if len(tokens) > 1:
        raise ValueError(f"Expected exactly 1 single term, but got {len(tokens)} tokens from '{term}'.")
        
    return tokens[0]

