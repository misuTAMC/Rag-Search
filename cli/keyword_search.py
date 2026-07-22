import os
import json
import pickle
from search_utils import text_processing
from collections import Counter

def load_movie():
    with open("data/movies.json", "r", encoding="utf-8") as file:
        movies_data = json.load(file)
    return movies_data.get("movies", []) if isinstance(movies_data, dict) else movies_data

class InvertedIndex:
    '''
    self.index = {
    "merida": {4651},          # Từ 'merida' chỉ xuất hiện ở phim Brave (4651)
    "braveri": {4651, 1024},    # Từ 'bravery' (đã stem thành braveri) xuất hiện ở cả 2 phim
    "lion": {1024}              # Từ 'lion' chỉ xuất hiện ở phim Lion King (1024)
}

    self.docmap = {
    4651: {
        "id": 4651, 
        "title": "Brave", 
        "description": "Princess Merida relies on her bravery..."
    },
    1024: {
        "id": 1024, 
        "title": "The Lion King", 
        "description": "A young lion prince..."
    }
}
    '''
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text): #*doc_id:film's id and text is title+description
        token_text = text_processing(text)
        for token in token_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.term_frequencies[doc_id].update(token_text) #{id_film:{'cat':2,dog:1,...}}


    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def build(self):
        movies = load_movie()
        for movie in movies:
            doc_id = movie['id']
            self.docmap[doc_id] = movie
            
            title = movie.get('title', '')
            description = movie.get('description', '')
            full_text = f"{title} {description}"
            self.__add_document(doc_id, full_text)
            
    def save(self):
        cache_dir = 'cache'
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(cache_dir, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(os.path.join(cache_dir, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)


    def load_from_cache(self) -> bool:
        """Nạp dữ liệu đã build từ ổ cứng lên bộ nhớ."""
        try:
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open("cache/term_frequencies.pkl", "rb") as file:
                self.term_frequencies = pickle.load(file)
            return True
            
        except FileNotFoundError:
            return False

    def search(self, query: str, max_results: int = 5) -> list:

        query_tokens = text_processing(query)
        if not query_tokens:
            return []
            
        matched_ids = set()
        for token in query_tokens:
            # Lấy ra set các ID phim chứa token này
            matched_ids.update(self.index.get(token, set()))#type <class 'set'> {1,2,3,...}
            
        results = []
        for doc_id in sorted(matched_ids):
            # Đảm bảo dừng chính xác khi đủ số lượng kết quả yêu cầu
            if len(results) >= max_results:
                break
                
            if doc_id in self.docmap:
                results.append(self.docmap[doc_id])
                
        return results

    def get_tf(self,doc_id,term):
        if term in self.term_frequencies[doc_id]:
            tf_token=self.term_frequencies[doc_id].get(term, 0)
            return tf_token
        return 0
        
    def load(self):
        """Nạp dữ liệu index và docmap từ đĩa. Ném ra lỗi nếu file không tồn tại."""
        cache_dir = 'cache'
        # Đề bài yêu cầu dùng đúng đường dẫn này
        index_path = f"{cache_dir}/index.pkl"
        docmap_path = f"{cache_dir}/docmap.pkl"
        term_frequencies_path=f"{cache_dir}/term_frequencies.pkl"
        
        if not os.path.exists(index_path) or not os.path.exists(docmap_path):
            raise FileNotFoundError("Index files do not exist. Please run 'build' first.")
            
        with open(index_path, "rb") as file:
            self.index = pickle.load(file)
            
        with open(docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(term_frequencies_path,"rb") as file:
            self.term_frequencies=pickle.load(file)
        