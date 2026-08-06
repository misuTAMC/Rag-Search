from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.keyword_search import load_movie
from lib.semantic_search import cosine_similarity
import numpy as np
class MultimodalSearch:
    def __init__(self,list_of_docs=None,model_name="clip-ViT-B-32"):
        self.model=SentenceTransformer(model_name)
        self.list_of_docs = list_of_docs if list_of_docs is not None else load_movie()
        self.texts = [
            f"{doc.get('title', '')}: {doc.get('description', '')}" 
            for doc in self.list_of_docs
        ]
        print("Encoding movie text metadata into CLIP space...")
        self.text_embeddings = self.model.encode(self.texts, 
                                                 show_progress_bar=True,
                                                 convert_to_numpy=True,
                                                 normalize_embeddings=True)
        
        
    def embed_image(self,image_path:str):
        img = Image.open(image_path)
        embeddings = self.model.encode([img],convert_to_numpy=True,normalize_embeddings=True)
        return embeddings[0] # Lấy phần tử đầu tiên để thu được mảng 1D thô
    def search_with_image(self,image_path:str):
        img_embedding = self.embed_image(image_path).flatten()
        img_norm = np.linalg.norm(img_embedding)
        results=[]
        for idx, doc in enumerate(self.list_of_docs):
            text_embedding = self.text_embeddings[idx].flatten()
            
            cosine_score = cosine_similarity(img_embedding, text_embedding)
            
            results.append({
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
                "similarity_score": float(cosine_score)
            })
        sorted_results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
        return sorted_results[:5]
        
def verify_image_embedding(image_path:str)->None:
    searcher=MultimodalSearch()
    
    embedding=searcher.embed_image(image_path)
    
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
    
def image_search_command(image_path: str) -> list[dict]:
    searcher = MultimodalSearch()
    results = searcher.search_with_image(image_path)
    return results