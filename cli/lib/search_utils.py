import string
from nltk.stem import PorterStemmer
from pathlib import Path

PROJECT_ROOT=Path(__file__).parent.parent.parent
DATA_PATH=PROJECT_ROOT/"data"/'movies.json'
STOPWORDS_PATH=PROJECT_ROOT/"data"/"stopwords.txt"
CACHE_DIR=PROJECT_ROOT/"cache"
PROMPTS_PATH=PROJECT_ROOT/"cli"/"lib"/"prompts"


PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)

def clean_and_tokenize(text: str) -> list:
    lower_text = text.lower()
    punc_text = lower_text.translate(PUNCTUATION_TABLE)
    return punc_text.split()

def load_stopwords(file_path: str) -> set:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_words = file.read().splitlines()
        
        processed_stopwords = set()
        for word in raw_words:
            tokens = clean_and_tokenize(word)
            if tokens:
                processed_stopwords.add(tokens[0]) 
        return processed_stopwords
    except FileNotFoundError:
        return set()



STOPWORDS = load_stopwords("data/stopwords.txt")
stemmer = PorterStemmer()

def text_processing(text: str) -> list:
    tokens = clean_and_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS]
    clean_tokens = [stemmer.stem(word) for word in tokens]
    return clean_tokens

def format_search_result(doc_id:int,
                         title:str,
                         document:str,
                         score:float,metadata:dict,
                         SCORE_PRECISION:int=4):
    return {
    "id": doc_id,
    "title": title,
    "document": document[:100],
    "score": round(score, SCORE_PRECISION),
    "metadata": metadata or {},
}
    
    