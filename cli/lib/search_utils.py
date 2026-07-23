import string
from nltk.stem import PorterStemmer

def clean_and_tokenize(text: str) -> list:
    lower_text = text.lower()
    punc_text = lower_text.translate(str.maketrans("", "", string.punctuation))
    return punc_text.split()

def load_stopwords(file_path: str) -> list:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_words = file.read().splitlines()
        
        processed_stopwords = []
        for word in raw_words:
            tokens = clean_and_tokenize(word)
            if tokens:
                processed_stopwords.append(tokens[0])
        return processed_stopwords
    except FileNotFoundError:
        return []

# Khởi tạo các tài nguyên dùng chung
STOPWORDS = load_stopwords("data/stopwords.txt")
stemmer = PorterStemmer()

def text_processing(text: str) -> list:
    tokens = clean_and_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS]
    clean_tokens = [stemmer.stem(word) for word in tokens]
    return clean_tokens
