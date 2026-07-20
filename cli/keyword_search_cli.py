import argparse
import json
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
                processed_stopwords.append(tokens[0]) # Lấy từ đã làm sạch
        return processed_stopwords
    except FileNotFoundError:
        return []

STOPWORDS = load_stopwords("data/stopwords.txt")
stemmer = PorterStemmer()


def text_processing(text: str) -> list:
    tokens = clean_and_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS]
    clean_tokens =[stemmer.stem(word) for word in tokens]
    return clean_tokens
    
# 4. Thuật toán tìm kiếm từ khóa khớp một phần
def keyword_searching(movies: list, query: str, max_results: int = 5) -> list:
    query_tokens = text_processing(query)
    if not query_tokens:
        return []

    results = []
    for movie in movies:
        title_tokens = text_processing(movie['title'])
        has_match = any(q_token in t_token for q_token in query_tokens for t_token in title_tokens)
        
        if has_match:
            results.append(movie)
    return results[:max_results]

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            with open("data/movies.json", "r", encoding="utf-8") as file:
                movies_data = json.load(file)
            
            movie_list = movies_data.get('movies', []) if isinstance(movies_data, dict) else movies_data
                
            match_movies = keyword_searching(movie_list, args.query, max_results=5)
            for index, movie in enumerate(match_movies, start=1):
                print(f"{index}. {movie['title']}")
                
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
