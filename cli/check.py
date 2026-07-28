import pickle
import json
import re
from collections import Counter,defaultdict
from pathlib import Path

from lib.semantic_search import semantic_chunking
from lib.keyword_search import InvertedIndex
#*************************************
# Open the file in binary mode
# with open('cache/docmap.pkl','rb') as file:
#     data = pickle.load(file)

# # View the contents
# print(data)

# matched_ids = set()
# matched_ids.update(set([1, 2, 3, 2, 1])  
# )
# print(matched_ids)

# for value in matched_ids:
#     print(value)
#*************************************
# count=Counter()
# count.update([10,20,30])
# print(count)

# count.update([10,30,40])
# print(count)
# counter=defaultdict(Counter)

# counter[3].update("hello how are you".split())
# print(counter)
# counter[10].update("hello how are you today and hello how are newday".split())
# print(counter)
# print(counter[10]['hello'])
#*************************************
# def load_movie():
#     with open("data/movies.json", "r", encoding="utf-8") as file:
#         movies_data = json.load(file)
#     return movies_data.get("movies", []) if isinstance(movies_data, dict) else movies_data
# movies_list=load_movie()#[dic1,dic2]
# print(type(movies_list))
#*************************************
# term_frequencies={}
# token_text="hello how are you today,i am fine thankyou and how are you today again".split()
# doc_id=1 #film co id la 1
# if 1 not in term_frequencies:
#         term_frequencies[doc_id] = Counter()
#         term_frequencies[doc_id].update(token_text) #{id_film:{'cat':2,dog:1,...}}
# print(f"term_frequencies : {term_frequencies}")
#*************************************
# print(len({"1":1,"2":2}))
#*************************************
# indexer=InvertedIndex()
# indexer.load()
# print(indexer.index.get('hero'))
#*************************************
# print([1,2,3]+[2,3,4])
#*************************************
# text="hello how are you ? I'm find thank you,and you. I'm find,thanks.See yah!"
# token_text=text.split()
# chunk_bottle=[]
# chunk_size=3
# overlap=2
# for i in range(0,len(token_text),chunk_size-overlap):
#     sentence = " ".join(token_text[i : i + chunk_size]) 
#     chunk_bottle.append(sentence[:])
# print(chunk_bottle)    
#*************************************
# text="First sentence here. Second sentence here. Third sentence here. Fourth sentence here"
# # words=re.split(r"(?<=[.!?])\s+", text, maxsplit=0, flags=0)
# # print(words)
# chunks=semantic_chunking(text,2,1)
# print(chunks)
#*************************************

# file_path=Path("cache/index.pkl")
# if file_path.exists():
#     print("The path exists.")
        