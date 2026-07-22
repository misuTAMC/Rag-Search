import pickle
from collections import Counter,defaultdict
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

# count=Counter()
# count.update([10,20,30])
# print(count)

# count.update([10,30,40])
# print(count)

counter=defaultdict(Counter)

# counter[3].update("hello how are you".split())
# print(counter)
counter[10].update("hello how are you today and hello how are newday".split())
print(counter)
print(counter[10]['hello'])