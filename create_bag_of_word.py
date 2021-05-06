# https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/


import re  

f = open("shakespere.txt", "r")
word_to_index = {}
index_to_word = []

for line in f:
    line = line.lower()
    line = re.sub(r'\W',' ',line)
    line = re.sub(r'\s+',' ',line)
    sentence = line.split()
    j = 0
    for word in sentence:
        if word not in word_to_index.keys():
            word_to_index[word] = j
            index_to_word.append(word)
            j +=1

print(word_to_index.keys()) 
print(len(index_to_word))
