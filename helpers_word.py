# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch
# https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/

import re  

f = open("shakespere.txt", "r")
# saves all words in the file in words_in_file
words_in_file = []
word_to_index = {}
index_to_word = []
j = 0

for line in f:
    line = line.lower()
    #line = re.sub(r'\W',' ',line)
    sentence = line.split()
    for word in sentence:
        if not (word in word_to_index.keys()):
            word_to_index[word] = j
            index_to_word.append(word)
            j +=1
        words_in_file.append(word)

# Reading and un-unicode-encoding data

print(word_to_index.keys())
n_characters = len(index_to_word)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Turning a string into a tensor

def char_tensor(words):
    tensor = torch.zeros(len(words)).long()
    for c in range(len(words)):
        try:
            tensor[c] = word_to_index[words[c]]
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

print(word_to_index['eyelids'])
print(char_tensor(['eyelids']))
print(char_tensor(["a"]))