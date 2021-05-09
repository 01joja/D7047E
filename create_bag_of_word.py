# https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/

import unidecode
import string
import random
import time
import math
import torch
import re
from pathlib  import Path

# Reading and un-unicode-encoding data

text = Path('shakespere.txt').read_text()

text = text.lower()

tokenized = re.findall(r"\w+|[^\w\s]", text)
index_to_word = []
word_to_index = {}

n_characters = 0
for word in tokenized:
    if not (word in index_to_word):
        index_to_word.append(word)
        word_to_index[word] = n_characters
        n_characters += 1
n_characters += 1


def read_file(filename):
    file = Path(filename).read_text()
    file = file.lower()
    file = re.findall(r"\w+|[^\w\s]", text)
    return file, len(file)

# Turning a string into a tensor

def word_tensor(token):
    tensor = torch.zeros(len(token)).long()
    for c in range(len(token)):
        try:
            tensor[c] = word_to_index[token[c]]
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
