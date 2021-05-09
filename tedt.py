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

print(index_to_word[560])
print(word_to_index["the"])