
import torch
import pickle

print("Load 1e-3 network")
with open('1e-3', 'rb') as handle:
    network = pickle.load(handle)

print(network)

network = nn.Sequential(*[network[i] for i in range(11)])

print(network)