import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from mpl_toolkits import mplot3d # 3D plotting
from matplotlib import cm # colour map

# save and load from disk
def load_data(path: str) -> dict:
    # Open the file in binary mode
    with open(path, 'rb') as file:
        # Call load method to deserialze
        data = pickle.load(file)
    return data

def save_data(data: dict, path: str) -> None:
    pickling_on = open(path,'wb')
    pickle.dump(data, pickling_on)
    pickling_on.close()

def sigmoid(y: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(y)))

# grabs parameters from their place in the genotype
def take_genes(genotype: np.ndarray, start: int, number: int) -> np.ndarray and int:
    new_start = start + number
    return genotype[start: start + number], new_start


## unpacking gene
def unpack_genotype(genotype: np.ndarray, nodes: dict[int]) -> dict:
    # start with gene position 0
    pos = 0

    # take appropriate chunks from array incrementally
    W1, pos = take_genes(genotype, pos, nodes['hidden'] * nodes['input'])
    W2, pos = take_genes(genotype, pos, nodes['hidden'] * nodes['output'])
    B1, pos = take_genes(genotype, pos, nodes['hidden'])
    B2, pos = take_genes(genotype, pos, nodes['output'])

    # scale the values of the weights from [0, 1] to [-10, 10]
    # w = -10 + w * 20
    unpacked_genotype = {
            'W1': W1.reshape(nodes['hidden'], nodes['input']),
            'W2': W2.reshape(nodes['output'], nodes['hidden']),
            'B1': B1.reshape(nodes['hidden'], 1),
            'B2': B2.reshape(nodes['output'], 1),
            }
    return unpacked_genotype


# generate a random agent genotype
def get_random_genotype(nodes: dict[str:int])-> np.ndarray:

    # get the require lengths of the paramters of the network
    W1 = nodes['hidden'] *  nodes['input']
    W2 = nodes['hidden'] *  nodes['output']
    B1 = nodes['hidden']
    B2 = nodes['output']

    genes_n = W1 + W2 + B1 + B2
    genotype = np.random.uniform(0, 1, genes_n)#  randomaly generate weights and baises

    return genotype
