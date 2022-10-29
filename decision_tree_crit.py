import numpy as np 

def gini(p):
    return (p) * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2((1 - p))

def error(p):
    return 1 -np.max([p, (1 - p)])