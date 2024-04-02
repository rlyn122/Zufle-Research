import pandas as pd
import pickle
import csv
import numpy as np
from mf import MF

file_path = "../data/test.pkl"

#load original data
with open(file_path, 'rb') as file:
    R = pickle.load(file)

# print(f"Dimensions of R ({len(R):d}, {len(R[0]):d})")
# print(f"Total size = {R.size:d}")

# matrix = MF(R, K=10, alpha=0.01, beta=0.01, iterations=1000)
# print(f"sparisty : {matrix.calculate_sparsity():.2%}")
# L = R

# nonzeroIndices = np.transpose(np.nonzero(R))
# msef = 0
# for i,j in nonzeroIndices:
#     msef += (R[i,j] - L[i,j])**2
# msef /= len(nonzeroIndices)*len(nonzeroIndices[0])
# print("number of nonzero values: ", len(nonzeroIndices)*len(nonzeroIndices[0]))
# print(msef)
