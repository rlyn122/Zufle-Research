import pandas as pd
import pickle
import csv
import numpy as np
from mf import MF

file_path = "../data/BrooklynFiltered.pkl"
file_path2 = "../data/NYfiltered2.pkl"
file_path3 = "../data/test.pkl"

#load original data
with open(file_path, 'rb') as file:
    R1 = pickle.load(file)
    RBrooklyn = MF(R, K=10, alpha=0.01, beta=0.01, iterations=1000)
with open(file_path2, 'rb') as file:
    R2 = pickle.load(file)
    RNY = MF(R, K=10, alpha=0.01, beta=0.01, iterations=1000)
with open(file_path3, 'rb') as file:
    R3 = pickle.load(file)
    Rtest = MF(R, K=10, alpha=0.01, beta=0.01, iterations=1000)

print("R Brooklyn:\n")
print(f"sparisty : {RBrooklyn.calculate_sparsity(R1):.2%}\n")
print(f"Dimensions of R ({len(RBrooklyn.Ri):d}, {len(RBrooklyn.Ri[0]):d})")

print("R NY :\n")
print(f"sparisty : {RNY.calculate_sparsity(R2):.2%}\n")
print(f"Dimensions of R ({len(RNY.Ri):d}, {len(RNY.Ri[0]):d})")

print("R test :\n")
print(f"sparisty : {Rtest.calculate_sparsity(R3):.2%}\n")
print(f"Dimensions of R ({len(Rtest.Ri):d}, {len(Rtest.Ri[0]):d})")



# print(f"Dimensions of R ({len(R):d}, {len(R[0]):d})")
# print(f"Total size = {R.size:d}")
# print(f"sparisty : {matrix.calculate_sparsity():.2%}")
# L = R

# nonzeroIndices = np.transpose(np.nonzero(R))
# msef = 0
# for i,j in nonzeroIndices:
#     msef += (R[i,j] - L[i,j])**2
# msef /= len(nonzeroIndices)*len(nonzeroIndices[0])
# print("number of nonzero values: ", len(nonzeroIndices)*len(nonzeroIndices[0]))
# print(msef)
