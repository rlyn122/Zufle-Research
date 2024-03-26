#Setting up prerequisites
#from numba import prange
from mf import MF
import pandas as pd
import numpy as np
import math
import re
import sklearn
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset
sns.set_style("darkgrid")
from cvxpy import *
from numpy import matrix
import pickle
from sklearn.metrics import mean_squared_error

print("Setup Complete\n")

file_path = ('../data/test.pkl')
print("\nLoading Data\n")

#load original data
with open(file_path, 'rb') as file:
    R = pickle.load(file)
print("Dimensions of R: ", R.shape)

#initialize MF object and modified matrix
mf = MF(R, K=10, alpha=0.0001, beta=0.01, iterations=1000)
sparsity = mf.calculate_sparsity()
print(f"Sparsity Level: {sparsity:.2%} ")
print("Creating a test set\n")
R1 = MF.zero_out(mf,3)


print("Original:\n",R)
print("Test Set:\n",R1)
mse = mean_squared_error(R1, R)
rmsei = mse**0.5
print("RMSE=",rmsei)

#train matrix factorization
print("\nTraining ...\n")
training_process = mf.train()
Rf = mf.full_matrix()
L = np.rint(Rf)
print("\nDone\n")

# # #analysis
# x = [x for x, y in training_process]
# y = [y for x, y in training_process]
# x = x[::10]
# y = y[::10]
# plt.figure(figsize=((16,4)))
# plt.plot(x, np.sqrt(y))
# plt.xticks(x, x)
# print("Minimizing Error on Training Set:\n")
# plt.xlabel("Iterations")
# plt.ylabel("Root Mean Square Error")
# plt.grid(axis="y")
# plt.show()

print("Original=\n",R)
print("Learnt=\n",Rf)
print("Rounded=\n L")

mf.evaluation(R,Rf)