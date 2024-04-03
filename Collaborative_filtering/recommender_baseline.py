from mf import MF
from cvxpy import *
import pickle

print("Setup Complete\n")

file_path = ('../data/NYfiltered.pkl')
print("\nLoading Data\n")

#load original data
with open(file_path, 'rb') as file:
    R = pickle.load(file)

mf = MF(R, K=10, alpha=0.0001, beta=0.01, iterations=100)
#run matrix factorization
mf.run_sgd()
mf.evaluation()