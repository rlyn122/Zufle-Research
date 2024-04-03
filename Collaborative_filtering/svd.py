import pickle
from mf import MF


with open('../data/NYfiltered.pkl', 'rb') as file:
    R = pickle.load(file)

mf = MF(R, K=10, alpha=0.01, beta=0.01, iterations=100)
print("running svd")
Rf = mf.svd()
print("svd finished")
mf.evaluation()
