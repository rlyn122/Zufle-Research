import pickle
from mf import MF


with open('../data/20plusfiltered.pkl', 'rb') as file:
    R = pickle.load(file)

mf = MF(R, K=10, alpha=0.01, beta=0.01, iterations=20)
print("running svd")
Rf = mf.rand_svd()
print("svd finished")
mf.evaluation()



