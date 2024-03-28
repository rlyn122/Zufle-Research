import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import randomized_svd
from mf import MF


with open('../data/test.pkl', 'rb') as file:
    R = pickle.load(file)


mf = MF(R, K=10, alpha=0.01, beta=0.01, iterations=1000)
Rf = mf.svd()
Rf = mf.evaluation()


# with open("./output/svd.pkl", 'wb') as file:
#     pickle.dump(predicted, file)