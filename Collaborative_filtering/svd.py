import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import randomized_svd
from mf import MF


with open('../data/test.pkl', 'rb') as file:
    R = pickle.load(file)


mf = MF(R, K=10, alpha=0.01, beta=0.01, iterations=1000)
print("Creating a test set\n")
R1 = MF.zero_out(mf,3)


print("Original:\n",R)
print("Test Set:\n",R1)
mse = mean_squared_error(R1, R)
rmsei = mse**0.5
print("RMSEi=",rmsei)

Rf = mf.svd()
mf.evaluation(R,Rf)


# with open("./output/svd.pkl", 'wb') as file:
#     pickle.dump(predicted, file)