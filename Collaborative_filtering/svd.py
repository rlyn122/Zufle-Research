import numpy as np
import pickle
from sklearn.utils.extmath import randomized_svd

with open('../data/pkl/NYfiltered.pkl', 'rb') as file:
    R = pickle.load(file)

K = 10

u,s,vh = randomized_svd(R,K,n_iter = 10)

S = np.zeros((u.shape[1], vh.shape[0]))
np.fill_diagonal(S,s)

predicted = np.dot(u,np.dot(S,vh))

with open("./output/svd.pkl", 'wb') as file:
    pickle.dump(predicted, file)