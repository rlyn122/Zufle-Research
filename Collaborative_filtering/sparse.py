from scipy.sparse import csr_matrix
import pickle
import numpy as np

file_path = "../data/BrooklynFiltered.pkl"
file_path2 = "../data/NYfiltered2.pkl"

#load original data
with open(file_path, 'rb') as file:
    R = pickle.load(file)

R_sparse = csr_matrix(R)

R_sLIL = R_sparse.tolil()
rows = R_sLIL.shape[0]
modifiedInd = []

#replace five indices with zero
for row_index in range(rows):
    print(R_sLIL[row_index])

nonzero_indices = (np.transpose(R_sparse.nonzero())[0])
print(nonzero_indices)

five_nonzero_indices = np.random.choice(nonzero_indices,5)
print(five_nonzero_indices)
#delete 5 from each row
R_sparse[five_nonzero_indices] = 0
# for i in R_sparse:
#     print(i)