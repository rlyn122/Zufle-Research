import pickle
import numpy

with open('nPmatrix20_data.pkl', 'rb') as file:
    nP = pickle.load(file)

with open('nQmatrix20_data.pkl', 'rb') as file:
    nQ = pickle.load(file)

R_est = numpy.dot(nP,nQ.T)
print(R_est)

with open('matrix_data.pkl', 'rb') as file:
    loaded_matrix = pickle.load(file)

err = (loaded_matrix-R_est)

print(err)