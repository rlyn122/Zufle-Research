import numpy as np
from scipy.sparse import coo_matrix
import pickle

def evaluate_factorization(actual, predicted):
    """
    Evaluate matrix factorization quality using RMSE and MAE.
    
    Parameters:
    - R: scipy.sparse.coo_matrix, original sparse ratings matrix.
    - P: numpy.ndarray, user feature matrix from matrix factorization.
    - Q: numpy.ndarray, item feature matrix from matrix factorization.
    
    Returns:
    - rmse: float, root mean square error.
    - mae: float, mean absolute error.
    """
    

    errors = predicted - actual
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(errors**2))
    
    # Calculate MAE
    mae = np.mean(np.abs(errors))
    
    return rmse, mae

#SVD ERROR CALCULATION
with open('./output/svd.pkl','rb') as file:
    R_svd = pickle.load(file)

with open('../data/pkl/NYfiltered.pkl', 'rb') as file:
    R = pickle.load(file)
    
rmse, mae = evaluate_factorization(R, R_svd)
print(f"SVD Factorization: ")
print(f"RMSE: {rmse}")
print(f"MAE: {mae} \n")

#SGD ERROR CALCULATION
with open('./output/sgd.pkl', 'rb') as file:
    R_sgd = pickle.load(file)

rmse, mae = evaluate_factorization(R, R_sgd)
print(f"SGD Factorization: ")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")