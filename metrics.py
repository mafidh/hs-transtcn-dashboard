import numpy as np

def MAE(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def RMSE(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))