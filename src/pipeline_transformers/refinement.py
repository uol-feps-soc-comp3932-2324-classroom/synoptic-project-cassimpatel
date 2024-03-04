from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# TODO: add error checking

class EpsilonNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, eps):
        self.eps = eps
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[X < self.eps] = 1
        X[X != 1] = 0
        np.fill_diagonal(X,0)
        

        return X