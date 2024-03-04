from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# TODO: add error checking

class DecompositionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method = 'dense'):
        if method != 'dense':
            raise ValueError(f"Required module parameter has not yet been implemented")
        self.method = method
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        eig_val, eig_vec = np.linalg.eig(X)
        eig_val = eig_val.real
        eig_vec = eig_vec.real

        # manually append eigenvalues column to eigenvectors
        return np.c_[eig_vec, eig_val]