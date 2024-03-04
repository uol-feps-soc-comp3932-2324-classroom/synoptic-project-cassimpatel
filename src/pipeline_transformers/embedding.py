from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# TODO: add error checking

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method = 'single'):
        if method != 'single':
            raise ValueError(f"Required module parameter has not yet been implemented")
        self.method = method
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # extract the last row as the eigenvalues, take rest as eigenvectors
        eig_val = X[-1 , :]
        eig_vec = X[:-1, :]

        eig_val  = eig_val.argsort()
        z_eigvec = eig_vec[:,eig_val][:,[1]]

        return z_eigvec