from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy

# TODO: add error checking

class DecompositionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method = 'dense'):
        if method not in ['dense', 'dense_eigh', 'sparse', 'sparse_eigh']:
            raise ValueError(f"Required module parameter has not yet been implemented")
        self.method = method
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.method == 'dense':
            eig_val, eig_vec = np.linalg.eig(X)
        elif self.method == 'dense_eigh':
            eig_val, eig_vec = np.linalg.eigh(X)
        elif self.method == 'sparse':
            eig_val, eig_vec = scipy.sparse.linalg.eigs(X, which='SM')
        elif self.method == 'sparse_eigh':
            eig_val, eig_vec = scipy.sparse.linalg.eigsh(X, which='SM')

        eig_val = eig_val.real
        eig_vec = eig_vec.real

        # manually stack the eigenvalues below the eigenvector matrix
        return np.vstack((eig_vec, eig_val))