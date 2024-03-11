from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# TODO: add error checking

class LaplacianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, normalize=False):
        self.normalize = normalize
    
    def fit(self, X, y=None):
        return self

    def transform(self, A, y=None):
        n = len(A)

        # calculate the degree matrix
        D = np.zeros((n, n))
        d = np.count_nonzero(A, axis = 0)
        np.fill_diagonal(D, d)

        # calculate simple unnormalised laplacian
        L = D - A

        if not self.normalize:
            return L

        # normalise the laplacian: 
        d2 = 1 / np.sqrt(np.diag(D))
        np.fill_diagonal(D, d2)
        L = np.matmul(np.matmul(D, L), D)
        return L