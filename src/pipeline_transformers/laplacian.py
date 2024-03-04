from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# TODO: add error checking

class LaplacianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, normalize=False):
        if normalize != False:
            raise ValueError(f"Required module parameter has not yet been implemented")
        self.normalize = normalize
    
    def fit(self, X, y=None):
        return self

    def transform(self, A, y=None):
        n = len(A)
        D = np.zeros((n, n))
        d = [np.sum(A[row,:]) for row in range(A.shape[0])]
        np.fill_diagonal(D, d)
        L = D - A

        return L